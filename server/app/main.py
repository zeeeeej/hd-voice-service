"""应用入口：lifespan 加载模型、中间件（X-Request-Id + 指标）、异常映射、路由注册。"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from . import metrics
from .api import rest, ws_asr, ws_tts
from .config import load_settings
from .errors import VoiceError
from .logging_conf import request_id_var, setup_logging
from .models import ModelHub

log = logging.getLogger(__name__)


def create_app(settings=None, hub=None) -> FastAPI:
    """settings/hub 可注入（测试用）；生产为 None → 加载配置与真实模型。"""
    settings = settings or load_settings()
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.asr_sem = asyncio.Semaphore(settings.limits.asr_sessions)
        app.state.tts_sem = asyncio.Semaphore(settings.limits.tts_sessions)
        if hub is None:
            real_hub = ModelHub(settings)
            app.state.hub = real_hub
            log.info("loading models from %s ...", settings.models_dir)
            await asyncio.to_thread(real_hub.load)  # fail fast：加载失败直接退出
        else:
            app.state.hub = hub
        log.info("service ready")
        yield

    app = FastAPI(title="hd-voice-service", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metrics.errors_total.labels(path=request.url.path).inc()
            raise
        metrics.rest_requests_total.labels(path=request.url.path, status=response.status_code).inc()
        response.headers["X-Request-Id"] = rid
        if response.status_code >= 500:
            metrics.errors_total.labels(path=request.url.path).inc()
        log.info("%s %s -> %d (%.0fms)", request.method, request.url.path,
                 response.status_code, (time.perf_counter() - t0) * 1000)
        return response

    @app.exception_handler(VoiceError)
    async def voice_error_handler(request: Request, exc: VoiceError):
        return JSONResponse(status_code=exc.status,
                            content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        metrics.errors_total.labels(path=request.url.path).inc()
        return JSONResponse(status_code=500,
                            content={"error": {"code": "internal_error", "message": "internal server error"}})

    # pydantic 校验失败 → 400（不得 500）
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:3])
        return JSONResponse(status_code=400,
                            content={"error": {"code": "bad_request", "message": f"invalid request: {msg}"}})

    app.include_router(ws_asr.router)
    app.include_router(ws_tts.router)
    app.include_router(rest.router)

    # /metrics 直接委托 Prometheus ASGI app（避免 mount 产生 307，抓取器不跟随重定向）
    metrics_app = make_asgi_app()

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint(request: Request):
        status_headers: dict = {}
        body_chunks: list[bytes] = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                status_headers["status"] = message["status"]
                status_headers["headers"] = message["headers"]
            elif message["type"] == "http.response.body":
                body_chunks.append(message.get("body", b""))

        await metrics_app(request.scope.copy(), receive, send)
        from fastapi.responses import Response
        headers = {k.decode(): v.decode() for k, v in status_headers.get("headers", [])}
        return Response(content=b"".join(body_chunks),
                        status_code=status_headers.get("status", 200), headers=headers)

    return app


app = create_app()
