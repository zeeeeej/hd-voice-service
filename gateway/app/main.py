"""Board-facing voice-command orchestration gateway."""
from __future__ import annotations

import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import Depends, FastAPI, File, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

log = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class Settings:
    gateway_api_key: str
    voice_api_key: str
    voice_base_url: str
    max_upload_mb: int = 25
    connect_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 180.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            gateway_api_key=os.environ.get("GATEWAY_API_KEY", "devkey-gateway"),
            voice_api_key=os.environ.get("VOICE_API_KEY", "devkey-local"),
            voice_base_url=os.environ.get("VOICE_BASE_URL", "http://voice:8090").rstrip("/"),
            max_upload_mb=int(os.environ.get("GATEWAY_MAX_UPLOAD_MB", "25")),
            connect_timeout_seconds=float(os.environ.get("GATEWAY_CONNECT_TIMEOUT", "10")),
            request_timeout_seconds=float(os.environ.get("GATEWAY_REQUEST_TIMEOUT", "180")),
        )


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _json_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.settings = settings
        if client is None:
            timeout = httpx.Timeout(
                settings.request_timeout_seconds,
                connect=settings.connect_timeout_seconds,
            )
            application.state.voice_client = httpx.AsyncClient(
                base_url=settings.voice_base_url,
                timeout=timeout,
            )
            application.state.owns_voice_client = True
        else:
            application.state.voice_client = client
            application.state.owns_voice_client = False
        yield
        if application.state.owns_voice_client:
            await application.state.voice_client.aclose()

    application = FastAPI(title="hd-voice-gateway", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    @application.exception_handler(GatewayError)
    async def gateway_error_handler(_request: Request, exc: GatewayError):
        return _json_error(exc.status, exc.code, exc.message)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError):
        detail = exc.errors()[0].get("msg", "invalid request") if exc.errors() else "invalid request"
        return _json_error(400, "bad_request", detail)

    async def require_gateway_key(
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if x_api_key is None or not hmac.compare_digest(
            x_api_key.encode("utf-8"), settings.gateway_api_key.encode("utf-8")
        ):
            raise GatewayError(401, "unauthorized", "invalid or missing X-API-Key")

    def downstream_headers(request: Request) -> dict[str, str]:
        return {
            "X-API-Key": settings.voice_api_key,
            "X-Request-Id": request.state.request_id,
        }

    async def call_voice(request: Request, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await request.app.state.voice_client.request(
                method,
                path,
                headers=downstream_headers(request),
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise GatewayError(504, "downstream_timeout", "voice service timed out") from exc
        except httpx.HTTPError as exc:
            raise GatewayError(502, "downstream_unavailable", "voice service is unavailable") from exc
        if not response.is_success:
            raise GatewayError(
                502,
                "downstream_error",
                f"voice service returned HTTP {response.status_code}",
            )
        return response

    async def recognize(request: Request, data: bytes, filename: str, media_type: str) -> str:
        upload = {"file": (filename, data, media_type)}
        response = await call_voice(
            request,
            "POST",
            "/v1/asr",
            params={"refine": "true"},
            files=upload,
        )
        try:
            text = str(response.json().get("text") or "").strip()
        except (ValueError, AttributeError) as exc:
            raise GatewayError(502, "downstream_invalid_response", "ASR returned invalid JSON") from exc
        if text:
            return text

        response = await call_voice(
            request,
            "POST",
            "/v1/asr",
            params={"refine": "false"},
            files=upload,
        )
        try:
            return str(response.json().get("text") or "").strip()
        except (ValueError, AttributeError) as exc:
            raise GatewayError(502, "downstream_invalid_response", "ASR returned invalid JSON") from exc

    @application.get("/v1/health")
    async def health(request: Request):
        try:
            response = await call_voice(request, "GET", "/v1/health")
            payload = response.json()
        except GatewayError:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "ready": False, "voice_ready": False},
            )
        except (ValueError, AttributeError):
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "ready": False, "voice_ready": False},
            )
        ready = bool(payload.get("ready"))
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ok" if ready else "not_ready", "ready": ready, "voice_ready": ready},
        )

    @application.post("/v1/voice-command", dependencies=[Depends(require_gateway_key)])
    async def voice_command(request: Request, file: UploadFile = File(...)):
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if not data:
            raise GatewayError(400, "bad_request", "empty upload")
        if len(data) > max_bytes:
            raise GatewayError(413, "upload_too_large", f"upload exceeds {settings.max_upload_mb}MB limit")

        filename = file.filename or "command.wav"
        media_type = file.content_type or "audio/wav"
        recognized = await recognize(request, data, filename, media_type)
        if not recognized:
            raise GatewayError(422, "speech_not_recognized", "no speech was recognized")

        reply_text = f"我收到了命令：{recognized}"
        log.info(
            "voice_command_text %s",
            json.dumps(
                {
                    "request_id": request.state.request_id,
                    "recognized_text": recognized,
                    "reply_text": reply_text,
                },
                ensure_ascii=False,
            ),
        )
        response = await call_voice(
            request,
            "POST",
            "/v1/tts",
            json={"text": reply_text, "speaker": None, "speed": 1.0, "sample_rate": 16000},
        )
        audio = response.content
        if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise GatewayError(502, "downstream_invalid_response", "TTS did not return a WAV file")
        return Response(content=audio, media_type="audio/wav")

    return application


app = create_app()
