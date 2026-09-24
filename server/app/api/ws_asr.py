"""WebSocket /v1/ws/asr —— 流式 ASR（契约见《自建 Linux 流式语音服务器.md》§3.4）。"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import metrics
from ..errors import UnsupportedOption
from ..logging_conf import request_id_var
from ..pipeline.asr_session import AsrSession
from ..pipeline.audio import AudioError, pcm16_bytes_to_f32
from .auth import Unauthorized, ws_authorized

router = APIRouter()
log = logging.getLogger(__name__)

V1_UNSUPPORTED = ("refine", "punctuate", "itn")  # v1 精简核心：显式拒绝


def _truthy(v) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    try:
        await ws.send_json({"type": "error", "code": code, "message": message})
    except Exception:
        pass


@router.websocket("/v1/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    settings = ws.app.state.settings
    hub = ws.app.state.hub
    rid = ws.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    limits = settings.limits

    async def fail(exc, close_code=None):
        await _send_error(ws, exc.code, exc.message)
        await ws.close(code=close_code or exc.ws_close_code)

    if not ws_authorized(ws.headers, ws.query_params, settings.api_key):
        await fail(Unauthorized("invalid or missing X-API-Key"))
        return
    if not hub.ready:
        from ..errors import NotReady
        await fail(NotReady("models not loaded"))
        return

    q = ws.query_params
    try:
        encoding = q.get("encoding", "pcm_s16le")
        if encoding != "pcm_s16le":
            raise UnsupportedOption(f"v1 supports encoding=pcm_s16le only, got {encoding!r}")
        if int(q.get("sample_rate", 16000)) != 16000:
            raise UnsupportedOption("v1 supports sample_rate=16000 only")
        if int(q.get("channels", 1)) != 1:
            raise UnsupportedOption("v1 supports channels=1 only")
        for flag in V1_UNSUPPORTED:
            if flag in q and _truthy(q.get(flag)):
                raise UnsupportedOption(f"{flag}=true is not available in v1 (refine stage deferred)")
    except (UnsupportedOption, ValueError) as e:
        await fail(e if isinstance(e, UnsupportedOption) else UnsupportedOption(str(e)))
        return

    denoise = _truthy(q.get("denoise")) if "denoise" in q else settings.defaults.denoise

    sem: asyncio.Semaphore = ws.app.state.asr_sem
    if sem.locked():
        metrics.rejected_total.labels(kind="asr_ws").inc()
        from ..errors import Overloaded
        await fail(Overloaded("ASR session limit reached"))
        return

    async with sem:
        metrics.asr_sessions_active.inc()
        metrics.asr_sessions_total.inc()
        try:
            session: AsrSession = await asyncio.to_thread(AsrSession, hub, denoise)
        except Exception as e:
            log.exception("failed to create ASR session")
            from ..errors import VoiceError
            await fail(VoiceError(f"session init failed: {e}"))
            metrics.asr_sessions_active.dec()
            return

        await ws.send_json({
            "type": "ready",
            "session_id": rid,
            "sample_rate": 16000,
            "models": hub.info,
        })

        started = time.monotonic()
        got_audio = False
        first_partial_observed = False
        finals = 0
        try:
            while True:
                if time.monotonic() - started > limits.max_session:
                    await _send_error(ws, "session_timeout", f"session exceeded {limits.max_session}s")
                    await ws.close(code=1000)
                    break
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=limits.idle_timeout)
                except asyncio.TimeoutError:
                    await _send_error(ws, "idle_timeout", f"no message for {limits.idle_timeout}s")
                    await ws.close(code=1000)
                    break

                if msg["type"] == "websocket.disconnect":
                    break

                if (data := msg.get("bytes")) is not None:
                    got_audio = True
                    try:
                        samples = pcm16_bytes_to_f32(data)
                    except AudioError as e:
                        await _send_error(ws, "bad_request", str(e))
                        await ws.close(code=1008)
                        break
                    t0 = time.perf_counter()
                    events = await asyncio.to_thread(session.feed, samples)
                    metrics.asr_chunk_ms.observe((time.perf_counter() - t0) * 1000)
                    for ev in events:
                        await ws.send_json(ev)
                        if ev["type"] == "final":
                            finals += 1
                            lat = session.first_partial_latency_ms
                            if lat is not None and not first_partial_observed:
                                metrics.asr_first_partial_ms.observe(lat)
                                first_partial_observed = True
                    continue

                text = msg.get("text")
                if text is None:
                    continue
                try:
                    ctrl = json.loads(text)
                except json.JSONDecodeError:
                    await _send_error(ws, "bad_request", "invalid JSON control frame")
                    await ws.close(code=1008)
                    break
                ctype = ctrl.get("type")

                if ctype == "eof":
                    events = await asyncio.to_thread(session.finish)
                    for ev in events:
                        await ws.send_json(ev)
                    await ws.close(code=1000)
                    break
                if ctype == "ping":
                    await ws.send_json({"type": "pong"})
                    continue
                if ctype == "cancel":
                    await ws.close(code=1000)
                    break
                if ctype == "config":
                    if got_audio:
                        await _send_error(ws, "bad_request", "config frame must precede audio")
                        await ws.close(code=1008)
                        break
                    new_denoise = _truthy(ctrl.get("denoise", False))
                    session = await asyncio.to_thread(AsrSession, hub, new_denoise)
                    continue
                await _send_error(ws, "bad_request", f"unknown control type {ctype!r}")
                await ws.close(code=1008)
                break
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("ASR WS internal error")
            metrics.errors_total.labels(path="/v1/ws/asr").inc()
            await _send_error(ws, "internal_error", "internal server error")
            try:
                await ws.close(code=1011)
            except Exception:
                pass
        finally:
            metrics.asr_sessions_active.dec()
            log.info("asr session closed rid=%s finals=%d audio=%s denoise=%s",
                     rid, finals, got_audio, session.denoise_enabled)
