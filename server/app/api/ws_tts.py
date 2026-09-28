"""WebSocket /v1/ws/tts —— 流式 TTS（契约见 md §3.4）。

架构：
- 主协程收控制帧，文本经 TextChunker 按句切分入 sentence 队列
- 合成协程逐句在线程中跑 OfflineTts.generate(callback)，回调把音频块推入 out 队列
- 发送协程独占 ws 写，从 out 队列取 ("bin"/"json") 帧发送，保证顺序
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid

import numpy as np
import soxr
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import metrics
from ..errors import BadRequest, Overloaded, UnsupportedOption, VoiceError
from ..logging_conf import request_id_var
from ..pipeline.audio import f32_to_pcm16_bytes
from ..pipeline.text_chunker import TextChunker
from .auth import Unauthorized, ws_authorized

router = APIRouter()
log = logging.getLogger(__name__)


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    try:
        await ws.send_json({"type": "error", "code": code, "message": message})
    except Exception:
        pass


@router.websocket("/v1/ws/tts")
async def ws_tts(ws: WebSocket):
    await ws.accept()
    settings = ws.app.state.settings
    hub = ws.app.state.hub
    rid = ws.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    limits = settings.limits

    async def fail(exc: VoiceError):
        await _send_error(ws, exc.code, exc.message)
        await ws.close(code=exc.ws_close_code)

    if not ws_authorized(ws.headers, ws.query_params, settings.api_key):
        await fail(Unauthorized("invalid or missing X-API-Key"))
        return
    if not hub.ready:
        from ..errors import NotReady
        await fail(NotReady("models not loaded"))
        return

    sem: asyncio.Semaphore = ws.app.state.tts_sem
    if sem.locked():
        metrics.rejected_total.labels(kind="tts_ws").inc()
        await fail(Overloaded("TTS session limit reached"))
        return

    async with sem:
        metrics.tts_sessions_active.inc()
        metrics.tts_requests_total.inc()
        loop = asyncio.get_running_loop()
        cancel = threading.Event()
        out_q: asyncio.Queue = asyncio.Queue()
        sent_q: asyncio.Queue = asyncio.Queue()
        chunker = TextChunker()
        started = time.monotonic()
        stats = {"sentences": 0, "total_audio_ms": 0, "first_chunk_ms": None}

        async def sender():
            while True:
                kind, payload = await out_q.get()
                if kind == "end":
                    return
                try:
                    if kind == "bin":
                        await ws.send_bytes(payload)
                    else:
                        await ws.send_json(payload)
                except Exception:
                    cancel.set()
                    return

        async def synth_worker():
            """逐句在线程中合成；音频块经 call_soon_threadsafe 推入 out_q。"""
            if True:
                tts = hub.tts
                native_sr = int(tts.sample_rate)
                while True:
                    item = await sent_q.get()
                    if item is None or cancel.is_set():
                        break
                    index, sentence, out_sr, sid, speed = item
                    resampler = (
                        soxr.ResampleStream(native_sr, out_sr, 1, dtype="float32")
                        if out_sr != native_sr else None
                    )
                    t0 = time.perf_counter()
                    first = {"done": False}
                    audio_ms = {"v": 0.0}

                    def cb(samples: np.ndarray, progress: float) -> int:
                        if cancel.is_set():
                            return 0
                        arr = np.asarray(samples, dtype=np.float32)
                        if resampler is not None:
                            arr = resampler.resample_chunk(arr)
                        if arr.size == 0:
                            return 1
                        audio_ms["v"] += len(arr) / out_sr * 1000.0
                        payload = f32_to_pcm16_bytes(arr)
                        if not first["done"]:
                            first["done"] = True
                            ms = (time.perf_counter() - t0) * 1000
                            metrics.tts_first_chunk_ms.observe(ms)
                            if stats["first_chunk_ms"] is None:
                                stats["first_chunk_ms"] = round(ms, 1)
                        loop.call_soon_threadsafe(out_q.put_nowait, ("bin", payload))
                        return 1

                    await asyncio.to_thread(tts.generate, sentence, sid, speed, cb)
                    if cancel.is_set():
                        break
                    dur_ms = (time.perf_counter() - t0) * 1000
                    metrics.tts_sentence_ms.observe(dur_ms)
                    stats["sentences"] += 1
                    stats["total_audio_ms"] += audio_ms["v"]
                    out_q.put_nowait(("json", {
                        "type": "sentence_done", "index": index,
                        "text": sentence, "audio_ms": round(audio_ms["v"], 1),
                    }))
                out_q.put_nowait(("json", {
                    "type": "done",
                    "total_audio_ms": round(stats["total_audio_ms"], 1),
                    "timings": {
                        "first_chunk_ms": stats["first_chunk_ms"],
                        "total_ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                }))
                out_q.put_nowait(("end", None))

        sender_task = asyncio.create_task(sender())
        worker_task = asyncio.create_task(synth_worker())

        try:
            # ---- 等待 start 帧 ----
            cfg = None
            while cfg is None:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=limits.idle_timeout)
                except asyncio.TimeoutError:
                    await _send_error(ws, "idle_timeout", "start frame not received")
                    await ws.close(code=1000)
                    return
                if msg["type"] == "websocket.disconnect":
                    return
                if msg.get("bytes") is not None:
                    await fail(BadRequest("expected start control frame, got binary"))
                    return
                try:
                    ctrl = json.loads(msg.get("text") or "")
                except json.JSONDecodeError:
                    await fail(BadRequest("invalid JSON"))
                    return
                if ctrl.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
                    continue
                if ctrl.get("type") != "start":
                    await fail(BadRequest("first frame must be type=start"))
                    return
                cfg = ctrl

            speaker = cfg.get("speaker")
            try:
                sid = hub.resolve_speaker(speaker)
            except ValueError as e:
                await fail(BadRequest(str(e)))
                return
            speed = float(cfg.get("speed", hub.s.tts.default_speed))
            if not (0.5 <= speed <= 2.0):
                await fail(BadRequest("speed must be in [0.5, 2.0]"))
                return
            fmt = cfg.get("format", "pcm_s16le")
            if fmt != "pcm_s16le":
                await fail(UnsupportedOption(f"v1 supports format=pcm_s16le only, got {fmt!r}"))
                return
            out_sr = int(cfg.get("sample_rate", 16000))
            if out_sr not in settings.tts.sample_rate_choices:
                await fail(UnsupportedOption(f"sample_rate must be one of {settings.tts.sample_rate_choices}"))
                return
            if int(cfg.get("channels", 1)) != 1:
                await fail(UnsupportedOption("v1 supports channels=1 only"))
                return

            await ws.send_json({"type": "ready", "session_id": rid, "sample_rate": out_sr, "speaker": sid})

            # ---- 主接收循环 ----
            index = 0
            while True:
                if time.monotonic() - started > limits.max_session:
                    await _send_error(ws, "session_timeout", f"session exceeded {limits.max_session}s")
                    cancel.set()
                    await ws.close(code=1000)
                    break
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=limits.idle_timeout)
                except asyncio.TimeoutError:
                    await _send_error(ws, "idle_timeout", f"no message for {limits.idle_timeout}s")
                    cancel.set()
                    await ws.close(code=1000)
                    break
                if msg["type"] == "websocket.disconnect":
                    cancel.set()
                    break
                if msg.get("bytes") is not None:
                    await _send_error(ws, "bad_request", "unexpected binary frame from client")
                    cancel.set()
                    await ws.close(code=1008)
                    break
                try:
                    ctrl = json.loads(msg.get("text") or "")
                except json.JSONDecodeError:
                    await _send_error(ws, "bad_request", "invalid JSON control frame")
                    cancel.set()
                    await ws.close(code=1008)
                    break
                ctype = ctrl.get("type")

                if ctype == "text":
                    t = ctrl.get("text", "")
                    if not isinstance(t, str) or not t:
                        await _send_error(ws, "bad_request", "text frame requires non-empty string 'text'")
                        continue
                    for sentence in chunker.push(t):
                        await sent_q.put((index, sentence, out_sr, sid, speed))
                        index += 1
                elif ctype == "flush":
                    for sentence in chunker.flush():
                        await sent_q.put((index, sentence, out_sr, sid, speed))
                        index += 1
                elif ctype == "eof":
                    for sentence in chunker.flush():
                        await sent_q.put((index, sentence, out_sr, sid, speed))
                        index += 1
                    await sent_q.put(None)
                    await worker_task
                    await sender_task
                    await ws.close(code=1000)
                    break
                elif ctype == "ping":
                    await out_q.put(("json", {"type": "pong"}))
                elif ctype == "cancel":
                    cancel.set()
                    await sent_q.put(None)
                    await ws.close(code=1000)
                    break
                else:
                    await _send_error(ws, "bad_request", f"unknown control type {ctype!r}")
                    cancel.set()
                    await ws.close(code=1008)
                    break
        except WebSocketDisconnect:
            cancel.set()
        except Exception:
            log.exception("TTS WS internal error")
            metrics.errors_total.labels(path="/v1/ws/tts").inc()
            cancel.set()
            await _send_error(ws, "internal_error", "internal server error")
            try:
                await ws.close(code=1011)
            except Exception:
                pass
        finally:
            cancel.set()
            await sent_q.put(None)
            for task in (worker_task, sender_task):
                if not task.done():
                    task.cancel()
            metrics.tts_sessions_active.dec()
            log.info("tts session closed rid=%s sentences=%d audio_ms=%.0f",
                     rid, stats["sentences"], stats["total_audio_ms"])
