"""REST 接口：/v1/asr /v1/tts /v1/vad /v1/denoise /v1/models /v1/health（契约见 md §3.4）。"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import soxr
from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field

from .. import metrics
from ..errors import BadRequest, NotReady
from ..pipeline.asr_session import SR, AsrSession
from ..pipeline.audio import AudioError, load_audio_bytes, resample, wav_bytes
from .auth import require_api_key

router = APIRouter()
log = logging.getLogger(__name__)


def get_hub(request: Request):
    hub = request.app.state.hub
    if not hub.ready:
        raise NotReady("models not loaded")
    return hub


async def _read_upload(file: UploadFile, request: Request) -> bytes:
    data = await file.read()
    max_bytes = request.app.state.settings.limits.max_upload_mb * 1024 * 1024
    if not data:
        raise BadRequest("empty upload")
    if len(data) > max_bytes:
        raise BadRequest(f"upload exceeds {max_bytes // (1024 * 1024)}MB limit")
    return data


def _load(data: bytes, request: Request) -> np.ndarray:
    try:
        samples, _sr = load_audio_bytes(data, max_mb=request.app.state.settings.limits.max_upload_mb)
    except AudioError as e:
        raise BadRequest(str(e)) from e
    return samples


@router.post("/v1/asr", dependencies=[Depends(require_api_key)])
async def rest_asr(request: Request, file: UploadFile = File(...), denoise: bool = Query(default=None)):
    hub = get_hub(request)
    settings = request.app.state.settings
    denoise = settings.defaults.denoise if denoise is None else denoise
    samples = _load(await _read_upload(file, request), request)

    def run():
        session = AsrSession(hub, denoise=denoise)
        events = []
        step = SR * 10  # 10s 分块喂入，避免单次线程占用过长
        for i in range(0, len(samples), step):
            events += session.feed(samples[i:i + step])
        events += session.finish()
        return events

    t0 = time.perf_counter()
    events = await asyncio.to_thread(run)
    wall_ms = (time.perf_counter() - t0) * 1000
    finals = [e for e in events if e["type"] == "final"]
    text = "".join(f["text"] for f in finals)
    audio_s = len(samples) / SR
    return {
        "text": text,
        "language": hub.info.get("asr", {}).get("language", "zh"),
        "refined": False,
        "denoise": denoise,
        "segments": [
            {"index": f["segment"], "start": f["start"], "end": f["end"], "text": f["text"]}
            for f in finals
        ],
        "timings": {"wall_ms": round(wall_ms, 1)},
        "rtf": round((wall_ms / 1000.0) / audio_s, 4) if audio_s > 0 else None,
    }


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    speaker: int | str = 0
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = 16000


@router.post("/v1/tts", dependencies=[Depends(require_api_key)])
async def rest_tts(request: Request, body: TtsRequest):
    hub = get_hub(request)
    settings = request.app.state.settings
    if body.sample_rate not in settings.tts.sample_rate_choices:
        raise BadRequest(f"sample_rate must be one of {settings.tts.sample_rate_choices}")
    try:
        sid = hub.resolve_speaker(body.speaker)
    except ValueError as e:
        raise BadRequest(str(e)) from e
    metrics.tts_requests_total.inc()
    t0 = time.perf_counter()
    audio = await asyncio.to_thread(hub.tts.generate, body.text, sid, body.speed)
    samples = np.asarray(audio.samples, dtype=np.float32)
    native_sr = int(audio.sample_rate)
    if body.sample_rate != native_sr:
        samples = soxr.resample(samples, native_sr, body.sample_rate).astype(np.float32)
    data = wav_bytes(samples, body.sample_rate)
    metrics.tts_sentence_ms.observe((time.perf_counter() - t0) * 1000)
    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            "X-Audio-Ms": str(round(len(samples) / body.sample_rate * 1000)),
            "X-Synth-Ms": str(round((time.perf_counter() - t0) * 1000)),
        },
    )


@router.post("/v1/vad", dependencies=[Depends(require_api_key)])
async def rest_vad(request: Request, file: UploadFile = File(...)):
    hub = get_hub(request)
    samples = _load(await _read_upload(file, request), request)

    def run():
        vad = hub.create_vad()
        w = 512
        for i in range(0, len(samples) - w + 1, w):
            vad.accept_waveform(samples[i:i + w])
        vad.flush()
        segs = []
        while not vad.empty():
            seg = vad.front
            vad.pop()
            segs.append({"start": round(seg.start / SR, 3),
                         "end": round((seg.start + len(seg.samples)) / SR, 3)})
        return segs

    return {"segments": await asyncio.to_thread(run)}


@router.post("/v1/denoise", dependencies=[Depends(require_api_key)])
async def rest_denoise(request: Request, file: UploadFile = File(...)):
    hub = get_hub(request)
    samples = _load(await _read_upload(file, request), request)

    def run():
        den = hub.create_denoiser()
        fs = int(den.frame_shift_in_samples)
        out = []
        for i in range(0, len(samples) - fs + 1, fs):
            out.append(np.asarray(den.run(samples[i:i + fs], SR).samples, dtype=np.float32))
        tail = np.asarray(den.flush().samples, dtype=np.float32)
        if tail.size:
            out.append(tail)
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    data = await asyncio.to_thread(run)
    return Response(content=wav_bytes(data, SR), media_type="audio/wav",
                    headers={"X-Audio-Ms": str(round(len(data) / SR * 1000))})


@router.get("/v1/models")
async def get_models(request: Request):
    hub = request.app.state.hub
    return {"ready": hub.ready, "provider": "cpu", "models": hub.info,
            "load_errors": hub.load_errors}


@router.get("/v1/health")
async def health(request: Request):
    hub = request.app.state.hub
    status = "ok" if hub.ready else "not_ready"
    return Response(
        content='{"status":"%s","ready":%s}' % (status, "true" if hub.ready else "false"),
        media_type="application/json",
        status_code=200 if hub.ready else 503,
    )
