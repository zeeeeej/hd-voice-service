"""Qwen3-TTS 私有 sidecar：模型只加载一次，推理串行执行。"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


SPEAKERS = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
)
log = logging.getLogger(__name__)


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    speaker: str = "Vivian"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class QwenRuntime:
    def __init__(self):
        self.model_path = os.environ.get(
            "QWEN_MODEL_PATH", "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        self.model_name = Path(self.model_path).name
        self.language = os.environ.get("QWEN_LANGUAGE", "Chinese")
        self.device = os.environ.get("QWEN_DEVICE", "cpu")
        self.sample_rate = 24000
        self.model = None
        self.load_seconds: float | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        started = time.monotonic()
        model = Qwen3TTSModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        self.model = model
        self.load_seconds = round(time.monotonic() - started, 3)
        log.info("Qwen3-TTS loaded in %.3fs on %s", self.load_seconds, self.device)

    def metadata(self) -> dict:
        return {
            "ready": self.model is not None,
            "model": self.model_name,
            "sample_rate": self.sample_rate,
            "speakers": list(SPEAKERS),
            "language": self.language,
            "device": self.device,
            "load_seconds": self.load_seconds,
        }

    def synthesize(self, text: str, speaker: str, speed: float) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        canonical = {name.casefold(): name for name in SPEAKERS}.get(speaker.casefold())
        if canonical is None:
            raise ValueError(f"unknown speaker {speaker!r}")
        with self.lock:
            wavs, sample_rate = self.model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=canonical,
                non_streaming_mode=True,
            )
        if int(sample_rate) != self.sample_rate:
            raise RuntimeError(f"unexpected sample rate: {sample_rate}")
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if speed != 1.0 and samples.size:
            import librosa
            samples = np.asarray(
                librosa.effects.time_stretch(samples, rate=speed), dtype=np.float32)
        return np.ascontiguousarray(samples, dtype="<f4")


def create_app(runtime: QwenRuntime | None = None) -> FastAPI:
    active_runtime = runtime or QwenRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = active_runtime
        if runtime is None:
            await asyncio.to_thread(active_runtime.load)
        yield

    app = FastAPI(title="qwen3-tts-sidecar", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        ready = active_runtime.model is not None
        return Response(
            content='{"status":"%s","ready":%s}' % (
                "ok" if ready else "not_ready", "true" if ready else "false"),
            media_type="application/json",
            status_code=200 if ready else 503,
        )

    @app.get("/metadata")
    async def metadata():
        if active_runtime.model is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return active_runtime.metadata()

    @app.post("/synthesize")
    async def synthesize(body: SynthesisRequest):
        try:
            samples = await asyncio.to_thread(
                active_runtime.synthesize, body.text, body.speaker, body.speed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=samples.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(active_runtime.sample_rate),
                "X-Audio-Format": "pcm_f32le",
            },
        )

    return app


app = create_app()
