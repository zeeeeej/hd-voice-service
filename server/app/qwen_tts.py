"""Qwen3-TTS 内部服务适配器。

主服务维持 sherpa-onnx OfflineTts 的最小接口，Qwen 的 PyTorch 运行时则隔离在
独立容器中。回调收到的是完整句子生成后切分出的 PCM，不代表模型级流式解码。
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np


QWEN_SPEAKERS = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
)


@dataclass
class GeneratedAudio:
    samples: np.ndarray
    sample_rate: int


def split_text(text: str, max_chars: int) -> list[str]:
    """按中文句读优先切分，单段绝不超过 max_chars。"""
    text = text.strip()
    if not text:
        return []
    if max_chars < 1:
        raise ValueError("qwen_max_chars must be positive")
    chunks: list[str] = []
    punctuation = "。！？!?；;，,、\n"
    while len(text) > max_chars:
        window = text[:max_chars]
        cut = max(window.rfind(mark) for mark in punctuation) + 1
        if cut < max_chars // 3:
            cut = max_chars
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


class QwenRemoteTts:
    """通过内部 HTTP 调用 Qwen3-TTS，并暴露与 OfflineTts 相同的关键属性。"""

    def __init__(self, base_url: str, timeout: float = 300.0,
                 max_chars: int = 300, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
        )
        self.max_chars = max_chars
        try:
            response = self._client.get("/metadata")
            response.raise_for_status()
            metadata = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"Qwen3-TTS metadata request failed: {exc}") from exc
        if not isinstance(metadata, dict):
            raise RuntimeError("Qwen3-TTS metadata response must be an object")
        self.metadata = metadata
        self.sample_rate = int(metadata.get("sample_rate", 24000))
        self.speaker_names = tuple(metadata.get("speakers") or QWEN_SPEAKERS)
        self.num_speakers = len(self.speaker_names)
        self.model_name = str(metadata.get("model") or "Qwen3-TTS")

    def generate(self, text: str, speaker: str, speed: float = 1.0, callback=None):
        arrays: list[np.ndarray] = []
        keep_going = True
        for chunk in split_text(text, self.max_chars):
            try:
                response = self._client.post(
                    "/synthesize",
                    json={"text": chunk, "speaker": speaker, "speed": speed},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                response = getattr(exc, "response", None)
                detail = getattr(response, "text", "") if response is not None else ""
                raise RuntimeError(f"Qwen3-TTS synthesis failed: {exc} {detail}".strip()) from exc
            actual_rate = int(response.headers.get("x-sample-rate", self.sample_rate))
            if actual_rate != self.sample_rate:
                raise RuntimeError(
                    f"Qwen3-TTS sample rate changed: {actual_rate} != {self.sample_rate}")
            if len(response.content) % 4:
                raise RuntimeError("Qwen3-TTS returned malformed float32 PCM")
            samples = np.frombuffer(response.content, dtype="<f4").copy()
            arrays.append(samples)
            if callback is not None:
                frame_samples = max(1, self.sample_rate // 10)
                for start in range(0, len(samples), frame_samples):
                    if callback(samples[start:start + frame_samples], 1.0) == 0:
                        keep_going = False
                        break
            if not keep_going:
                break
        combined = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
        return GeneratedAudio(samples=combined, sample_rate=self.sample_rate)

    def close(self) -> None:
        self._client.close()
