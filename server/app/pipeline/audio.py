"""音频工具：PCM/WAV 解析、重采样、格式转换。"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000


class AudioError(ValueError):
    pass


def pcm16_bytes_to_f32(data: bytes) -> np.ndarray:
    if len(data) % 2 != 0:
        raise AudioError("pcm_s16le chunk has odd byte length")
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def f32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples
    return soxr.resample(samples, src_sr, dst_sr).astype(np.float32)


def load_audio_bytes(data: bytes, target_sr: int = TARGET_SR, max_mb: int = 25) -> tuple[np.ndarray, int]:
    """解析上传音频（wav/flac/ogg 等 soundfile 支持的格式）→ 单声道 float32 @target_sr。"""
    if len(data) > max_mb * 1024 * 1024:
        raise AudioError(f"audio too large (> {max_mb}MB)")
    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioError(f"unsupported or corrupted audio: {e}") from e
    if samples.shape[0] == 0:
        raise AudioError("empty audio")
    mono = samples.mean(axis=1).astype(np.float32)
    return resample(mono, sr, target_sr), target_sr


def wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
