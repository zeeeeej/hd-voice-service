import io

import numpy as np
import pytest
import soundfile as sf

from app.pipeline.audio import (AudioError, f32_to_pcm16_bytes, load_audio_bytes,
                                pcm16_bytes_to_f32, resample, wav_bytes)


def test_pcm16_roundtrip():
    x = np.array([0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    y = pcm16_bytes_to_f32(f32_to_pcm16_bytes(x))
    assert np.allclose(x, y, atol=1e-4)


def test_odd_bytes_rejected():
    with pytest.raises(AudioError):
        pcm16_bytes_to_f32(b"\x01\x02\x03")


def test_load_wav_resamples_to_16k_mono():
    sr = 44100
    x = (0.2 * np.sin(np.linspace(0, 4000, sr * 2, dtype=np.float32))).reshape(-1, 1)
    stereo = np.repeat(x, 2, axis=1)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    samples, out_sr = load_audio_bytes(buf.getvalue())
    assert out_sr == 16000
    assert samples.ndim == 1
    assert abs(len(samples) - 16000 * 2) < 10


def test_load_corrupt_raises_audio_error():
    with pytest.raises(AudioError):
        load_audio_bytes(b"not audio at all" * 10)


def test_wav_bytes_header():
    data = wav_bytes(np.zeros(1600, dtype=np.float32), 16000)
    assert data[:4] == b"RIFF"
    samples, sr = sf.read(io.BytesIO(data), dtype="float32")
    assert sr == 16000 and len(samples) == 1600


def test_resample_identity():
    x = np.zeros(100, dtype=np.float32)
    assert resample(x, 16000, 16000) is x
