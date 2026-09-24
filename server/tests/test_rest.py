import io
import json
import wave

import numpy as np
import soundfile as sf

from tests.conftest import FakeHub, FakeTts, make_app_with_hub


def make_wav(seconds=1.0, sr=16000) -> bytes:
    x = (0.1 * np.sin(np.linspace(0, 2000, int(seconds * sr), dtype=np.float32)))
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_health_ready(client):
    r = client.get("/v1/health")
    assert r.status_code == 200 and r.json()["ready"] is True


def test_health_not_ready(settings):
    hub = FakeHub(settings)
    hub.ready = False
    from fastapi.testclient import TestClient
    with TestClient(make_app_with_hub(settings, hub)) as c:
        assert c.get("/v1/health").status_code == 503


def test_models_public(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "cpu" and body["ready"] is True


def test_tts_wav(client):
    r = client.post("/v1/tts", headers={"X-API-Key": "test-key"},
                    json={"text": "你好。", "speaker": 0, "speed": 1.0, "sample_rate": 16000})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"
    assert int(r.headers["X-Audio-Ms"]) > 100
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2


def test_tts_unauthorized(client):
    r = client.post("/v1/tts", json={"text": "hi"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


def test_tts_validation_400(client):
    r = client.post("/v1/tts", headers={"X-API-Key": "test-key"}, json={"text": ""})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"
    r = client.post("/v1/tts", headers={"X-API-Key": "test-key"},
                    json={"text": "hi", "sample_rate": 8000})
    assert r.status_code == 400


def test_vad_endpoint(client):
    r = client.post("/v1/vad", headers={"X-API-Key": "test-key"},
                    files={"file": ("a.wav", make_wav(), "audio/wav")})
    assert r.status_code == 200
    assert "segments" in r.json()


def test_bad_audio_is_400_not_500(client):
    r = client.post("/v1/vad", headers={"X-API-Key": "test-key"},
                    files={"file": ("a.bin", b"garbage" * 100, "application/octet-stream")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


def test_asr_rest_with_fakes(client):
    r = client.post("/v1/asr", headers={"X-API-Key": "test-key"},
                    files={"file": ("a.wav", make_wav(0.5), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["refined"] is False
    assert "segments" in body and "rtf" in body


def test_denoise_rest(client):
    r = client.post("/v1/denoise", headers={"X-API-Key": "test-key"},
                    files={"file": ("a.wav", make_wav(0.5), "audio/wav")})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"voice_asr_sessions" in r.content or b"voice_rest_requests" in r.content


def test_request_id_echo(client):
    r = client.get("/v1/health", headers={"X-Request-Id": "rid-123"})
    assert r.headers.get("x-request-id") == "rid-123"
