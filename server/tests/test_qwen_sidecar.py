import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwen_service.app import create_app  # noqa: E402


class FakeQwenRuntime:
    model = object()
    model_name = "fake-qwen"
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def metadata(self):
        return {"ready": True, "model": self.model_name, "sample_rate": self.sample_rate,
                "speakers": ["Vivian"], "language": "Chinese", "device": "cpu"}

    def synthesize(self, text, speaker, speed):
        self.calls.append((text, speaker, speed))
        if speaker != "Vivian":
            raise ValueError("unknown speaker")
        return np.ones(2400, dtype="<f4")


def test_qwen_sidecar_contract():
    runtime = FakeQwenRuntime()
    with TestClient(create_app(runtime)) as client:
        assert client.get("/health").status_code == 200
        metadata = client.get("/metadata").json()
        assert metadata["sample_rate"] == 24000 and metadata["speakers"] == ["Vivian"]
        response = client.post(
            "/synthesize", json={"text": "你好", "speaker": "Vivian", "speed": 1.0})
        assert response.status_code == 200
        assert response.headers["x-audio-format"] == "pcm_f32le"
        assert len(response.content) == 2400 * 4
        assert runtime.calls == [("你好", "Vivian", 1.0)]
        assert client.post(
            "/synthesize", json={"text": "你好", "speaker": "bad", "speed": 1.0}
        ).status_code == 400
