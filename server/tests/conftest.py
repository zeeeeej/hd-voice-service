"""测试基建：Fake 模型组件 + 可注入的 FastAPI app（不加载真实模型）。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.models import ModelHub  # noqa: E402

SR = 16000
WINDOW = 512


class FakeResult:
    def __init__(self, text=""):
        self.text = text


class FakeStream:
    def __init__(self):
        self.result = FakeResult()
        self.fed_samples = 0
        self.pending = 0

    def accept_waveform(self, sr, samples):
        self.fed_samples += len(samples)
        self.pending += 1


class FakeRecognizer:
    """按脚本推进识别文本；decode 次数达到 endpoint_after 时触发端点。"""

    def __init__(self, texts=None, endpoint_after=None):
        self.texts = list(texts or [])
        self.endpoint_after = endpoint_after
        self.decode_calls = 0
        self.stream = None
        self._endpoint_fired = False

    def create_stream(self):
        self.stream = FakeStream()
        return self.stream

    def is_ready(self, stream):
        return stream.pending > 0

    def decode_stream(self, stream):
        stream.pending = max(0, stream.pending - 1)
        self.decode_calls += 1
        if self.texts:
            stream.result.text = self.texts[min(self.decode_calls, len(self.texts)) - 1]

    def get_result(self, stream):
        return stream.result.text

    def is_endpoint(self, stream):
        # 真实 recognizer 端点依赖尾静音计时，reset 后不会立即复燃；fake 只触发一次
        if self.endpoint_after is not None and self.decode_calls >= self.endpoint_after \
                and not self._endpoint_fired:
            self._endpoint_fired = True
            return True
        return False

    def reset(self, stream):
        stream.result.text = ""
        self.decode_calls = 0
        # 真实语义：reset 后旧音频不会重新解码；无新语音则无新文本
        self.texts = []


class FakeSegment:
    def __init__(self, start, samples):
        self.start = start
        self.samples = samples


class FakeVad:
    """speech_windows: 每个 512 窗口的语音状态脚本；
    segments: (start_win, end_win) 列表，窗口越过 end_win 后变为已完成段。"""

    def __init__(self, speech_windows=None, segments=None):
        self.speech_windows = list(speech_windows or [])
        self.segments = list(segments or [])
        self.i = 0
        self.completed: list[FakeSegment] = []
        self.flushed = False
        self.config = SimpleNamespace(silero_vad=SimpleNamespace(window_size=WINDOW))

    def accept_waveform(self, samples):
        assert len(samples) == WINDOW
        self.i += 1
        for (sw, ew) in list(self.segments):
            if self.i > ew:
                self.segments.remove((sw, ew))
                self.completed.append(FakeSegment(sw * WINDOW, np.zeros((ew - sw) * WINDOW, dtype=np.float32)))

    def is_speech_detected(self):
        if self.i < len(self.speech_windows):
            return self.speech_windows[self.i]
        return False

    def empty(self):
        return not self.completed

    @property
    def front(self):
        return self.completed[0]

    def pop(self):
        return self.completed.pop(0)

    def flush(self):
        self.flushed = True
        for (sw, ew) in list(self.segments):
            self.segments.remove((sw, ew))
            self.completed.append(FakeSegment(sw * WINDOW, np.zeros((ew - sw) * WINDOW, dtype=np.float32)))

    def reset(self):
        pass


class FakeDenoisedAudio:
    def __init__(self, samples, sample_rate=SR):
        self.samples = samples
        self.sample_rate = sample_rate


class FakeDenoiser:
    frame_shift_in_samples = 256
    sample_rate = SR

    def __init__(self, gain=1.0):
        self.gain = gain

    def run(self, samples, sample_rate=SR):
        return FakeDenoisedAudio(np.asarray(samples, dtype=np.float32) * self.gain)

    def flush(self):
        return FakeDenoisedAudio(np.empty(0, dtype=np.float32))

    def reset(self):
        pass


class FakeGeneratedAudio:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate


class FakeTts:
    """每句产出 0.3s 正弦音频；有 callback 时分 3 块回调。"""

    def __init__(self, sample_rate=24000, num_speakers=4):
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.calls: list[tuple] = []

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls.append((text, sid, speed, callback is not None))
        n = int(0.3 * self.sample_rate)
        full = (0.3 * np.sin(np.linspace(0, 200 * np.pi, n, dtype=np.float32))).astype(np.float32)
        if callback is None:
            return FakeGeneratedAudio(full, self.sample_rate)
        part = n // 3
        for i in range(0, n, part):
            if callback(full[i:i + part], i / n) == 0:
                break
        return FakeGeneratedAudio(full, self.sample_rate)


class FakeHub(ModelHub):
    """继承 ModelHub 以复用 resolve_speaker；不加载真实模型。"""

    def __init__(self, settings, recognizer=None, vad_factory=None, tts=None, denoiser_factory=None):
        super().__init__(settings)
        self.recognizer = recognizer or FakeRecognizer()
        self._vad_factory = vad_factory or (lambda: FakeVad())
        self.tts = tts or FakeTts()
        self._denoiser_factory = denoiser_factory or (lambda: FakeDenoiser())
        self.info = {"provider": "cpu", "asr": {"language": "zh"}, "tts": {"type": "fake"}}
        self.ready = True

    def create_denoiser(self):
        return self._denoiser_factory()

    def create_vad(self):
        return self._vad_factory()


@pytest.fixture
def settings():
    s = Settings()
    s.api_key = "test-key"
    s.limits.asr_sessions = 2
    s.limits.tts_sessions = 1
    s.limits.idle_timeout = 5
    s.limits.max_session = 30
    return s


@pytest.fixture
def app(settings):
    from app.main import create_app
    hub = FakeHub(settings)
    return create_app(settings=settings, hub=hub)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


def make_app_with_hub(settings, hub):
    from app.main import create_app
    return create_app(settings=settings, hub=hub)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)
