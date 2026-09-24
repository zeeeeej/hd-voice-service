import numpy as np

from tests.conftest import SR, WINDOW, FakeDenoiser, FakeHub, FakeRecognizer, FakeVad, silence
from app.config import Settings
from app.pipeline.asr_session import AsrSession


def make_settings():
    s = Settings()
    s.api_key = "test-key"
    return s


def windows(n_speech: int, n_total: int):
    """前 n_speech 个窗口为语音。"""
    return [True] * n_speech + [False] * (n_total - n_speech)


def test_silence_produces_no_final():
    hub = FakeHub(make_settings(), recognizer=FakeRecognizer(texts=[""]), vad_factory=lambda: FakeVad())
    sess = AsrSession(hub, denoise=False)
    events = sess.feed(silence(1.0))
    events += sess.finish()
    assert not [e for e in events if e["type"] == "final"]


def test_speech_segment_yields_partial_and_final():
    texts = ["今", "今天", "今天天气"]
    rec = FakeRecognizer(texts=texts)
    # 6 个窗口语音(0..5)，段(0,6)；再喂 4 个静音窗口让段完成
    vad = FakeVad(speech_windows=windows(6, 20), segments=[(0, 6)])
    hub = FakeHub(make_settings(), recognizer=rec, vad_factory=lambda: vad)
    sess = AsrSession(hub, denoise=False)

    events = []
    audio = np.ones(10 * WINDOW, dtype=np.float32) * 0.01
    for i in range(0, 10 * WINDOW, WINDOW):
        events += sess.feed(audio[i:i + WINDOW])

    types = [e["type"] for e in events]
    assert "vad" in types
    assert any(e.get("state") == "speech_start" for e in events if e["type"] == "vad")
    partials = [e for e in events if e["type"] == "partial"]
    assert partials and partials[-1]["text"] == "今天天气"
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1
    f = finals[0]
    assert f["text"] == "今天天气"
    assert f["refined"] is False
    assert f["confidence"] is None
    assert f["segment"] == 0
    assert f["start"] == 0.0 and f["end"] == round(6 * WINDOW / SR, 2)
    assert set(f["timings"]) == {"denoise_ms", "asr_ms", "refine_ms", "punc_ms"}
    assert f["rtf"] is not None
    assert any(e.get("state") == "speech_end" for e in events if e["type"] == "vad")


def test_finish_flushes_pending_speech():
    rec = FakeRecognizer(texts=["你好世界"])
    vad = FakeVad(speech_windows=[True] * 50, segments=[(0, 4)])  # 段未完成
    hub = FakeHub(make_settings(), recognizer=rec, vad_factory=lambda: vad)
    sess = AsrSession(hub, denoise=False)
    sess.feed(np.ones(4 * WINDOW, dtype=np.float32) * 0.01)
    events = sess.finish()
    finals = [e for e in events if e["type"] == "final"]
    assert finals and finals[0]["text"] == "你好世界"
    assert vad.flushed


def test_endpoint_forced_final_no_duplicate():
    # recognizer 端点在 VAD 段完成前触发 → 出 final；随后 VAD 段弹出不得再出重复 final
    rec = FakeRecognizer(texts=["长句子"], endpoint_after=3)
    vad = FakeVad(speech_windows=windows(8, 20), segments=[(0, 8)])
    hub = FakeHub(make_settings(), recognizer=rec, vad_factory=lambda: vad)
    sess = AsrSession(hub, denoise=False)
    events = []
    audio = np.ones(12 * WINDOW, dtype=np.float32) * 0.01
    for i in range(0, 12 * WINDOW, WINDOW):
        events += sess.feed(audio[i:i + WINDOW])
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"] == "长句子"


def test_denoise_path_runs():
    rec = FakeRecognizer(texts=["降噪后"])
    vad = FakeVad(speech_windows=windows(4, 10), segments=[(0, 4)])
    hub = FakeHub(make_settings(), recognizer=rec, vad_factory=lambda: vad,
                  denoiser_factory=lambda: FakeDenoiser(gain=0.9))
    sess = AsrSession(hub, denoise=True)
    events = sess.feed(np.ones(8 * WINDOW, dtype=np.float32) * 0.1)
    finals = [e for e in events if e["type"] == "final"]
    assert finals and finals[0]["timings"]["denoise_ms"] is not None
