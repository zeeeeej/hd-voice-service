"""v2 句末精修通道测试（SenseVoice 替身 + 标点替身）。"""
import json

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.conftest import (SR, WINDOW, FakeHub, FakeOfflineRecognizer, FakePunctuation,
                            FakeRecognizer, FakeVad, make_app_with_hub, silence)
from app.config import Settings
from app.pipeline.asr_session import AsrSession
from app.pipeline.audio import f32_to_pcm16_bytes


def make_settings():
    s = Settings()
    s.api_key = "test-key"
    return s


def windows(n_speech, n_total):
    return [True] * n_speech + [False] * (n_total - n_speech)


def hub_with_refine(recognizer=None, vad=None, refine_text="精修文本", fail=False, punct="。"):
    return FakeHub(make_settings(),
                   recognizer=recognizer or FakeRecognizer(texts=["流式文本"]),
                   vad_factory=lambda: vad or FakeVad(),
                   refine_recognizer=FakeOfflineRecognizer(refine_text, fail=fail),
                   punctuation=FakePunctuation(punct) if punct is not None else None)


def feed_windows(sess, n, events):
    audio = np.ones(WINDOW, dtype=np.float32) * 0.01
    for _ in range(n):
        events += sess.feed(audio)


def test_final_refined_no_double_punct():
    vad = FakeVad(speech_windows=windows(12, 26), segments=[(0, 12)])
    hub = hub_with_refine(vad=vad)
    sess = AsrSession(hub, refine=True, punctuate=True)
    events = []
    feed_windows(sess, 16, events)
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1
    f = finals[0]
    assert f["refined"] is True
    assert f["text"] == "精修文本"          # SenseVoice 自带标点，不叠加 ct-transformer
    assert isinstance(f["timings"]["refine_ms"], float)
    assert f["timings"]["punc_ms"] is None
    # 精修确实拿到了段音频（含 pre-roll，≥6 窗）
    fed = hub.refine_recognizer.streams[0].fed
    assert len(fed) >= 12 * WINDOW


def test_refine_failure_falls_back_to_streaming():
    vad = FakeVad(speech_windows=windows(12, 26), segments=[(0, 12)])
    hub = hub_with_refine(vad=vad, fail=True)
    sess = AsrSession(hub, refine=True, punctuate=True)
    events = []
    feed_windows(sess, 16, events)
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["refined"] is False
    assert finals[0]["text"] == "流式文本。"   # 回退 + 标点
    assert not [e for e in events if e["type"] == "error"]


def test_punctuate_applies_to_streaming_fallback():
    # refine=false + punctuate=true → 流式文本补标点
    vad = FakeVad(speech_windows=windows(12, 26), segments=[(0, 12)])
    hub = hub_with_refine(vad=vad)
    sess = AsrSession(hub, refine=False, punctuate=True)
    events = []
    feed_windows(sess, 16, events)
    f = [e for e in events if e["type"] == "final"][0]
    assert f["refined"] is False
    assert f["text"] == "流式文本。"
    assert isinstance(f["timings"]["punc_ms"], float)


def test_refine_failure_fallback_gets_punctuated():
    vad = FakeVad(speech_windows=windows(12, 26), segments=[(0, 12)])
    hub = hub_with_refine(vad=vad, fail=True)
    sess = AsrSession(hub, refine=True, punctuate=True)
    events = []
    feed_windows(sess, 16, events)
    f = [e for e in events if e["type"] == "final"][0]
    assert f["refined"] is False
    assert f["text"] == "流式文本。"   # 回退文本仍有标点


def test_refine_disabled_session():
    vad = FakeVad(speech_windows=windows(12, 26), segments=[(0, 12)])
    hub = hub_with_refine(vad=vad)
    sess = AsrSession(hub, refine=False, punctuate=False)
    events = []
    feed_windows(sess, 16, events)
    f = [e for e in events if e["type"] == "final"][0]
    assert f["refined"] is False and f["text"] == "流式文本"
    assert not hub.refine_recognizer.streams  # 未调用精修


def test_refine_model_missing_graceful():
    vad = FakeVad(speech_windows=windows(12, 26), segments=[(0, 12)])
    hub = FakeHub(make_settings(), recognizer=FakeRecognizer(texts=["流式文本"]),
                  vad_factory=lambda: vad)  # 无 refine_recognizer
    sess = AsrSession(hub, refine=True, punctuate=True)
    events = []
    feed_windows(sess, 16, events)
    f = [e for e in events if e["type"] == "final"][0]
    # 精修与标点模型均缺失 → 纯流式文本，无标点
    assert f["refined"] is False and f["text"] == "流式文本"


def test_seg_pcm_cap():
    vad = FakeVad(speech_windows=[True] * 900, segments=[(0, 899)])
    hub = FakeHub(make_settings(), recognizer=FakeRecognizer(texts=["长"]),
                  vad_factory=lambda: vad)
    sess = AsrSession(hub, refine=True, seg_cap_seconds=25.0)
    events = []
    feed_windows(sess, 800, events)  # 25.6s 音频
    assert sess._seg_pcm_len <= int(25.0 * SR) + WINDOW


def test_ws_refine_handshake(settings):
    hub = hub_with_refine()
    app = make_app_with_hub(settings, hub)
    with TestClient(app) as c:
        # refine=true 不再被拒绝
        with c.websocket_connect("/v1/ws/asr?api_key=test-key&refine=true&punctuate=true") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_text(json.dumps({"type": "eof"}))
        # itn=false 显式拒绝
        with c.websocket_connect("/v1/ws/asr?api_key=test-key&itn=false") as ws:
            ev = ws.receive_json()
            assert ev["type"] == "error" and ev["code"] == "unsupported_option"


def test_ws_refine_final_flow(settings):
    vad = FakeVad(speech_windows=windows(12, 26), segments=[(0, 12)])
    hub = hub_with_refine(vad=vad, punct="！")
    app = make_app_with_hub(settings, hub)
    with TestClient(app) as c:
        with c.websocket_connect("/v1/ws/asr?api_key=test-key&refine=true") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(f32_to_pcm16_bytes(np.ones(16 * WINDOW, dtype=np.float32) * 0.01))
            ws.send_text(json.dumps({"type": "eof"}))
            finals = []
            while True:
                try:
                    m = ws.receive()
                except WebSocketDisconnect:
                    break
                if m["type"] in ("websocket.disconnect", "websocket.close"):
                    break
                if (t := m.get("text")) is not None:
                    ev = json.loads(t)
                    if ev["type"] == "final":
                        finals.append(ev)
            assert finals and finals[0]["refined"] is True
            assert finals[0]["text"] == "精修文本"


def test_rest_refine_path(settings):
    import io
    import soundfile as sf
    hub = hub_with_refine()
    app = make_app_with_hub(settings, hub)
    x = 0.1 * np.sin(np.linspace(0, 2000, SR, dtype=np.float32))
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    with TestClient(app) as c:
        r = c.post("/v1/asr", headers={"X-API-Key": "test-key"},
                   files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
        assert r.status_code == 200
        body = r.json()
        assert body["refined"] is True
        assert body["text"] == "精修文本"
        # refine=false 回退流式路径
        r = c.post("/v1/asr?refine=false", headers={"X-API-Key": "test-key"},
                   files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
        assert r.json()["refined"] is False
