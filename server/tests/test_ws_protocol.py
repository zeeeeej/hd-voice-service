import json

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.conftest import (FakeHub, FakeRecognizer, FakeTts, FakeVad,
                            make_app_with_hub, silence)
from app.config import Settings
from app.pipeline.audio import f32_to_pcm16_bytes
from fastapi.testclient import TestClient


def drain(ws):
    """收消息直到服务端关闭。"""
    msgs = []
    while True:
        try:
            m = ws.receive()
        except WebSocketDisconnect:
            return msgs
        if m["type"] in ("websocket.disconnect", "websocket.close"):
            return msgs
        msgs.append(m)


def jsons(msgs):
    out = []
    for m in msgs:
        if (t := m.get("text")) is not None:
            out.append(json.loads(t))
    return out


def test_ws_asr_ready_and_eof(client):
    with client.websocket_connect("/v1/ws/asr?api_key=test-key") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["sample_rate"] == 16000
        assert ready["models"]["provider"] == "cpu"
        ws.send_bytes(f32_to_pcm16_bytes(silence(0.2)))
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json()["type"] == "pong"
        ws.send_text(json.dumps({"type": "eof"}))
        msgs = drain(ws)
        assert not [m for m in jsons(msgs) if m["type"] == "error"]


def test_ws_asr_unauthorized(client):
    with client.websocket_connect("/v1/ws/asr?api_key=wrong") as ws:
        ev = ws.receive_json()
        assert ev["type"] == "error" and ev["code"] == "unauthorized"


def test_ws_asr_unsupported_options(client):
    # v2：refine/punctuate 已支持；itn=false 显式拒绝（模型以 ITN-on 加载）
    for qs in ("itn=false", "encoding=opus", "sample_rate=8000", "channels=2"):
        with client.websocket_connect(f"/v1/ws/asr?api_key=test-key&{qs}") as ws:
            ev = ws.receive_json()
            assert ev["type"] == "error" and ev["code"] == "unsupported_option", qs


def test_ws_asr_overload_rejected(settings):
    settings.limits.asr_sessions = 0
    app = make_app_with_hub(settings, FakeHub(settings))
    with TestClient(app) as c:
        with c.websocket_connect("/v1/ws/asr?api_key=test-key") as ws:
            ev = ws.receive_json()
            assert ev["type"] == "error" and ev["code"] == "overloaded"


def test_ws_asr_bad_control_frame(client):
    with client.websocket_connect("/v1/ws/asr?api_key=test-key") as ws:
        ws.receive_json()
        ws.send_text("not json")
        msgs = drain(ws)
        codes = [m["code"] for m in jsons(msgs) if m["type"] == "error"]
        assert "bad_request" in codes


def test_ws_asr_config_frame_before_audio(client):
    with client.websocket_connect("/v1/ws/asr?api_key=test-key") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "config", "denoise": True}))
        ws.send_bytes(f32_to_pcm16_bytes(silence(0.1)))
        ws.send_text(json.dumps({"type": "eof"}))
        msgs = drain(ws)
        assert not [m for m in jsons(msgs) if m["type"] == "error"]


def test_ws_tts_full_flow(settings):
    hub = FakeHub(settings, tts=FakeTts())
    app = make_app_with_hub(settings, hub)
    with TestClient(app) as c:
        with c.websocket_connect("/v1/ws/tts?api_key=test-key") as ws:
            ws.send_text(json.dumps({"type": "start", "speaker": 0, "speed": 1.0,
                                     "format": "pcm_s16le", "sample_rate": 16000}))
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            ws.send_text(json.dumps({"type": "text", "text": "你好。世界！"}))
            ws.send_text(json.dumps({"type": "eof"}))
            bins = 0
            sentence_dones = []
            done = None
            for m in drain(ws):
                if m.get("bytes"):
                    bins += 1
                elif (t := m.get("text")) is not None:
                    ev = json.loads(t)
                    if ev["type"] == "sentence_done":
                        sentence_dones.append(ev)
                    elif ev["type"] == "done":
                        done = ev
                    elif ev["type"] == "error":
                        pytest.fail(f"tts error: {ev}")
            assert bins >= 2  # 两句、每句多块
            assert [s["index"] for s in sentence_dones] == [0, 1]
            assert sentence_dones[0]["text"] == "你好。"
            assert done is not None and done["total_audio_ms"] > 0
            assert done["timings"]["first_chunk_ms"] is not None
            # 合成调用收到的是切好的整句
            assert [c[0] for c in hub.tts.calls] == ["你好。", "世界！"]


def test_ws_tts_incremental_feed(settings):
    hub = FakeHub(settings, tts=FakeTts())
    app = make_app_with_hub(settings, hub)
    with TestClient(app) as c:
        with c.websocket_connect("/v1/ws/tts?api_key=test-key") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            assert ws.receive_json()["type"] == "ready"
            for piece in ["今天", "天气", "不错。", "走吧"]:
                ws.send_text(json.dumps({"type": "text", "text": piece}))
            ws.send_text(json.dumps({"type": "flush"}))
            ws.send_text(json.dumps({"type": "eof"}))
            drain(ws)
            assert [c_[0] for c_ in hub.tts.calls] == ["今天天气不错。", "走吧"]


def test_ws_tts_bad_start(settings):
    hub = FakeHub(settings, tts=FakeTts())
    app = make_app_with_hub(settings, hub)
    with TestClient(app) as c:
        with c.websocket_connect("/v1/ws/tts?api_key=test-key") as ws:
            ws.send_text(json.dumps({"type": "start", "format": "opus"}))
            ev = ws.receive_json()
            assert ev["type"] == "error" and ev["code"] == "unsupported_option"
        with c.websocket_connect("/v1/ws/tts?api_key=test-key") as ws:
            ws.send_text(json.dumps({"type": "start", "speaker": "zf_001"}))
            ev = ws.receive_json()
            assert ev["type"] == "error" and ev["code"] == "bad_request"
        with c.websocket_connect("/v1/ws/tts?api_key=test-key") as ws:
            ws.send_text(json.dumps({"type": "text", "text": "x"}))
            ev = ws.receive_json()
            assert ev["type"] == "error" and ev["code"] == "bad_request"
