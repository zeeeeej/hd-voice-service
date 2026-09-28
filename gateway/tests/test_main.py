from __future__ import annotations

import struct
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import Settings, create_app


def wav_bytes() -> bytes:
    return b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + b"\x00" * 24


def make_client(handler):
    async_client = httpx.AsyncClient(
        base_url="http://voice:8090",
        transport=httpx.MockTransport(handler),
    )
    settings = Settings(
        gateway_api_key="board-key",
        voice_api_key="voice-key",
        voice_base_url="http://voice:8090",
        max_upload_mb=1,
    )
    return TestClient(create_app(settings, async_client)), async_client


def test_voice_command_refines_and_synthesizes():
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        assert request.headers["x-api-key"] == "voice-key"
        assert request.headers["x-request-id"] == "test-request"
        if request.url.path == "/v1/asr":
            assert request.url.params["refine"] == "true"
            return httpx.Response(200, json={"text": "打开灯。"})
        if request.url.path == "/v1/tts":
            assert request.content
            assert "我收到了命令：打开灯。" in request.content.decode("utf-8")
            return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})
        raise AssertionError(request.url)

    client, async_client = make_client(handler)
    with patch("app.main.log.info") as log_info:
        with client:
            response = client.post(
                "/v1/voice-command",
                headers={"X-API-Key": "board-key", "X-Request-Id": "test-request"},
                files={"file": ("command.wav", wav_bytes(), "audio/wav")},
            )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-request-id"] == "test-request"
    assert response.content == wav_bytes()
    assert len(calls) == 2
    log_info.assert_called_once()
    log_payload = log_info.call_args.args[1]
    assert '"request_id": "test-request"' in log_payload
    assert '"recognized_text": "打开灯。"' in log_payload
    assert '"reply_text": "我收到了命令：打开灯。"' in log_payload
    import asyncio
    asyncio.run(async_client.aclose())


def test_empty_refine_falls_back_to_streaming_asr():
    refinements = []

    def handler(request: httpx.Request):
        if request.url.path == "/v1/asr":
            refinements.append(request.url.params["refine"])
            text = "" if len(refinements) == 1 else "播放音乐"
            return httpx.Response(200, json={"text": text})
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})

    client, async_client = make_client(handler)
    with client:
        response = client.post(
            "/v1/voice-command",
            headers={"X-API-Key": "board-key"},
            files={"file": ("command.wav", wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 200
    assert refinements == ["true", "false"]
    import asyncio
    asyncio.run(async_client.aclose())


def test_no_recognized_speech_returns_422():
    def handler(request: httpx.Request):
        return httpx.Response(200, json={"text": ""})

    client, async_client = make_client(handler)
    with client:
        response = client.post(
            "/v1/voice-command",
            headers={"X-API-Key": "board-key"},
            files={"file": ("command.wav", wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "speech_not_recognized"
    import asyncio
    asyncio.run(async_client.aclose())


def test_auth_upload_limit_and_downstream_errors():
    def handler(request: httpx.Request):
        return httpx.Response(500, json={"error": "failed"})

    client, async_client = make_client(handler)
    with client:
        missing_key = client.post(
            "/v1/voice-command",
            files={"file": ("command.wav", wav_bytes(), "audio/wav")},
        )
        too_large = client.post(
            "/v1/voice-command",
            headers={"X-API-Key": "board-key"},
            files={"file": ("command.wav", b"x" * (1024 * 1024 + 1), "audio/wav")},
        )
        downstream = client.post(
            "/v1/voice-command",
            headers={"X-API-Key": "board-key"},
            files={"file": ("command.wav", wav_bytes(), "audio/wav")},
        )
    assert missing_key.status_code == 401
    assert too_large.status_code == 413
    assert downstream.status_code == 502
    assert downstream.json()["error"]["code"] == "downstream_error"
    import asyncio
    asyncio.run(async_client.aclose())


def test_health_reflects_voice_service_state():
    def handler(request: httpx.Request):
        return httpx.Response(200, json={"status": "ok", "ready": True})

    client, async_client = make_client(handler)
    with client:
        response = client.get("/v1/health", headers={"X-Request-Id": "health-request"})
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.headers["x-request-id"] == "health-request"
    import asyncio
    asyncio.run(async_client.aclose())


def test_downstream_timeout_returns_504():
    def handler(request: httpx.Request):
        raise httpx.ReadTimeout("slow voice service", request=request)

    client, async_client = make_client(handler)
    with client:
        response = client.post(
            "/v1/voice-command",
            headers={"X-API-Key": "board-key"},
            files={"file": ("command.wav", wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "downstream_timeout"
    import asyncio
    asyncio.run(async_client.aclose())


def test_invalid_tts_audio_is_rejected():
    def handler(request: httpx.Request):
        if request.url.path == "/v1/asr":
            return httpx.Response(200, json={"text": "开灯"})
        return httpx.Response(200, content=b'{"error":"not audio"}', headers={"content-type": "application/json"})

    client, async_client = make_client(handler)
    with client:
        response = client.post(
            "/v1/voice-command",
            headers={"X-API-Key": "board-key"},
            files={"file": ("command.wav", wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "downstream_invalid_response"
    import asyncio
    asyncio.run(async_client.aclose())
