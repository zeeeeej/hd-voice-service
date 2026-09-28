from app.config import load_settings


def test_default_tts_engine_is_melo(monkeypatch):
    monkeypatch.delenv("VOICE_TTS_ENGINE", raising=False)
    assert load_settings().tts.engine == "melo"


def test_tts_engine_environment_override(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_ENGINE", "aishell3")
    assert load_settings().tts.engine == "aishell3"


def test_qwen_environment_overrides(monkeypatch):
    monkeypatch.setenv("VOICE_QWEN_TTS_URL", "http://qwen-test:9000")
    monkeypatch.setenv("VOICE_QWEN_TTS_TIMEOUT", "123")
    monkeypatch.setenv("VOICE_TTS_SESSIONS", "1")
    settings = load_settings()
    assert settings.tts.qwen_url == "http://qwen-test:9000"
    assert settings.tts.qwen_timeout == 123
    assert settings.limits.tts_sessions == 1
