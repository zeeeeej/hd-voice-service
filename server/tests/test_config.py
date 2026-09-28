from app.config import load_settings


def test_default_tts_engine_is_melo(monkeypatch):
    monkeypatch.delenv("VOICE_TTS_ENGINE", raising=False)
    assert load_settings().tts.engine == "melo"


def test_tts_engine_environment_override(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_ENGINE", "aishell3")
    assert load_settings().tts.engine == "aishell3"
