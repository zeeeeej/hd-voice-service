import httpx
import numpy as np
import pytest

from app.qwen_tts import QwenRemoteTts, split_text


def _client(handler):
    return httpx.Client(base_url="http://qwen-test", transport=httpx.MockTransport(handler))


def test_split_text_prefers_chinese_punctuation_and_hard_limit():
    chunks = split_text("第一句很短。第二句比较长，需要继续切分。最后一句。", 12)
    assert "".join(chunks) == "第一句很短。第二句比较长，需要继续切分。最后一句。"
    assert all(len(chunk) <= 12 for chunk in chunks)


def test_qwen_adapter_metadata_generate_and_callback():
    audio = np.linspace(-0.2, 0.2, 5000, dtype="<f4")
    posts = []

    def handler(request):
        if request.url.path == "/metadata":
            return httpx.Response(200, json={
                "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "sample_rate": 24000,
                "speakers": ["Vivian", "Serena"],
            })
        posts.append(request)
        return httpx.Response(200, content=audio.tobytes(), headers={"X-Sample-Rate": "24000"})

    chunks = []
    tts = QwenRemoteTts("http://qwen-test", max_chars=5, client=_client(handler))
    result = tts.generate("你好世界。再次问好。", "Vivian", 1.0,
                          lambda samples, progress: chunks.append(samples.copy()) or 1)
    assert tts.sample_rate == 24000
    assert tts.speaker_names == ("Vivian", "Serena")
    assert len(posts) == 2
    assert len(result.samples) == len(audio) * 2
    assert sum(len(chunk) for chunk in chunks) == len(result.samples)


def test_qwen_adapter_surfaces_sidecar_error():
    def handler(request):
        if request.url.path == "/metadata":
            return httpx.Response(200, json={"sample_rate": 24000, "speakers": ["Vivian"]})
        return httpx.Response(503, text="model failed")

    tts = QwenRemoteTts("http://qwen-test", client=_client(handler))
    with pytest.raises(RuntimeError, match="synthesis failed"):
        tts.generate("你好", "Vivian")
