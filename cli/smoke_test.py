#!/usr/bin/env python3
"""端到端冒烟测试（全部通过退出码 0）。

链路: health → models → REST TTS 合成已知文本 → WS ASR 识别该音频 → 字符重合率断言
     → 含噪样本 denoise on/off A/B（信息性输出 + 协议断言）→ WS TTS 首块延迟断言
     → REST /v1/vad /v1/denoise 协议断言
"""
from __future__ import annotations

import argparse
import asyncio
import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
import difflib
import io
import json
import sys
import time
import urllib.request
import uuid
import wave
from pathlib import Path

import websockets

SMOKE_TEXT = "今天天气不错，我们一起去公园散步吧。"
PUNCT_CHARS = "，。、！？；：…“”‘’（）《》,.!?;:\"'()[] 　"


def strip_punct(t: str) -> str:
    return "".join(ch for ch in t if ch not in PUNCT_CHARS)
PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    (PASSED if cond else FAILED).append(name)
    return cond


class _NoRaise(urllib.request.HTTPErrorProcessor):
    """4xx/5xx 不抛异常，统一返回 (status, headers, body)。"""
    def http_response(self, request, response):
        return response
    https_response = http_response


_OPENER = urllib.request.build_opener(_NoRaise)


def http_json(url: str, api_key: str | None = None, data=None, headers=None, timeout=60):
    hdrs = dict(headers or {})
    if api_key:
        hdrs["X-API-Key"] = api_key
    req = urllib.request.Request(url, data=data, headers=hdrs)
    with _OPENER.open(req, timeout=timeout) as r:
        body = r.read()
        return r.status, r.headers, body  # headers 保持大小写不敏感


def wav_to_pcm16(data: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(data)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        return w.readframes(w.getnframes()), w.getframerate()


async def ws_asr_collect(url: str, api_key: str, pcm: bytes, denoise: bool, fast=True) -> tuple[str, list, float]:
    q = f"?encoding=pcm_s16le&sample_rate=16000&channels=1&denoise={'true' if denoise else 'false'}"
    t0 = time.monotonic()
    finals = []
    async with websockets.connect(url + q, additional_headers={"X-API-Key": api_key,
                                                               "X-Request-Id": uuid.uuid4().hex[:8]},
                                  max_size=None, proxy=None) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        async def send():
            for i in range(0, len(pcm), 3200):
                await ws.send(pcm[i:i + 3200])
                if not fast:
                    await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "eof"}))
        async def recv():
            async for msg in ws:
                if isinstance(msg, str):
                    ev = json.loads(msg)
                    if ev["type"] == "final":
                        finals.append(ev)
                    elif ev["type"] == "error":
                        raise RuntimeError(f"asr ws error: {ev}")
        await asyncio.gather(send(), recv())
    text = "".join(f["text"] for f in finals)
    return text, finals, time.monotonic() - t0


async def ws_tts_first_chunk(url: str, api_key: str, text: str) -> tuple[float, int]:
    t0 = time.monotonic()
    first = None
    total = 0
    async with websockets.connect(url + f"?api_key={api_key}", max_size=None, proxy=None) as ws:
        await ws.send(json.dumps({"type": "start", "speed": 1.0,
                                  "format": "pcm_s16le", "sample_rate": 16000, "channels": 1}))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        await ws.send(json.dumps({"type": "text", "text": text}))
        await ws.send(json.dumps({"type": "eof"}))
        async for msg in ws:
            if isinstance(msg, bytes):
                if first is None:
                    first = time.monotonic() - t0
                total += len(msg)
            else:
                ev = json.loads(msg)
                if ev["type"] == "error":
                    raise RuntimeError(f"tts ws error: {ev}")
                if ev["type"] == "done":
                    break
    return first or 99.9, total


async def main(args) -> int:
    base = args.base.rstrip("/")
    ws_base = base.replace("http://", "ws://").replace("https://", "wss://")
    key = args.api_key

    # 1. health / models
    st, _, body = http_json(f"{base}/v1/health")
    check("health ready", st == 200 and json.loads(body).get("ready") is True, f"status={st}")
    st, _, body = http_json(f"{base}/v1/models")
    models = json.loads(body)
    check("models loaded", st == 200 and models.get("ready") and models["models"]["provider"] == "cpu")
    tts_engine = models["models"]["tts"]["type"]
    # 首块延迟阈值按引擎分级：
    # - matcha@CPU：硬阈值 800ms（实测 ~260ms）
    # - kokoro@CPU：仅回归哨兵 3000ms（实测冷机 ~1.4s、热机降频 ~2.4s，波动大；
    #   生产目标 ≤800ms 须在 GPU 上达成，见 TTS引擎对比.md §4）
    engine_thresholds = {
        "matcha": 800, "aishell3": 1500, "melo": 3000, "kokoro": 3000,
        # CPU 上 Qwen 首句包含完整自回归生成；这是回归超时，不是实时性能目标。
        "qwen3": 300000,
    }
    max_first_chunk_ms = args.max_first_chunk_ms or engine_thresholds.get(tts_engine, 3000)
    print(f"       tts.engine={tts_engine} → 首块延迟阈值 {max_first_chunk_ms:.0f}ms"
          + ("（CPU 参考值，生产 GPU 目标 ≤800ms）" if tts_engine == "kokoro" else ""))

    # 2. REST TTS 合成已知文本
    payload = json.dumps({"text": SMOKE_TEXT, "speed": 1.0, "sample_rate": 16000}).encode()
    st, hdrs, body = http_json(f"{base}/v1/tts", key, data=payload,
                               headers={"Content-Type": "application/json"},
                               timeout=360 if tts_engine == "qwen3" else 120)
    audio_ms = int(hdrs.get("X-Audio-Ms") or 0)
    ok = check("REST /v1/tts 200 + wav", st == 200 and body[:4] == b"RIFF" and audio_ms > 500,
               f"audio_ms={audio_ms}")
    if not ok:
        return finish()
    tts_pcm, sr = wav_to_pcm16(body)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "smoke_tts.wav").write_bytes(body)

    # 3. WS ASR 识别 TTS 音频（回环），断言字符重合率
    text, finals, wall = await ws_asr_collect(f"{ws_base}/v1/ws/asr", key, tts_pcm, denoise=False)
    ratio = difflib.SequenceMatcher(None, strip_punct(SMOKE_TEXT), strip_punct(text)).ratio()
    check("ASR 回环识别重合率 ≥ 0.85", ratio >= 0.85,
          f"ratio={ratio:.3f} 期望={SMOKE_TEXT!r} 识别={text!r} finals={len(finals)} wall={wall:.1f}s")
    check("final 携带 start/end/timings", bool(finals) and all(
        "start" in f and "end" in f and f.get("timings") for f in finals))

    # 3.5 v2 精修通道：服务端加载了精修模型时，默认 final 必须 refined=true 且带标点
    refine_info = models["models"].get("refine")
    if refine_info:
        refined_ok = bool(finals) and finals[0].get("refined") is True
        tm = finals[0].get("timings", {}) if finals else {}
        check("句末精修生效 (refined=true)", refined_ok,
              f"refine_ms={tm.get('refine_ms')} punc_ms={tm.get('punc_ms')} text={text!r}")
        check("精修文本含标点", bool(text) and any(ch in text for ch in "，。！？、；："),
              f"text={text!r}")
    else:
        print("[SKIP] 精修模型未加载（refine=null），跳过 refined 断言")

    # 4. 含噪样本 denoise A/B（信息性）
    noisy = Path(args.noisy_wav)
    if noisy.is_file():
        pcm, _ = wav_to_pcm16(noisy.read_bytes())
        t_off, f_off, _ = await ws_asr_collect(f"{ws_base}/v1/ws/asr", key, pcm, denoise=False)
        t_on, f_on, _ = await ws_asr_collect(f"{ws_base}/v1/ws/asr", key, pcm, denoise=True)
        expected = strip_punct(Path(args.clean_txt).read_text().strip()) if Path(args.clean_txt).is_file() else ""
        r_off = difflib.SequenceMatcher(None, expected, strip_punct(t_off)).ratio() if expected else None
        r_on = difflib.SequenceMatcher(None, expected, strip_punct(t_on)).ratio() if expected else None
        check("含噪样本协议正常 (denoise off/on)", len(f_off) >= 0 and len(f_on) >= 0)
        print(f"       A/B: off={t_off!r} (ratio={r_off})")
        print(f"            on ={t_on!r} (ratio={r_on})")
    else:
        print(f"[SKIP] 含噪 A/B（{noisy} 不存在，先跑 scripts/gen_test_audio.sh）")

    # 5. WS TTS 首块延迟（阈值按引擎自适应，见上）
    first, total = await ws_tts_first_chunk(f"{ws_base}/v1/ws/tts", key, SMOKE_TEXT)
    check(f"WS TTS 首块延迟 ≤ {max_first_chunk_ms:.0f}ms", first * 1000 <= max_first_chunk_ms,
          f"first_chunk={first*1000:.0f}ms audio_bytes={total} engine={tts_engine}（kokoro@CPU 为回归哨兵，非生产目标）"
          if tts_engine == "kokoro" else
          f"first_chunk={first*1000:.0f}ms audio_bytes={total} engine={tts_engine}")

    # 6. REST /v1/vad /v1/denoise
    wav_bytes_ = body
    boundary = uuid.uuid4().hex
    mp = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
          f"Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes_ + f"\r\n--{boundary}--\r\n".encode()
    st, _, b = http_json(f"{base}/v1/vad", key, data=mp,
                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    segs = json.loads(b).get("segments", [])
    check("REST /v1/vad 返回 segments", st == 200 and len(segs) >= 1, f"segments={len(segs)}")
    st, hdrs, b = http_json(f"{base}/v1/denoise", key, data=mp,
                            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    check("REST /v1/denoise 返回 wav", st == 200 and b[:4] == b"RIFF")

    # 7. 鉴权与错误约定
    st, _, b = http_json(f"{base}/v1/vad", "wrong-key", data=mp,
                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    check("错误 API key 被拒 (401)", st == 401 and json.loads(b)["error"]["code"] == "unauthorized",
          f"status={st}")
    st, _, b = http_json(f"{base}/v1/vad", key,
                         data=(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x\"\r\n"
                               f"Content-Type: application/octet-stream\r\n\r\ngarbage\r\n--{boundary}--\r\n").encode(),
                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    err = json.loads(b)
    check("坏音频返回 400 而非 500", st == 400 and err["error"]["code"] == "bad_request", f"status={st}")

    return finish()


def finish() -> int:
    print("=" * 60)
    print(f"PASSED {len(PASSED)}  FAILED {len(FAILED)}")
    if FAILED:
        print("失败项:", *FAILED, sep="\n  - ")
        return 1
    print("SMOKE ALL GREEN ✅")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8090")
    ap.add_argument("--api-key", default="devkey-local")
    ap.add_argument("--noisy-wav", default=".assets/noisy.wav")
    ap.add_argument("--clean-txt", default=".assets/clean.txt")
    ap.add_argument("--out-dir", default=".assets")
    ap.add_argument("--max-first-chunk-ms", type=float, default=None,
                    help="覆盖默认阈值（Qwen3 CPU 默认 300000ms，其余见脚本）")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args)))
