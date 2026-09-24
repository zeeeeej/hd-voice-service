#!/usr/bin/env python3
"""TTS WebSocket 测试客户端。

用法:
  python cli/tts_cli.py --text "你好，世界。" --out /tmp/tts.wav [--play] \
      [--speaker 0] [--speed 1.0] [--sample-rate 16000] [--url ws://localhost:8090/v1/ws/tts]

流式接收音频块写 WAV，打印首块延迟与每句 sentence_done。
退出码: 0 成功, 2 协议/服务错误。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import wave

import websockets  # >=14: proxy=None 绕过系统 SOCKS 代理（本地测试）


async def run(args) -> int:
    url = args.url + ("&" if "?" in args.url else "?") + f"api_key={args.api_key}"
    audio = bytearray()
    first_chunk_at = None
    sentences = []
    t_start = time.monotonic()
    sample_rate = args.sample_rate

    async with websockets.connect(url, max_size=None, proxy=None) as ws:
        await ws.send(json.dumps({
            "type": "start", "speaker": args.speaker, "speed": args.speed,
            "format": "pcm_s16le", "sample_rate": sample_rate, "channels": 1,
        }))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            print(f"错误: 期望 ready, 收到 {ready}", file=sys.stderr)
            return 2
        print(f"[ready] session={ready.get('session_id')} sr={ready.get('sample_rate')} speaker={ready.get('speaker')}")

        # 模拟大模型 token 增量：按 --feed-chunk 字符数分批喂入
        text = args.text
        n = args.feed_chunk
        for i in range(0, len(text), n):
            await ws.send(json.dumps({"type": "text", "text": text[i:i + n]}))
            if not args.fast:
                await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "eof"}))

        error = None
        async for msg in ws:
            if isinstance(msg, bytes):
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                    print(f"[first-chunk] {1000 * (first_chunk_at - t_start):.0f} ms")
                audio += msg
                if not args.quiet:
                    print(f"\r[recv] {len(audio) / 2 / sample_rate:.2f}s audio", end="", flush=True)
            else:
                ev = json.loads(msg)
                t = ev.get("type")
                if t == "sentence_done":
                    sentences.append(ev)
                    print(f"\n[sentence_done #{ev['index']}] audio_ms={ev['audio_ms']} text={ev['text']}")
                elif t == "done":
                    print(f"\n[done] total_audio_ms={ev['total_audio_ms']} timings={ev['timings']}")
                elif t == "error":
                    error = ev
                    print(f"\n[error] {ev.get('code')}: {ev.get('message')}", file=sys.stderr)
        if error:
            return 2

    if not audio:
        print("错误: 未收到任何音频", file=sys.stderr)
        return 2
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(audio))
    dur = len(audio) / 2 / sample_rate
    print(f"已写入 {args.out} ({dur:.2f}s, {len(sentences)} 句, "
          f"合成实时率 wall/audio={((time.monotonic() - t_start) / dur):.3f})")
    if args.play:
        subprocess.run(["afplay", args.out], check=False)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True, help="输出 WAV 路径")
    ap.add_argument("--url", default="ws://localhost:8090/v1/ws/tts")
    ap.add_argument("--api-key", default="devkey-local")
    ap.add_argument("--speaker", default="0", help="v1 为整数 sid（kokoro 音色序号）")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--sample-rate", type=int, default=16000, choices=[16000, 24000])
    ap.add_argument("--feed-chunk", type=int, default=8, help="模拟流式喂入的分批字符数")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--play", action="store_true", help="完成后 afplay 播放")
    args = ap.parse_args()
    sp = args.speaker
    args.speaker = int(sp) if str(sp).lstrip("-").isdigit() else sp
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
