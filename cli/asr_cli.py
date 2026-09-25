#!/usr/bin/env python3
"""ASR WebSocket 测试客户端。

用法:
  python cli/asr_cli.py --wav .assets/clean.wav [--denoise] [--fast] \
      [--url ws://localhost:8090/v1/ws/asr] [--api-key devkey-local]

按 100ms 分片推流（默认实时节奏，--fast 全速），partial 同行刷新、final 逐行打印，
结束输出识别全文与时延/RTF 汇总。退出码: 0 成功, 2 协议/服务错误, 3 音频不合规。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave

import websockets  # >=14: proxy=None 绕过系统 SOCKS 代理（本地测试）


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != 16000:
            print(f"错误: {path} 必须是 16kHz 单声道 s16le WAV（可用 ffmpeg -i in -ar 16000 -ac 1 转换）", file=sys.stderr)
            sys.exit(3)
        return w.readframes(w.getnframes())


def build_url(args) -> str:
    sep = "&" if "?" in args.url else "?"
    params = ["encoding=pcm_s16le", "sample_rate=16000", "channels=1"]
    if args.denoise:
        params.append("denoise=true")
    if args.no_refine:
        params.append("refine=false")
    if args.no_punctuate:
        params.append("punctuate=false")
    return f"{args.url}{sep}{'&'.join(params)}"


async def run(args) -> int:
    pcm = read_wav(args.wav)
    url = build_url(args)
    headers = {"X-API-Key": args.api_key}
    if args.request_id:
        headers["X-Request-Id"] = args.request_id

    finals: list[dict] = []
    partial_count = 0
    errors: list[str] = []
    t_start = time.monotonic()
    first_partial_at = None

    async with websockets.connect(url, additional_headers=headers, max_size=None, proxy=None) as ws:
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            print(f"错误: 期望 ready 帧, 收到 {ready}", file=sys.stderr)
            return 2
        print(f"[ready] session={ready['session_id']} sr={ready['sample_rate']}")

        stop = asyncio.Event()

        async def sender():
            chunk = 3200  # 100ms @16k s16le
            for i in range(0, len(pcm), chunk):
                if stop.is_set():
                    return
                await ws.send(pcm[i:i + chunk])
                if not args.fast:
                    await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "eof"}))

        async def receiver():
            nonlocal partial_count, first_partial_at
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        continue
                    ev = json.loads(msg)
                    t = ev.get("type")
                    if t == "partial":
                        partial_count += 1
                        if first_partial_at is None:
                            first_partial_at = time.monotonic()
                        if not args.quiet:
                            print(f"\r[partial] {ev['text']}", end="", flush=True)
                    elif t == "final":
                        finals.append(ev)
                        tm = ev.get("timings") or {}
                        tag = "精修" if ev.get("refined") else "流式"
                        print(f"\n[final #{ev['segment']}|{tag}] {ev['start']:.2f}-{ev['end']:.2f}s "
                              f"rtf={ev.get('rtf')} asr_ms={tm.get('asr_ms')} "
                              f"refine_ms={tm.get('refine_ms')} punc_ms={tm.get('punc_ms')} "
                              f"denoise_ms={tm.get('denoise_ms')}\n"
                              f"  文本: {ev['text']}")
                    elif t == "vad":
                        if args.verbose:
                            print(f"[vad] {ev['state']} t={ev['t']}")
                    elif t == "pong":
                        if args.verbose:
                            print("[pong]")
                    elif t == "error":
                        errors.append(f"{ev.get('code')}: {ev.get('message')}")
                        print(f"\n[error] {ev.get('code')}: {ev.get('message')}", file=sys.stderr)
            except websockets.ConnectionClosed:
                pass
            finally:
                stop.set()

        recv_task = asyncio.create_task(receiver())
        await sender()
        await recv_task

    wall = time.monotonic() - t_start
    text = "".join(f["text"] for f in finals)
    print("-" * 60)
    print(f"全文: {text}")
    print(f"finals={len(finals)} partials={partial_count} 音频={len(pcm)/32000:.2f}s "
          f"wall={wall:.2f}s")
    if first_partial_at:
        print(f"首个 partial: {first_partial_at - t_start:.2f}s (含建连)")
    if finals:
        rtfs = [f["rtf"] for f in finals if f.get("rtf")]
        if rtfs:
            print(f"平均 RTF: {sum(rtfs)/len(rtfs):.4f}")
    return 2 if errors else (0 if text else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True, help="16kHz 单声道 s16le WAV 文件")
    ap.add_argument("--url", default="ws://localhost:8090/v1/ws/asr")
    ap.add_argument("--api-key", default="devkey-local")
    ap.add_argument("--denoise", action="store_true", help="开启 GTCRN 流式降噪")
    ap.add_argument("--no-refine", action="store_true", help="关闭句末 SenseVoice 精修（默认跟随服务端配置=开）")
    ap.add_argument("--no-punctuate", action="store_true", help="关闭标点恢复")
    ap.add_argument("--fast", action="store_true", help="不按实时节奏，全速推流")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="不打印 partial 刷新")
    ap.add_argument("--request-id", default=None)
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
