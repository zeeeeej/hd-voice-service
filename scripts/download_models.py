#!/usr/bin/env python3
"""下载 v1 所需模型到 ./models/（幂等）。

源：hf-mirror（HF_ENDPOINT 可覆盖）+ GitHub Release。
  - streaming-zipformer-zh-int8-2025-06-30  (~168MB, 仅 int8 权重+tokens)
  - kokoro-int8-multi-lang-v1_1             (~215MB 全量)
  - gtcrn_simple.onnx                        (0.54MB, GitHub Release)
  - silero_vad.onnx                          (2.3MB, GitHub Release)
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

GH_FILES = {
    "gtcrn_simple.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx",
        500_000,
    ),
    "silero_vad.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        600_000,
    ),
    # matcha 轻量 TTS 的 vocoder（低延迟备选引擎用）
    "vocos-22khz-univ.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx",
        50_000_000,
    ),
}

HF_REPOS = [
    ("csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
     "streaming-zipformer-zh-int8-2025-06-30",
     ["*.onnx", "tokens.txt"]),
    ("csukuangfj/kokoro-int8-multi-lang-v1_1",
     "kokoro-int8-multi-lang-v1_1",
     None),  # 全量
    ("csukuangfj/matcha-icefall-zh-baker",
     "matcha-icefall-zh-baker",
     None),  # 低延迟备选 TTS（单女声，注意 baker 数据集仅限非商用）
]


def fetch_github() -> None:
    for name, (url, min_size) in GH_FILES.items():
        dst = MODELS / name
        if dst.is_file() and dst.stat().st_size >= min_size:
            print(f"[skip] {name} 已存在 ({dst.stat().st_size / 1e6:.2f} MB)")
            continue
        print(f"[get ] {name} <- {url}")
        tmp = dst.with_suffix(".part")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, tmp)
                if tmp.stat().st_size < min_size:
                    raise IOError(f"size too small: {tmp.stat().st_size}")
                tmp.rename(dst)
                print(f"       ok {dst.stat().st_size / 1e6:.2f} MB")
                break
            except Exception as e:
                print(f"       retry {attempt + 1}/3: {e}", file=sys.stderr)
                tmp.unlink(missing_ok=True)
        else:
            sys.exit(f"下载失败: {name}")


def fetch_hf() -> None:
    from huggingface_hub import snapshot_download
    for repo_id, subdir, patterns in HF_REPOS:
        dst = MODELS / subdir
        marker = dst / ".download_complete"
        if marker.is_file():
            print(f"[skip] {subdir} 已完成")
            continue
        print(f"[get ] {repo_id} -> {dst} (endpoint={os.environ['HF_ENDPOINT']})")
        snapshot_download(repo_id=repo_id, local_dir=str(dst), allow_patterns=patterns,
                          max_workers=4)
        marker.write_text("ok")
        total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        print(f"       ok {total / 1e6:.1f} MB")


def verify() -> None:
    required = [
        "gtcrn_simple.onnx",
        "silero_vad.onnx",
        "streaming-zipformer-zh-int8-2025-06-30/encoder.int8.onnx",
        "streaming-zipformer-zh-int8-2025-06-30/decoder.onnx",
        "streaming-zipformer-zh-int8-2025-06-30/joiner.int8.onnx",
        "streaming-zipformer-zh-int8-2025-06-30/tokens.txt",
        "kokoro-int8-multi-lang-v1_1/model.int8.onnx",
        "kokoro-int8-multi-lang-v1_1/tokens.txt",
        "kokoro-int8-multi-lang-v1_1/voices.bin",
        "kokoro-int8-multi-lang-v1_1/lexicon-zh.txt",
        "kokoro-int8-multi-lang-v1_1/lexicon-us-en.txt",
        "kokoro-int8-multi-lang-v1_1/espeak-ng-data",
        "kokoro-int8-multi-lang-v1_1/dict",
        "kokoro-int8-multi-lang-v1_1/date-zh.fst",
        "kokoro-int8-multi-lang-v1_1/number-zh.fst",
        "kokoro-int8-multi-lang-v1_1/phone-zh.fst",
        "matcha-icefall-zh-baker/model-steps-3.onnx",
        "matcha-icefall-zh-baker/lexicon.txt",
        "matcha-icefall-zh-baker/tokens.txt",
        "matcha-icefall-zh-baker/dict",
        "matcha-icefall-zh-baker/date.fst",
        "matcha-icefall-zh-baker/number.fst",
        "matcha-icefall-zh-baker/phone.fst",
        "vocos-22khz-univ.onnx",
    ]
    missing = [r for r in required if not (MODELS / r).exists()]
    if missing:
        sys.exit(f"模型缺失: {missing}")
    print(f"[ok  ] 全部模型就绪: {MODELS}")


if __name__ == "__main__":
    MODELS.mkdir(parents=True, exist_ok=True)
    fetch_github()
    fetch_hf()
    verify()
