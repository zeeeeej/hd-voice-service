#!/usr/bin/env python3
"""下载模型到 ./models/（幂等；--gpu/--qwen 追加可选权重）。

源：hf-mirror（HF_ENDPOINT 可覆盖）+ GitHub Release。
  - streaming-zipformer-zh-int8-2025-06-30  (~168MB, 仅 int8 权重+tokens)
  - kokoro-int8-multi-lang-v1_1             (~215MB 全量)
  - vits-melo-tts-zh_en                      (~196MB，中文母语单音色)
  - vits-icefall-zh-aishell3                 (~211MB，纯中文 174 音色，含规则库)
  - gtcrn_simple.onnx                        (0.54MB, GitHub Release)
  - silero_vad.onnx                          (2.3MB, GitHub Release)
  - Qwen3-TTS-12Hz-0.6B-CustomVoice          (~2.5GB, --qwen)
"""
from __future__ import annotations

import os
import sys
import tarfile
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

TTS_ARCHIVES = {
    "vits-melo-tts-zh_en": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
        "vits-melo-tts-zh_en.tar.bz2",
        ["model.onnx", "lexicon.txt", "tokens.txt", "date.fst", "number.fst",
         "phone.fst", "LICENSE"],
    ),
    "vits-icefall-zh-aishell3": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
        "vits-icefall-zh-aishell3.tar.bz2",
        ["model.onnx", "lexicon.txt", "tokens.txt", "date.fst", "number.fst",
         "phone.fst"],
    ),
}

GPU_REPOS = [
    # --gpu：fp16/fp32 权重（CUDA EP 不支持 int8 量化算子）
    ("csukuangfj/sherpa-onnx-streaming-zipformer-zh-fp16-2025-06-30",
     "streaming-zipformer-zh-fp16-2025-06-30",
     ["*.onnx", "tokens.txt"]),                                  # ~314MB
    ("csukuangfj/kokoro-multi-lang-v1_1",
     "kokoro-multi-lang-v1_1",
     None),                                                       # fp32 ~427MB
    ("csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
     "sense-voice-fp32-tmp",
     ["model.onnx", "tokens.txt"]),                               # fp32 938MB
]

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
    # ---- v2 句末精修通道 ----
    ("csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
     "sense-voice-zh-en-ja-ko-yue-2024-07-17",
     ["model.int8.onnx", "tokens.txt"]),          # 239MB，离线精修 ASR
    ("csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
     "punct-ct-transformer-zh-en-vocab272727-2024-04-12",
     ["model.onnx", "tokens.json"]),              # 294MB，标点恢复
]

QWEN_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN_DIR = "Qwen3-TTS-12Hz-0.6B-CustomVoice"


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


def fetch_tts_archives() -> None:
    """下载 sherpa-onnx 官方 TTS 包；限制归档只能写入各自模型目录。"""
    for subdir, (url, required) in TTS_ARCHIVES.items():
        dst = MODELS / subdir
        marker = dst / ".download_complete"
        if marker.is_file() and all((dst / name).is_file() for name in required):
            print(f"[skip] {subdir} 已完成")
            continue
        archive = MODELS / f".{subdir}.tar.bz2.part"
        print(f"[get ] {subdir} <- {url}")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, archive)
                with tarfile.open(archive, "r:bz2") as tf:
                    members = tf.getmembers()
                    for member in members:
                        parts = Path(member.name).parts
                        if not parts or parts[0] != subdir or member.issym() or member.islnk():
                            raise IOError(f"unsafe archive member: {member.name!r}")
                    tf.extractall(MODELS, members=members, filter="data")
                missing = [name for name in required if not (dst / name).is_file()]
                if missing:
                    raise IOError(f"archive missing required files: {missing}")
                marker.write_text("ok", encoding="utf-8")
                archive.unlink(missing_ok=True)
                total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
                print(f"       ok {total / 1e6:.1f} MB")
                break
            except Exception as e:
                print(f"       retry {attempt + 1}/3: {e}", file=sys.stderr)
                archive.unlink(missing_ok=True)
        else:
            sys.exit(f"下载失败: {subdir}")


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
        "vits-melo-tts-zh_en/model.onnx",
        "vits-melo-tts-zh_en/lexicon.txt",
        "vits-melo-tts-zh_en/tokens.txt",
        "vits-melo-tts-zh_en/date.fst",
        "vits-melo-tts-zh_en/number.fst",
        "vits-melo-tts-zh_en/phone.fst",
        "vits-melo-tts-zh_en/LICENSE",
        "vits-icefall-zh-aishell3/model.onnx",
        "vits-icefall-zh-aishell3/lexicon.txt",
        "vits-icefall-zh-aishell3/tokens.txt",
        "vits-icefall-zh-aishell3/date.fst",
        "vits-icefall-zh-aishell3/number.fst",
        "vits-icefall-zh-aishell3/phone.fst",
        "sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
        "sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
        "punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx",
        "punct-ct-transformer-zh-en-vocab272727-2024-04-12/tokens.json",
    ]
    missing = [r for r in required if not (MODELS / r).exists()]
    if missing:
        sys.exit(f"模型缺失: {missing}")
    print(f"[ok  ] 全部模型就绪: {MODELS}")


def fetch_gpu() -> None:
    """下载 gpu-fp16 profile 权重；SenseVoice fp32 并入既有目录。"""
    from huggingface_hub import snapshot_download
    for repo_id, subdir, patterns in GPU_REPOS:
        dst = MODELS / subdir
        if subdir == "sense-voice-fp32-tmp":
            dst = MODELS / "sense-voice-zh-en-ja-ko-yue-2024-07-17"
        marker = dst / f".gpu_download_complete"
        if marker.is_file():
            print(f"[skip] {subdir} (gpu) 已完成")
            continue
        print(f"[gpu ] {repo_id} -> {dst}")
        snapshot_download(repo_id=repo_id, local_dir=str(dst), allow_patterns=patterns,
                          max_workers=4)
        marker.write_text("ok")


def fetch_qwen() -> None:
    """下载 Qwen3-TTS 0.6B CustomVoice 全量权重及语音 tokenizer。"""
    from huggingface_hub import snapshot_download

    dst = MODELS / QWEN_DIR
    marker = dst / ".download_complete"
    required = ("config.json", "model.safetensors")
    if marker.is_file() and all((dst / name).is_file() for name in required):
        print(f"[skip] {QWEN_DIR} 已完成")
        return
    print(f"[qwen] {QWEN_REPO} -> {dst} (endpoint={os.environ['HF_ENDPOINT']})")
    snapshot_download(repo_id=QWEN_REPO, local_dir=str(dst), max_workers=4)
    missing = [name for name in required if not (dst / name).is_file()]
    if missing:
        sys.exit(f"Qwen3-TTS 模型缺失: {missing}")
    marker.write_text("ok", encoding="utf-8")
    total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"       ok {total / 1e9:.2f} GB")


def verify_qwen() -> None:
    required = [
        f"{QWEN_DIR}/config.json",
        f"{QWEN_DIR}/model.safetensors",
        f"{QWEN_DIR}/.download_complete",
    ]
    missing = [rel for rel in required if not (MODELS / rel).is_file()]
    if missing:
        sys.exit(f"Qwen3-TTS 模型缺失: {missing}")
    print(f"[ok  ] Qwen3-TTS 模型就绪: {MODELS / QWEN_DIR}")


if __name__ == "__main__":
    MODELS.mkdir(parents=True, exist_ok=True)
    fetch_github()
    fetch_tts_archives()
    fetch_hf()
    verify()
    if "--gpu" in sys.argv:
        fetch_gpu()
        print("[ok  ] GPU 权重就绪")
    if "--qwen" in sys.argv:
        fetch_qwen()
        verify_qwen()
