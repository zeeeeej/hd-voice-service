#!/usr/bin/env bash
# 引导仓库 .venv（huggingface_hub）并下载模型到 ./models/
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip >/dev/null
.venv/bin/pip install -q "huggingface_hub>=0.25" -r cli/requirements.txt
exec .venv/bin/python scripts/download_models.py
