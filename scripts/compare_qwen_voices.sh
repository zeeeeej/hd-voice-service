#!/usr/bin/env bash
# 生成 Qwen3-TTS 三个代表音色的中文试听样本。
set -euo pipefail

cd "$(dirname "$0")/.."

TEXT="${1:-你好，我是中文语音助手。今天天气不错，我们一起测试数字一百二十三。}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${2:-${HOME}/Documents/tmp/qwen3-tts-compare/${STAMP}}"
PORT="${VOICE_PORT:-8090}"
API_KEY="${VOICE_API_KEY:-devkey-local}"
BASE="http://127.0.0.1:${PORT}"
WS="ws://127.0.0.1:${PORT}/v1/ws/tts"
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.qwen.yml)

DOCKER_MEMORY_BYTES="$(docker info --format '{{.MemTotal}}')"
MIN_MEMORY_BYTES=$((15 * 1024 * 1024 * 1024))
if [ "$DOCKER_MEMORY_BYTES" -lt "$MIN_MEMORY_BYTES" ]; then
  echo "Docker VM 内存不足：当前约 $((DOCKER_MEMORY_BYTES / 1024 / 1024 / 1024))GB，Qwen3-TTS 请至少分配 16GB。" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

wait_ready() {
  for _ in $(seq 1 180); do
    if curl -fsS "${BASE}/v1/models" 2>/dev/null | .venv/bin/python -c \
      'import json,sys; d=json.load(sys.stdin); assert d.get("ready") and d["models"]["tts"]["type"] == "qwen3"' \
      2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "等待 Qwen3-TTS ready 超时" >&2
  return 1
}

./scripts/download_models.sh --qwen
"${COMPOSE[@]}" up -d --build
wait_ready
curl -fsS "${BASE}/v1/models" >"$OUT_DIR/models.json"

for voice in Vivian Serena Uncle_Fu; do
  echo "生成 ${voice}.wav ..."
  .venv/bin/python cli/tts_cli.py \
    --text "$TEXT" --speaker "$voice" --sample-rate 24000 \
    --url "$WS" --api-key "$API_KEY" --out "$OUT_DIR/${voice}.wav" \
    --fast --quiet | tee "$OUT_DIR/${voice}.log"
done

echo "试听文件已生成：$OUT_DIR"
find "$OUT_DIR" -maxdepth 1 -name '*.wav' -print | sort
