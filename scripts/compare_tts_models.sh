#!/usr/bin/env bash
# 依次启动三个中文 TTS 引擎并生成同文本试听样本；结束后恢复默认 Melo。
set -euo pipefail

cd "$(dirname "$0")/.."

TEXT="${1:-你好，我是中文语音助手。今天天气不错，我们一起测试数字一百二十三和日期二零二六年九月二十八日。}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${2:-${HOME}/Documents/tmp/tts-compare/${STAMP}}"
PORT="${VOICE_PORT:-8090}"
API_KEY="${VOICE_API_KEY:-devkey-local}"
BASE="http://127.0.0.1:${PORT}"
WS="ws://127.0.0.1:${PORT}/v1/ws/tts"

mkdir -p "$OUT_DIR"

restore_default() {
  echo "恢复默认引擎 melo ..."
  VOICE_TTS_ENGINE=melo docker compose up -d --force-recreate voice >/dev/null || true
}
trap restore_default EXIT INT TERM

wait_ready() {
  local expected="$1"
  local body
  for _ in $(seq 1 90); do
    body="$(curl -fsS "${BASE}/v1/models" 2>/dev/null || true)"
    if [ -n "$body" ] && EXPECTED_ENGINE="$expected" .venv/bin/python -c \
      'import json, os, sys; d=json.load(sys.stdin); sys.exit(0 if d.get("ready") and d["models"]["tts"]["type"] == os.environ["EXPECTED_ENGINE"] else 1)' \
      <<<"$body"; then
      printf '%s\n' "$body" >"$OUT_DIR/${expected}-models.json"
      return 0
    fi
    sleep 2
  done
  echo "等待引擎 ${expected} ready 超时" >&2
  return 1
}

generate() {
  local engine="$1"
  local sid="$2"
  local sample_rate="$3"
  local name="$4"
  echo "生成 ${name}.wav（engine=${engine}, sid=${sid}）..."
  VOICE_TTS_ENGINE="$engine" docker compose up -d --force-recreate voice >/dev/null
  wait_ready "$engine"
  .venv/bin/python cli/tts_cli.py \
    --text "$TEXT" \
    --speaker "$sid" \
    --sample-rate "$sample_rate" \
    --url "$WS" \
    --api-key "$API_KEY" \
    --out "$OUT_DIR/${name}.wav" \
    --fast --quiet | tee "$OUT_DIR/${name}.log"
}

./scripts/download_models.sh

generate melo 0 24000 melo-sid0
generate aishell3 0 16000 aishell3-sid0
generate aishell3 10 16000 aishell3-sid10
generate aishell3 33 16000 aishell3-sid33
generate aishell3 99 16000 aishell3-sid99

echo "注意：Matcha/Baker 仅限非商业试听，不得随商用产品交付。"
generate matcha 0 24000 matcha-sid0

restore_default
trap - EXIT INT TERM

echo "试听文件已生成：$OUT_DIR"
find "$OUT_DIR" -maxdepth 1 -name '*.wav' -print | sort
