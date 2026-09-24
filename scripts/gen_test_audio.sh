#!/usr/bin/env bash
# 用 macOS say + ffmpeg 生成测试音频：clean.wav(+期望文本) 与 noisy.wav(混粉噪 ~5dB SNR)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=".assets"; mkdir -p "$OUT"
TEXT="今天天气不错，我们一起去公园散步吧。语音服务器冒烟测试。"

# 逐个尝试中文音色，要求产出非静音（部分音色未下载会输出空音频）
VOICE=""
for v in Tingting "Yue (Premium)" Meijia Sinji Flo Sandy; do
  say -v "$v" -o "$OUT/_probe.aiff" "$TEXT" 2>/dev/null || continue
  sz=$(stat -f%z "$OUT/_probe.aiff" 2>/dev/null || stat -c%s "$OUT/_probe.aiff" 2>/dev/null || echo 0)
  if [ "$sz" -gt 20000 ]; then VOICE="$v"; mv "$OUT/_probe.aiff" "$OUT/clean.aiff"; break; fi
done
rm -f "$OUT/_probe.aiff"
if [ -z "$VOICE" ]; then
  echo "未找到可用中文 say 音色，请手动提供 $OUT/clean.wav (16k 单声道 s16le)" >&2; exit 1
fi
echo "使用音色: $VOICE"
ffmpeg -y -loglevel error -i "$OUT/clean.aiff" -ar 16000 -ac 1 -c:a pcm_s16le "$OUT/clean.wav"
rm -f "$OUT/clean.aiff"
echo "$TEXT" > "$OUT/clean.txt"

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/clean.wav")
ffmpeg -y -loglevel error -i "$OUT/clean.wav" \
  -f lavfi -i "anoisesrc=d=${DUR}:c=pink:r=16000:a=0.30" \
  -filter_complex "[0:a]volume=1.6[s];[s][1:a]amix=inputs=2:duration=first:normalize=0" \
  -ar 16000 -ac 1 -c:a pcm_s16le "$OUT/noisy.wav"

ls -la "$OUT"/clean.wav "$OUT"/noisy.wav
echo "已生成: $OUT/clean.wav $OUT/noisy.wav $OUT/clean.txt"
