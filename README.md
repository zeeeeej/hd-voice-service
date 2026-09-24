# hd-voice-service

自建 Linux 流式语音服务器（sherpa-onnx）：**降噪 → VAD → 流式 ASR → 流式 TTS**，
对应用服务器暴露 REST + WebSocket。设计与实测依据见
[《自建 Linux 流式语音服务器.md》](./自建%20Linux%20流式语音服务器.md)，实施计划见 [plan.md](./plan.md)。

v1（精简核心）：GTCRN 流式降噪（可开关）+ Silero VAD + streaming zipformer zh int8 + 流式 TTS。
SenseVoice 精修 / 标点 / ITN 延后（WS 参数显式拒绝 `unsupported_option`）。CPU 推理，Docker 部署。

**TTS 引擎二选一**（`server/config.yaml` → `tts.engine`，重启即切换）：

| 引擎 | 音色 | 本机实测（M 系列 arm64, Docker） | 许可 |
|---|---|---|---|
| `matcha`（默认） | 单中文女声 (baker) | **RTF ≈0.05，WS 首块 ~260ms** | ⚠️ baker 数据集**仅限非商用** |
| `kokoro` | 中英 103 音色 | RTF ≈0.8，每次合成固定开销 ~700ms，WS 首块 ~1.4s | Apache-2.0 权重 |

> kokoro int8 在 CPU 上达不到 800ms 首块目标（回调粒度粗 + 固定开销大，线程 2→6 收益甚微）；
> 生产 x86/GPU 可复测后切回。切换只改一行配置，接口/协议完全一致。

## 快速开始（macOS / Linux，需 Docker）

```bash
# 1. 下载模型（~390MB，走 hf-mirror + GitHub Release，幂等）
./scripts/download_models.sh

# 2. 构建并启动（arm64 机器上即 arm64 镜像；x86 服务器上构建即得 amd64）
cp .env.example .env          # 按需改 VOICE_API_KEY / VOICE_PORT
docker compose up --build -d
curl -s http://localhost:8090/v1/health   # {"status":"ok","ready":true}

# 3. 生成测试音频（macOS say + ffmpeg）
./scripts/gen_test_audio.sh   # → .assets/clean.wav .assets/noisy.wav

# 4. CLI 验证
python3 -m venv .venv && .venv/bin/pip install -r cli/requirements.txt   # 首次
.venv/bin/python cli/tts_cli.py --text "今天天气不错，我们一起去公园散步吧。" \
    --out /tmp/tts.wav --play
.venv/bin/python cli/asr_cli.py --wav /tmp/tts.wav
.venv/bin/python cli/asr_cli.py --wav .assets/noisy.wav --denoise
.venv/bin/python cli/smoke_test.py                       # 端到端自动断言，全绿退出码 0
```

## API 一览（契约详见设计文档 §3.4）

| 接口 | 说明 |
|---|---|
| `WS /v1/ws/asr` | 流式识别：二进制 pcm_s16le 分片上行；`ready/vad/partial/final/pong/error` 下行；`eof` 收尾 |
| `WS /v1/ws/tts` | 流式合成：`start/text/flush/eof/cancel` 上行；二进制音频块 + `sentence_done/done` 下行 |
| `POST /v1/asr` | 整文件识别（multipart `file`，可选 `?denoise=true`） |
| `POST /v1/tts` | 整段合成（JSON `{text,speaker,speed,sample_rate}` → wav） |
| `POST /v1/vad` | 整文件 VAD 分段 |
| `POST /v1/denoise` | 整文件降噪（调音/排障） |
| `GET /v1/models` `GET /v1/health` `GET /metrics` | 模型信息 / 健康 / Prometheus |

鉴权：`X-API-Key` 头（WS 亦支持 `?api_key=`）；health/models/metrics 免鉴权。
错误约定：400（请求/音频问题）、401、429（过载，WS 用 1013 关闭）、500；透传 `X-Request-Id`。

WS ASR 示例（wscat 风格伪码）：

```
connect ws://localhost:8090/v1/ws/asr?api_key=devkey-local&denoise=true
← {"type":"ready","session_id":...}
→ <binary 100ms pcm_s16le chunks>
← {"type":"vad","state":"speech_start",...}
← {"type":"partial","text":"今天天",...}
← {"type":"final","text":"今天天气不错","refined":false,"start":0.3,"end":2.1,...}
→ {"type":"eof"}
← (flush 后的 final) + close 1000
```

## 本机实测性能（Apple Silicon arm64 / Docker VM 8C8G，2026-09-25）

| 指标 | 实测 |
|---|---|
| 模型加载（全部常驻） | ~6s |
| 稳态 RAM | ~470MB（matcha）/ ~1.2GB（kokoro） |
| WS ASR（6s 中文语音，实时节奏） | 16 个 partial 连续刷新；final 全文正确；RTF ≈0.09–0.15 |
| REST ASR 整文件 | RTF ≈0.12 |
| 8 路并发 WS ASR | 全部正确、零错误 |
| TTS 首块（matcha） | 257–332ms（目标 ≤800ms） |
| TTS 合成实时率（matcha） | wall/audio ≈0.18 |
| 限流 | asr_sessions=1 时第 2 路正确收到 `overloaded`(1013) |
| 冒烟测试 | `cli/smoke_test.py` 11/11 PASS |

降噪 A/B（合成粉噪 ~5dB SNR 样本）：zipformer 对该噪声水平本身鲁棒，denoise on/off 识别结果一致；
GTCRN 链路已验证可用，真实板端麦克风噪声的 A/B 待板子接入后复测（默认 off）。

## 开发

```bash
# 单元测试（不依赖模型/网络，全部 Fake）
python3 -m venv .venv-dev && .venv-dev/bin/pip install -r server/requirements.txt -r server/requirements-dev.txt
cd server && ../.venv-dev/bin/python -m pytest tests/ -q     # 42 passed

# 开发迭代：docker-compose.override.yml 已挂载源码，改代码后
docker compose restart voice      # 即可生效，无需重建镜像
# 生产部署（自包含镜像）：
docker compose -f docker-compose.yml up -d --build

# 配置：server/config.yaml（模型路径/VAD 参数/限流/默认值），
# 环境变量覆盖：VOICE_CONFIG / VOICE_MODELS_DIR / VOICE_API_KEY / VOICE_PORT
```

## 目录

```
server/   FastAPI 应用 + Dockerfile + config.yaml + 单元测试
cli/      测试客户端（asr_cli / tts_cli / smoke_test）
scripts/  模型下载、测试音频生成
models/   模型文件（git 忽略，由 download_models.sh 填充）
.assets/  测试音频产物（git 忽略）
```

## 路线图

- [x] v1 精简核心：GTCRN 降噪 + VAD + 流式 ASR + 流式 TTS + CLI 验证（本机 Docker CPU）
- [ ] v2：SenseVoice 句末精修 + 标点恢复 + ITN（双通道 final）
- [ ] v2：Opus 编解码（4G 链路）
- [ ] v3：GPU profile（CUDA EP + fp16 权重，注意 int8 与 CUDA EP 不兼容）
