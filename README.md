# hd-voice-service

自建 Linux 流式语音服务器（sherpa-onnx）：**降噪 → VAD → 流式 ASR → 流式 TTS**，
对应用服务器暴露 REST + WebSocket。设计与实测依据见《自建 Linux 流式语音服务器.md》（已移至
`~/Documents/ai/luckfox-audio/`，本仓库不再随附），实施计划与实施记录见 [plan.md](./plan.md)。

**v2（当前）**：GTCRN 流式降噪（可开关）+ Silero VAD + streaming zipformer zh int8 流式识别
+ **SenseVoice 句末精修（含标点/ITN，双通道）** + ct-transformer 流式文本标点 + 流式 TTS。
CPU 推理，Docker 部署；gpu-fp16 profile 文件已备好（`docker-compose.gpu.yml`，待实机验证）。

双通道语义（与设计文档 §3.3 一致）：
- `refine=true`（默认）：句末端点 → 段缓存音频送 SenseVoice（自带标点/ITN）→ `final.refined=true`；
  **精修失败自动回退流式文本**（`refined=false`），不报错
- `punctuate=true`（默认）：仅作用于**流式回退文本**（ct-transformer 补标点），不叠加在 SenseVoice 输出上
- `itn=false`：显式拒绝（精修模型以 ITN-on 加载，全局开关在 `config.yaml refine.use_itn`）

**TTS 四引擎**（`server/config.yaml` → `tts.engine` 或环境变量 `VOICE_TTS_ENGINE`，重启即切换）：

中文支持规则：默认 TTS 面向中文环境，临时默认使用 Melo 中文音色。请求省略 `speaker` 时服务按当前引擎选择默认 sid。

| 引擎 | 定位 | 本机 CPU 实测 | 许可 |
|---|---|---|---|
| `melo`（**临时默认**） | 中文母语单音色，44.1kHz，兼顾中文自然度和 CPU 推理 | RTF ≈1.46，WS 首块 0.88–1.97s | MIT，可商用 |
| `aishell3` | 严格纯中文，174 音色；原生 8kHz | RTF ≈0.10，WS 首块 0.11–0.18s | Apache-2.0 |
| `matcha` | 中文母语单女声，低延迟试听基线 | RTF ≈0.17，WS 首块 ~0.20s | ⚠️ Baker 数据仅限非商用 |
| `kokoro` | 中英多音色历史方案；中文口音不满足当前需求 | RTF ≈0.8，WS 首块 ~1.4s | Apache-2.0 |

> 详细对比、许可链查证与延迟预算见 [TTS引擎对比.md](./TTS引擎对比.md)。

## 快速开始（macOS / Linux，需 Docker）

```bash
# 1. 下载模型（约 1.4GB，含四套 TTS 与精修模型；走 hf-mirror + GitHub Release，幂等）
./scripts/download_models.sh
# GPU 机器额外下载 fp16/fp32 权重（~1.7GB）：./scripts/download_models.sh --gpu

# 2. 构建并启动（arm64 机器上即 arm64 镜像；x86 服务器上构建即得 amd64）
cp .env.example .env          # 按需改 VOICE_API_KEY / VOICE_PORT / VOICE_TTS_ENGINE
docker compose up --build -d
curl -s http://localhost:8090/v1/health   # {"status":"ok","ready":true}

# 3. 生成测试音频（macOS say + ffmpeg）
./scripts/gen_test_audio.sh   # → .assets/clean.wav .assets/noisy.wav

# 4. CLI 验证
python3 -m venv .venv && .venv/bin/pip install -r cli/requirements.txt   # 首次
.venv/bin/python cli/tts_cli.py --text "今天天气不错，我们一起去公园散步吧。" \
    --out ~/Documents/tmp/tts.wav --play
.venv/bin/python cli/asr_cli.py --wav ~/Documents/tmp/tts.wav
.venv/bin/python cli/asr_cli.py --wav .assets/noisy.wav --denoise
.venv/bin/python cli/smoke_test.py                       # 端到端自动断言，全绿退出码 0
```

## API 一览（契约详见设计文档 §3.4）

| 接口 | 说明 |
|---|---|
| `WS /v1/ws/asr` | 流式识别：二进制 pcm_s16le 分片上行；`ready/vad/partial/final/pong/error` 下行；`eof` 收尾 |
| `WS /v1/ws/tts` | 流式合成：`start/text/flush/eof/cancel` 上行；二进制音频块 + `sentence_done/done` 下行 |
| `POST /v1/asr` | 整文件识别（multipart `file`，可选 `?denoise=true`） |
| `POST /v1/tts` | 整段合成（JSON `{text,speaker?,speed,sample_rate}` → wav） |
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
| 模型加载（全部常驻，含精修+标点） | ~6.3s |
| 稳态 RAM | ~1.2GB（kokoro+精修全量） |
| **句末精修附加延迟（6s 语音段）** | **refine ~371ms + 标点由 SenseVoice 内置**（文档目标 ≤500ms ✅） |
| 流式文本标点（ct-transformer） | ~26ms |
| WS ASR（6s 中文语音，实时节奏） | 16 个 partial 连续刷新；final 全文正确；RTF ≈0.09–0.15 |
| REST ASR 整文件 | RTF ≈0.12 |
| 8 路并发 WS ASR | 全部正确、零错误 |
| TTS 首块（matcha） | 257–332ms（目标 ≤800ms） |
| TTS 合成实时率（matcha） | wall/audio ≈0.18 |
| 限流 | asr_sessions=1 时第 2 路正确收到 `overloaded`(1013) |
| 冒烟测试 | `cli/smoke_test.py` 13/13 PASS（含精修断言） |
| ASR 回环重合率（TTS→ASR，精修后） | **1.000** |

降噪 A/B（合成粉噪 ~5dB SNR 样本）：zipformer 对该噪声水平本身鲁棒，denoise on/off 识别结果一致；
GTCRN 链路已验证可用，真实板端麦克风噪声的 A/B 待板子接入后复测（默认 off）。

## 开发

```bash
# 单元测试（不依赖模型/网络，全部 Fake）
python3 -m venv .venv-dev && .venv-dev/bin/pip install -r server/requirements.txt -r server/requirements-dev.txt
cd server && ../.venv-dev/bin/python -m pytest tests/ -q     # 61 passed

# 开发迭代：docker-compose.override.yml 已挂载源码，改代码后
docker compose restart voice      # 即可生效，无需重建镜像
# 生产部署（自包含镜像）：
docker compose -f docker-compose.yml up -d --build

# 配置：server/config.yaml（模型路径/VAD 参数/限流/默认值），
# 环境变量覆盖：VOICE_CONFIG / VOICE_MODELS_DIR / VOICE_API_KEY / VOICE_PORT / VOICE_TTS_ENGINE
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
- [x] v2：SenseVoice 句末精修 + 标点/ITN（双通道 final）+ ct-transformer 流式标点
- [x] v2（预备）：gpu-fp16 profile 文件（compose/config/requirements/下载 --gpu，**待 GPU 实机验证**）
- [ ] v2：Opus 编解码（4G 链路，需镜像加装 ffmpeg）
- [ ] v3：GPU 实机部署 + kokoro fp16 首块延迟复测（届时冒烟阈值收紧回 800ms）
