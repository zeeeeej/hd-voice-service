# hd-voice-service v1：本机 Docker CPU 流式语音服务器（精简核心 + GTCRN 降噪）

> 状态：**v1 已完成并验证**（2026-09-25，冒烟 11/11、单测 42/42、8 路并发通过）
>
> 实施记录（与原计划的差异/新决策）：
> 1. **TTS 双引擎配置化（`config.yaml tts.engine`）**。曾一度默认 matcha（kokoro int8 在 CPU 上
>    每次 generate 固定开销 ~700ms、RTF≈0.8、首块 ~1.4s；matcha 实测 RTF≈0.05、首块 ~260ms）。
>    **最终决策（2026-09-25，xpl）：默认 kokoro**——商用许可干净（Apache-2.0 + 龙猫数据宽容授权）、
>    103 音色，**正式应用走 GPU**（fp16 权重 + CUDA EP，CPU 固定开销届时消失）；matcha 降级为
>    测试对比可选项（⚠️ baker/CSMSC 数据仅限非商用，不得随产品交付）。
>    详见 [TTS引擎对比.md](./TTS引擎对比.md)。冒烟测试首块阈值按引擎自适应（kokoro@CPU 1600ms / matcha 800ms）。
> 2. sherpa-onnx 1.13.8 Python API 实况：流式解码需 `while recognizer.is_ready(stream):
>    decode_stream(stream)` 循环（单次调用只推进一步）；文本取 `recognizer.get_result(stream)`
>    （`stream.result` 不存在）；降噪 `OnlineSpeechDenoiser.run(samples, sample_rate)` 返回
>    `DenoisedAudio`（内部 1 帧延迟，flush() 收尾）。
> 3. eof 尾补零 0.32s→1.28s（zipformer 右侧上下文 ~0.64s×2，否则句尾丢字，如「测试」被吞）。
> 4. 镜像 v1 不装 ffmpeg（仅 PCM/WAV），healthcheck 用 python stdlib；apt/pip 国内源对容器出口 IP
>    403 → 默认官方 PyPI（本机可直连），换源走 `--build-arg`。
> 5. starlette 锁 <1.0（1.7.0 的 TestClient WS close 语义变化导致测试挂起）；测试 drain 需同时处理
>    `websocket.close` 消息。
> 6. `/metrics` 由 mount 改直接委托（mount 产生 307，Prometheus 抓取器不跟随重定向）。
> 7. TTS 首句快速切分（TextChunker fast_first）：会话首句在第一个软边界提前切，压低首包延迟。
> 8. 模型下载实测：hf-mirror（zipformer 167MB ~21min、kokoro 215MB ~15min、matcha 92MB ~20s）+
>    GitHub Release（gtcrn 0.54MB、silero_vad 实测 643854B 而非文档 2.3MB、vocos 53.9MB）。
> 9. 开发迭代用 docker-compose.override.yml 挂载源码（生产 `docker compose -f docker-compose.yml`）。
> 设计依据：《自建 Linux 流式语音服务器.md》（API 契约 §3.4、流水线 §3.3、模型清单 §3.2）
> 已核实：sherpa-onnx 1.13.8 Python API 实测存在
> `OnlineSpeechDenoiser`(GTCRN 流式降噪) / `VoiceActivityDetector` / `OnlineRecognizer.from_transducer` /
> `OfflineTts.generate(callback=…)`(真流式回调)

## Summary

在空仓库 `hd-voice-service` 中从零实现语音服务器本体：FastAPI + sherpa-onnx（Python，pin `1.13.8`），
单 uvicorn 进程、模型启动加载常驻。v1 为**精简核心**：GTCRN 流式降噪（可开关）→ Silero 流式 VAD →
streaming zipformer zh int8（partial + final）+ Kokoro int8 流式 TTS；**SenseVoice 精修 / 标点 / ITN 延后**
（接口参数保留、显式拒绝）。arm64 原生 Docker 镜像（Dockerfile 架构无关，未来 x86 服务器直接重建即得
amd64），模型经下载脚本落宿主机 `./models/` 后 volume 挂载。提供宿主机 venv 运行的 CLI 测试客户端与
自动化冒烟脚本完成验证。板端改动（§3.6）与应用服务器转发（§3.5）不在本期范围。

## 仓库结构与关键实现

```
server/            FastAPI 应用（app/main.py、config.py、models.py、pipeline/{asr_session,text_chunker,audio}.py、
                   api/{ws_asr,ws_tts,rest,auth}.py）、config.yaml、requirements.txt、Dockerfile、tests/
cli/               asr_cli.py、tts_cli.py、smoke_test.py、requirements.txt（websockets、numpy）
scripts/           download_models.sh(.py)、gen_test_audio.sh
docker-compose.yml、.env.example、README.md
models/            （下载产物，git ignore）
```

- **镜像/部署**：`python:3.11-slim` + apt 装 `ffmpeg`、`curl`；pip 装 `sherpa-onnx==1.13.8 fastapi
  uvicorn[standard] websockets soundfile numpy pydantic-settings prometheus-client python-multipart PyYAML soxr`。
  compose 单服务 `voice`，端口 `8090:8090`，`./models:/opt/voice/models:ro` 挂载，`VOICE_API_KEY` 环境变量
  （默认 dev key），内存 limit 4GB，healthcheck 打 `/v1/health`。本机不加 nginx（生产再加）。
- **模型下载**（`scripts/download_models.sh`，共约 390MB）：repo venv 里的 `huggingface_hub`
  （`HF_ENDPOINT=https://hf-mirror.com`）拉取 `sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30`
  （仅 int8 encoder/decoder/joiner + tokens.txt，owner 依次尝试 csukuangfj → k2-fsa）与
  `kokoro-int8-multi-lang-v1_1`（全量 215MB）；`gtcrn_simple.onnx`(0.54MB) 与 `silero_vad.onnx`(2.3MB)
  走 GitHub Release 直链 curl。幂等：已存在且大小匹配则跳过。
- **模型加载**（`models.py`，lifespan 单例）：GTCRN 降噪器与 Silero VAD 为**每会话实例**（有状态、创建廉价），
  `OnlineRecognizer`(int8 zipformer) 与 `OfflineTts`(kokoro, 默认 speaker `zf_001`) 全局共享权重。
  VAD 参数 threshold=0.5 / min_silence=0.5 / min_speech=0.25 / max_speech=20 / window 512@16k。
  任一加载失败 → 启动即失败，health 报 not ready。全部路径/参数走 `config.yaml`+env，禁止硬编码。
- **WS ASR `/v1/ws/asr`**：严格按 md §3.4 契约实现（ready/vad/partial/final/pong/error 消息、二进制分片上行、
  eof 必须 flush 降噪+VAD 后出 final 再关闭、X-API-Key/X-Request-Id）。v1 差异：仅 `encoding=pcm_s16le`、
  `sample_rate=16000`、`channels=1`（其余 error 帧）；`refine/punctuate/itn=true` → error
  `code=unsupported_option` 并 1008 关闭；final 恒 `refined=false`、`confidence=null`。会话状态机：
  音频分片 → GTCRN 逐块降噪（`denoise` 参数可关，服务端默认 **off**，A/B 后再定）→ 连续喂 recognizer 出
  partial；VAD `speech_end`（或 is_endpoint/max_speech 兜底）触发 final 并 reset 起新 segment；
  无 speech_start 的纯静音段不出 final。解码在 `asyncio.to_thread` 执行避免阻塞事件循环。
  空闲 60s / 单会话 300s 超时关闭。
- **WS TTS `/v1/ws/tts`**：按 §3.4 契约（start/text/flush/eof/cancel；二进制音频块合成即发；
  sentence_done/done）。`text_chunker.py`：遇 `。！？；\n` 必切，缓冲 >30 字按 `，、` 切，禁止单字碎片。
  `OfflineTts.generate(callback=…)` 在 worker 线程跑，回调经 `loop.call_soon_threadsafe` 推 asyncio queue
  流式下发；callback 返回 0 响应 cancel。输出 `pcm_s16le`，16k（soxr 从 24k 重采）或 24k 单声道。
- **REST**：`POST /v1/asr`（整文件 → 可选降噪 → VAD 分段 → 流式 recognizer 文件模式出全文，无标点）、
  `POST /v1/tts`（整段合成返 wav）、`POST /v1/vad`（返 segments JSON）、`POST /v1/denoise`（返降噪后 wav）、
  `GET /v1/models`、`GET /v1/health`、`GET /metrics`（prometheus-client：QPS、各阶段耗时、并发会话、429/5xx）。
  WAV 解析用 soundfile，非 16k 单声道 soxr 重采样。错误约定按 md：400/429/500 分明，透传 X-Request-Id 结构化日志。
- **限流**：信号量 ASR ≤32、TTS ≤4；WS 超限 1013 关闭，REST 返 429。
- **CLI（宿主机 `.venv`）**：
  - `cli/asr_cli.py --wav f.wav [--denoise] [--fast] [--api-key K] [--url ws://localhost:8090/v1/ws/asr]`：
    按实时节奏（或 --fast 全速）推流，partial 同行刷新、final 逐行打印，结束输出 timings/RTF 汇总。
  - `cli/tts_cli.py --text "…" [--speaker zf_001] [--speed 1.0] --out out.wav [--play]`：走 WS 流式收块写 wav，
    `--play` 用 macOS `afplay`；打印首块延迟。
  - `cli/smoke_test.py`：自动断言链——health ready → REST TTS 合成已知文本 → WS ASR 识别该音频 →
    字符重合率 ≥90% → denoise on/off 各跑一遍含噪样本 → TTS 首块延迟 ≤800ms（本机 CPU 参考值）→
    全过退出码 0。
- **测试音频**（`scripts/gen_test_audio.sh`，macOS `say` + ffmpeg）：生成 `clean.wav`（附期望文本）、
  `noisy.wav`（ffmpeg `anoisesrc` 混噪 ~5dB SNR），均 16k 单声道 s16le。

## Test Plan

- **单元测试**（`server/tests/`，pytest + starlette TestClient，模型全用 fake，不需下载）：text_chunker
  切分规则（含 >30 字兜底、单字碎片防护）；audio 解析/重采样；ASR 会话状态机（静音不出 final、eof flush、
  超时）；WS 协议（握手参数校验、unsupported_option、cancel、1013 限流）；REST 错误码约定（坏音频 400 而非 500）。
- **集成验证**（README 步骤，需模型 + Docker）：`download_models.sh` → `docker compose up --build -d` →
  `gen_test_audio.sh` → `python cli/smoke_test.py` 全绿；手动 `asr_cli.py --wav clean.wav`（含 `--denoise`
  A/B 对比 noisy.wav）、`tts_cli.py --play` 可听；`curl /v1/health /v1/models /metrics` 正常。
- **验收标准**：容器 RAM <3GB 稳态；clean.wav 识别字符重合率 ≥90%；TTS WS 首块延迟 ≤800ms；ASR partial
  持续输出、final 带 start/end/timings；`docker compose restart` 后自动恢复 ready。

## Assumptions

- v1 精简核心：无 SenseVoice 精修/标点/ITN（final `refined=false`），相关 WS 参数显式拒绝而非静默忽略；
  模型与代码路径已为其预留（config 化）。
- 仅 `pcm_s16le`/WAV、16kHz 单声道；Opus 编解码（4G 链路）后续用容器内 ffmpeg 补。
- arm64 原生镜像；Dockerfile/compose 不写死架构，生产 x86 上 `docker build` 即得 amd64
  （sherpa-onnx PyPI 双架构均有 wheel）。
- 降噪服务端默认关闭（遵循 md「默认值由 A/B 实测决定」），CLI `--denoise`/`?denoise=true` 开启，
  A/B 数据由 smoke_test 输出后再定默认。
- 本机部署不含 nginx，鉴权仅 `X-API-Key` 中间件（WS 允许 `?api_key=` 查询参数兜底），端口只应绑定内网/本机。
- docker 命令沙箱受限时逐个请求批准；模型源以 hf-mirror + GitHub Release 为准（huggingface.co 被墙）。
- Kokoro 默认音色 `zf_001`（中文女声），speaker 全量可配。
