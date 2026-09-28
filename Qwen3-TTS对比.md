# Qwen3-TTS / Melo / AISHELL3 对比与接入决策

> 更新日期：2026-09-28
>
> 当前接入模型：`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

## 1. 结论

如果优先解决“像外国人说中文”的听感，Qwen3-TTS 0.6B CustomVoice 是当前三个候选中
最值得先试听的方案：它有 5 个以中文为母语的预置音色，模型代际和韵律能力也明显新于
Melo 与 AISHELL3 VITS。当前实现以 `Vivian` 作为中文默认音色，同时保留 `Serena`、
`Uncle_Fu`、`Dylan`、`Eric` 四个中文音色供选择。

代价是资源消耗和延迟显著增加。Melo、AISHELL3 是轻量 ONNX 模型；Qwen3-TTS 是
0.6B 参数的 PyTorch 生成模型。本项目因此将它放在独立 sidecar 容器中，并要求 Docker
Desktop 至少分配 16GB 内存。CPU 首句延迟只作为功能验证，生产低延迟仍建议 CUDA。

## 2. 核心对比

| 维度 | Qwen3-TTS 0.6B CustomVoice | Melo 中文模型 | AISHELL3 VITS |
|---|---|---|---|
| 当前服务引擎名 | `qwen3` | `melo` | `aishell3` |
| 中文定位 | 多语言模型，但有 5 个中文母语/方言原生音色 | 中文模型，支持中英混读 | 仅中文语料训练 |
| 可用音色 | 9 个预置音色，其中 5 个中文原生 | 当前 ONNX 导出 1 个 | 174 个 sid |
| 中文默认音色 | `Vivian` | sid `0` | sid `0` |
| 原生采样率 | 24kHz | 44.1kHz | 8kHz |
| 模型/运行时 | 0.6B Transformer、PyTorch | VITS、ONNX Runtime | VITS、ONNX Runtime |
| 当前本机速度 | RTF ≈9.8–10.6，WS 首块 13–27s | RTF ≈1.46 | RTF ≈0.10 |
| 资源级别 | 高；模型包约 2.5GB，建议 ≥16GB Docker 内存 | 中 | 低 |
| 适合场景 | 中文自然度、角色感优先 | 单音色中文、部署简单 | 低延迟、多说话人、窄带语音 |
| 主要限制 | CPU 慢；0.6B 无 instruction 控制；当前适配不是真流式 | 只有一个音色，韵律上限较低 | 原生 8kHz，音质和自然度受限 |
| 许可 | Apache-2.0 | MIT | 模型工具链宽松；商用时仍应复核 AISHELL3 语料条款 |

来源：[Qwen3-TTS 官方仓库](https://github.com/QwenLM/Qwen3-TTS)、
[MeloTTS 官方仓库](https://github.com/myshell-ai/MeloTTS)、
[sherpa-onnx VITS 模型说明](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html)。

## 3. Qwen3-TTS 0.6B 的 9 个预置音色

| 编号 | 名称 | 原生语言/方言 | 官方描述摘要 |
|---:|---|---|---|
| 0 | `Vivian` | 中文 | 明亮、略带锋芒的年轻女声 |
| 1 | `Serena` | 中文 | 温暖、柔和的年轻女声 |
| 2 | `Uncle_Fu` | 中文 | 低沉、醇厚的成熟男声 |
| 3 | `Dylan` | 中文/北京话 | 清晰自然的年轻男声 |
| 4 | `Eric` | 中文/四川话 | 活泼、略带沙哑感的成都男声 |
| 5 | `Ryan` | 英文 | 节奏感较强的男声 |
| 6 | `Aiden` | 英文/美式 | 阳光、清晰中频的男声 |
| 7 | `Ono_Anna` | 日文 | 轻快俏皮的女声 |
| 8 | `Sohee` | 韩文 | 温暖、情感丰富的女声 |

所有音色都能生成模型支持的其他语言，但官方建议优先使用音色的原生语言。因此中文默认
选择 `Vivian`，而不是英文、日文或韩文原生音色。音色表见
[Qwen3-TTS 官方 README](https://github.com/QwenLM/Qwen3-TTS#custom-voice-generate)。

## 4. 为什么选择 0.6B CustomVoice

- `CustomVoice` 直接提供 9 个固定音色，最符合当前“文本 + 音色 → 语音”的 API。
- 0.6B 比 1.7B 更适合先在本机 CPU 验证，也降低后续 GPU 显存要求。
- 0.6B CustomVoice **不支持 instruction 风格控制**。虽然 1.7B CustomVoice 支持用自然语言
  控制情绪和风格，但本版接口没有暴露 `instruct`，也不会静默假装支持。
- 语音克隆需要 `Base` 变体，VoiceDesign 需要 1.7B VoiceDesign；这两类需求不属于本次范围。

## 5. 本项目的实现边界

部署拓扑如下：主 `voice` 服务继续负责公开 REST/WS、鉴权、限流、重采样；私有
`qwen-tts` sidecar 只负责加载 Qwen 权重和合成 24kHz float32 PCM。这样不会让原来的
sherpa-onnx 镜像承担 PyTorch 依赖，也能独立迁移到 CUDA 服务器。

公开 API 保持不变：

```json
{"text":"你好，欢迎使用中文语音服务。","speaker":"Vivian","speed":1.0,"sample_rate":24000}
```

数字音色编号仍可使用，例如 `speaker: 2` 等价于 `Uncle_Fu`；音色名不区分大小写。
`GET /v1/models` 会返回 `speaker_names`。

需要特别说明：当前官方高层 Python API 在 `generate_custom_voice()` 完成后一次返回完整波形。
本项目 WS 会按句调用模型，再将完成的句子切成约 100ms PCM 块发送，因此是协议兼容的
“句级流式”，不是模型边生成边输出的真流式。后续若首块延迟有硬指标，应接官方底层流式
接口或 vLLM/CUDA 后端，而不是把当前分块误当成真流式。

## 6. 实机验收结果

环境：Apple Silicon ARM64、Docker VM 16GB、CPU float32 + SDPA。2026-09-28 实测：

1. `/v1/models`：`type=qwen3`、默认 `Vivian`、9 个音色，模型加载 4.25s；
2. REST 与 WS 均生成有效 WAV/PCM，端到端冒烟 `13/13 PASS`；
3. `Vivian`、`Serena`、`Uncle_Fu` 三个 WAV 哈希不同；
4. TTS → 本服务 ASR 的中文回环字符重合率 `1.000`；
5. 三音色 WS 首块分别约 13.45s、16.33s、27.12s，wall/audio RTF 约 9.78–10.56；
6. 空闲容器内存约为 Qwen 5.79GiB + 主服务 0.87GiB。

结论：功能与中文链路均通过，但本机 CPU 比实时慢约 10 倍，仅适合试听、离线生成和
功能开发；实时对话必须评估 CUDA 或真正的流式后端。

实际启动与命令见 [启动与使用说明.md](./启动与使用说明.md)。
