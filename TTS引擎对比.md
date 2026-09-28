# TTS 引擎对比与选型决策：Melo / AISHELL3 / Matcha / Kokoro

> 更新日期：2026-09-28
> 环境：Apple Silicon (arm64) macOS / Docker VM 8C8G / CPU int8 推理
> 全部速度数据为本机 sherpa-onnx 1.13.8 实测，非估算

## 1. 当前决策（待试听定稿）

| 决策项 | 结论 |
|---|---|
| **临时默认引擎** | **Melo**（中文母语单音色、44.1kHz、MIT；试听后再定最终生产模型） |
| 严格纯中文候选 | **AISHELL3 VITS**（174 音色、Apache-2.0；原生仅 8kHz） |
| Matcha 定位 | 中文发音与延迟对比基线；**不得随商用产品交付**（训练数据非商用许可） |
| Kokoro 定位 | 保留兼容和历史基线，不再作为当前中文默认候选 |
| 切换方式 | `VOICE_TTS_ENGINE=melo\|aishell3\|matcha\|kokoro` 后重建容器；API/协议不变 |

一键试听：`./scripts/compare_tts_models.sh`。脚本生成同文本 WAV 到
`~/Documents/tmp/tts-compare/<时间戳>/`，并在结束后恢复 Melo。

## 2. 许可链分析（逐条查证）

### 2.1 matcha-icefall-zh-baker ⚠️ 商用受限

| 层 | 许可 | 说明 |
|---|---|---|
| 代码（Matcha-TTS / icefall） | MIT / Apache-2.0 | 无问题 |
| 导出权重（k2-fsa 社区） | Apache-2.0 | 权重声明本身宽松 |
| **训练数据：标贝 CSMSC** | **仅限非商业用途** | 官方条款：「此数据集是免费非商业用途；如有商业需求，请与我们联系获得商业使用授权」。10000 句 / ~12h / 单女声 |
| vocos-22khz-univ 声码器 | MIT（代码） | 权重无明确商用限制，风险较低 |

关键：**模型权重是训练数据的衍生物**，CSMSC 的非商用限制传导至权重。
本项目为 RV1106 板子产品的商用语音链路（替换百度 TTS）→ matcha 不可随产品交付，
除非向标贝科技购买 CSMSC 商业授权。本机开发/内部测试使用属非商业范围，合规。

### 2.2 kokoro-int8-multi-lang-v1_1 ✅ 商用干净

模型卡（hexgrad/Kokoro-82M-v1.1-zh）查证：

- 权重 **Apache-2.0**；
- 中文 100 音色数据：「The Chinese data was **freely and permissively granted** to us by
  LongMaoData（龙猫数据）」——数据方主动宽容授权；
- 英文补充数据：3 小时众包**合成**语音（作者援引美国版权局指引：合成数据一般不受版权保护）；
- 官方 FAQ：「Kokoro has been deployed in numerous projects and **commercial APIs**.
  We **welcome** deployment of the model in actual use.」无需申请。

## 3. 本轮中文候选实测

同一段中文、CPU、脚本自动重启后的首轮结果（输出分别重采样为 Melo/Matcha 24kHz、AISHELL3 16kHz）：

| 引擎/音色 | 原生采样率 | WS 首块 | wall/audio |
|---|---:|---:|---:|
| Melo sid 0 | 44.1kHz | 878ms（冒烟长句热态 1972ms） | 1.458 |
| AISHELL3 sid 0/10/33/99 | 8kHz | 139–182ms | 0.096–0.109 |
| Matcha sid 0 | 22.05kHz | 200ms | 0.169 |

所有样本均成功生成有效 PCM WAV；默认 Melo 端到端冒烟 13/13、TTS→ASR 中文回环重合率 1.000。
语音自然度与音色偏好仍以 xpl 实听为准。

## 3.1 历史对比（Kokoro / Matcha）

| 维度 | kokoro-int8-multi-lang-v1_1 | matcha-icefall-zh-baker + vocos |
|---|---|---|
| 架构 | StyleTTS2 系非自回归 + 内置 iSTFTNet 声码器，82M 参数 | Conditional Flow-Matching 声学模型（2 步 ODE）+ 独立 vocos 声码器 |
| 体积 | 215MB（int8，含 jieba/espeak-ng-data/双 lexicon/FST） | 146MB（声学 ~30MB + vocos 54MB + 词典/FST） |
| 音色 | **103 个**（中文 zf/zm 100 + 英文 3），男女声/风格可切 | **1 个**（标贝女声，新闻播报风） |
| 语言 | 中文 + 英文 + 中英混读（lexicon-gb/us-en） | 中文为主，英文词经 espeak-ng 兜底（口音较硬） |
| 音质 | 更自然、有韵律变化，接近商用 TTS | 清晰标准但音色单一、韵律偏平 |
| CPU RTF（本机实测） | ≈0.8 | **≈0.05（快 ~16×）** |
| CPU 每次 generate 固定开销 | **~700ms**（线程 2→6 仅 1598→1351ms，扩展性差） | 可忽略 |
| 回调粒度 | 粗（~2s 音频/次；短句整句才回调一次） | 整句级，但合成极快故首块即出 |
| **WS 首块延迟（本机实测）** | **~1.4s（冷机）~2.4s（持续负载热机降频）**（❌ CPU 上超 800ms 目标，波动 ±30%） | **~260ms**（✅ 3 倍余量，基本不受热状态影响） |
| 采样率 | 24kHz | 22.05kHz |
| 文本前端 | jieba + date/number/phone-zh FST | jieba + date/number/phone FST（等价） |
| GPU 适配 | **fp16 + CUDA EP 为 gpu-profile 正主**（注意 int8 权重与 CUDA EP 不兼容，须换 fp32/fp16 权重 427MB 版） | 可上 GPU 但无必要 |
| 许可 | **Apache-2.0，商用欢迎** | **数据非商用**，商用需标贝授权 |

结论一句话：**matcha 赢在 CPU 延迟 20×，kokoro 赢在音色数、质量与商用许可**。
kokoro 在 CPU 上慢的根因是推理管线固定开销 + 回调粒度粗，与线程数基本无关；
换 x86 CPU 改善有限，**GPU 才是 kokoro 的正确部署形态**。

## 4. 延迟预算对照（半双工，感知延迟目标 ≤2.5s）

| 环节 | kokoro@CPU（本机） | kokoro@GPU（预期，待实测） | matcha@CPU |
|---|---|---|---|
| ASR 尾包（speech_end→final） | ~0.3s | ~0.3s | ~0.3s |
| LLM 首 token（应用服务器侧） | ~0.8s | ~0.8s | ~0.8s |
| TTS 首块 | ~1.4s | **<0.3s** | ~0.26s |
| 合计 | ≈2.5s（顶格无余量） | **≈1.4s** | ≈1.4s |

## 5. 中文候选说明

1. **Melo**：中文母语单音色，支持中英混读，44.1kHz，MIT；当前临时默认。
2. **AISHELL3**：严格纯中文、174 音色、Apache-2.0；8kHz 原生带宽是主要限制。
3. **Matcha**：中文母语单女声且延迟低，但 Baker/CSMSC 数据仅限非商业使用。
4. **Kokoro**：许可宽松、音色多，但当前实听存在明显非母语中文口音。

## 5.1 本机 Apple GPU（CoreML EP）实测：无收益

sherpa-onnx `provider` 支持 `coreml`（Apple GPU/神经引擎路径，Mac 上无 CUDA/Metal EP）。
本机实测（2026-09-25，同机热态对比）：

| 项 | CPU EP | CoreML EP |
|---|---|---|
| kokoro TTS「今天天气不错，」合成 | 1885ms | 1814ms（~4%，噪声级） |
| ASR zipformer int8 加载 | ~3s | 10.2s（子图编译反而更慢） |

原因：**int8 动态量化算子 + 动态 shape 在 CoreML/ANE 覆盖率极低**，子图基本全部回退
CPU。结论：本机维持 CPU；若未来重试 Apple GPU，前提换 fp32/fp16 权重，预期收益仍有限。
`gpu-fp16` profile 仅面向生产 NVIDIA 机器（CUDA EP）。

## 6. 后续行动项

- [ ] GPU 机器到位后：接 `gpu-fp16` profile（CUDA EP + kokoro fp32/fp16 权重），实测首块延迟；
      冒烟测试 kokoro 阈值届时从 CPU 哨兵值 3000ms 收紧回 800ms（`cli/smoke_test.py` 注释已标）
- [x] 接入 Melo 与 AISHELL3 VITS，引擎默认 speaker 按模型配置
- [x] 一键生成 Melo / AISHELL3 / Matcha 同文本试听样本
- [ ] xpl 试听后确定最终生产默认模型与 speaker
