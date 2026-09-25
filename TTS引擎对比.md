# TTS 引擎对比与选型决策：Kokoro vs Matcha

> 记录日期：2026-09-25
> 环境：Apple Silicon (arm64) macOS / Docker VM 8C8G / CPU int8 推理
> 全部速度数据为本机 sherpa-onnx 1.13.8 实测，非估算

## 1. 决策（已定）

| 决策项 | 结论 |
|---|---|
| **默认/生产引擎** | **Kokoro**（`kokoro-int8-multi-lang-v1_1`，商用许可干净、103 音色、质量最好） |
| 生产硬件路线 | **GPU**（正式应用走 CUDA EP + fp16 权重；CPU 上 kokoro 的固定开销在 GPU 上基本消失，首块预期 <300ms，到位后实测确认） |
| Matcha 定位 | **可选项**：测试/调协议时对比使用（CPU 低延迟 20×），本机开发迭代可临时切换；**不得随商用产品交付**（训练数据非商用许可） |
| 切换方式 | `server/config.yaml` → `tts.engine: kokoro \| matcha`，重启容器即生效，接口/协议完全一致 |

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

## 3. 全面对比（含本机实测）

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

## 5. 若生产被迫纯 CPU 且商用的备选（当前不启用）

1. **`vits-melo-tts-zh_en`（MIT 许可）**：MyShell MeloTTS 的 sherpa-onnx 官方发布版，
   中英混读、单女声、VITS 架构 CPU 上通常 RTF 0.1–0.3。引擎开关已配置化，
   接入约半天（下载 163MB + `models.py` 加 vits 分支）。
2. 向标贝科技购买 CSMSC 商业授权，继续用 matcha。
3. 硬扛 kokoro CPU 首块 1.4s（延迟预算顶格，不推荐）。

## 6. 后续行动项

- [ ] GPU 机器到位后：接 `gpu-fp16` profile（CUDA EP + kokoro fp32/fp16 权重），实测首块延迟；
      冒烟测试 kokoro 阈值届时从 CPU 哨兵值 3000ms 收紧回 800ms（`cli/smoke_test.py` 注释已标）
- [ ] （可选）接入 melo 第三引擎，覆盖纯 CPU 商用兜底场景
- [ ] 测试对比流程：同一文本分别以两引擎合成（改 `tts.engine` 重启即可），
      用 `cli/tts_cli.py --play` 盲听 + `cli/smoke_test.py` 看延迟/回环指标
