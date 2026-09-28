"""模型加载与共享。

进程内单例 ModelHub：
- recognizer / tts 全局共享权重（ASR 每会话 create_stream()）
- 降噪器与 VAD 有状态，提供工厂方法按会话创建（模型极小，创建廉价）
任一模型加载失败 → 抛异常 → 启动失败（fail fast），health 报 not ready。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import sherpa_onnx

from .config import Settings

log = logging.getLogger(__name__)


class ModelHub:
    def __init__(self, settings: Settings):
        self.s = settings
        self.recognizer = None
        self.tts = None
        self.refine_recognizer = None   # SenseVoice 离线精修（可缺失 → None）
        self.punctuation = None         # ct-transformer 标点（可缺失 → None）
        self.tts_engine: str | None = None
        self.info: dict = {}
        self.ready = False
        self.load_errors: list[str] = []

    @property
    def refine_available(self) -> bool:
        return self.refine_recognizer is not None

    # ---- 工厂（每会话） ----
    def create_denoiser(self):
        cfg = sherpa_onnx.OnlineSpeechDenoiserConfig()
        cfg.model.gtcrn.model = str(self.s.model_path(self.s.denoiser.gtcrn_model))
        cfg.model.num_threads = self.s.denoiser.num_threads
        cfg.model.provider = self.s.provider
        if not cfg.validate():
            raise RuntimeError(f"invalid denoiser config: {cfg.model.gtcrn.model}")
        return sherpa_onnx.OnlineSpeechDenoiser(config=cfg)

    def create_vad(self):
        v = self.s.vad
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(self.s.model_path(v.model))
        cfg.silero_vad.threshold = v.threshold
        cfg.silero_vad.min_silence_duration = v.min_silence
        cfg.silero_vad.min_speech_duration = v.min_speech
        cfg.silero_vad.max_speech_duration = v.max_speech
        cfg.silero_vad.window_size = v.window_size
        cfg.sample_rate = v.sample_rate
        cfg.num_threads = v.num_threads
        cfg.provider = self.s.provider
        return sherpa_onnx.VoiceActivityDetector(config=cfg, buffer_size_in_seconds=v.buffer_seconds)

    # ---- 加载（启动时一次） ----
    def load(self) -> None:
        t0 = time.monotonic()
        a = self.s.asr
        for rel in (a.encoder, a.decoder, a.joiner, a.tokens):
            p = self.s.model_path(rel)
            if not p.is_file():
                raise FileNotFoundError(f"ASR model file missing: {p}")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(self.s.model_path(a.tokens)),
            encoder=str(self.s.model_path(a.encoder)),
            decoder=str(self.s.model_path(a.decoder)),
            joiner=str(self.s.model_path(a.joiner)),
            num_threads=a.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=a.rule1_min_trailing_silence,
            rule2_min_trailing_silence=a.rule2_min_trailing_silence,
            rule3_min_utterance_length=a.rule3_min_utterance_length,
            provider=self.s.provider,
        )
        log.info("ASR loaded in %.1fs", time.monotonic() - t0)

        t1 = time.monotonic()
        tt = self.s.tts
        engine = (tt.engine or "melo").lower()
        self.tts_engine = engine
        if engine == "matcha":
            missing = [rel for rel in (tt.matcha_model, tt.matcha_lexicon,
                                        tt.matcha_tokens, tt.matcha_vocoder)
                       if not self.s.model_path(rel).is_file()]
            if missing:
                raise FileNotFoundError(f"TTS(matcha) model file missing: {missing}")
            data_dir = tt.matcha_data_dir or tt.data_dir  # 复用 kokoro 的 espeak-ng-data
            matcha = sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(self.s.model_path(tt.matcha_model)),
                lexicon=str(self.s.model_path(tt.matcha_lexicon)),
                tokens=str(self.s.model_path(tt.matcha_tokens)),
                data_dir=str(self.s.model_path(data_dir)) if data_dir else "",
                dict_dir=str(self.s.model_path(tt.matcha_dict_dir)) if tt.matcha_dict_dir else "",
                vocoder=str(self.s.model_path(tt.matcha_vocoder)),
                noise_scale=tt.matcha_noise_scale,
                length_scale=tt.matcha_length_scale,
            )
            model_cfg = sherpa_onnx.OfflineTtsModelConfig(
                matcha=matcha, num_threads=tt.num_threads, provider=self.s.provider, debug=False)
            rule_fsts = ",".join(str(self.s.model_path(x)) for x in tt.matcha_rule_fsts.split(",") if x)
            selected_model = tt.matcha_model
        elif engine in ("melo", "aishell3"):
            model = getattr(tt, f"{engine}_model")
            lexicon = getattr(tt, f"{engine}_lexicon")
            tokens = getattr(tt, f"{engine}_tokens")
            rule_fst_cfg = getattr(tt, f"{engine}_rule_fsts")
            missing = [rel for rel in (model, lexicon, tokens)
                       if not self.s.model_path(rel).is_file()]
            if missing:
                raise FileNotFoundError(f"TTS({engine}) model file missing: {missing}")
            vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(self.s.model_path(model)),
                lexicon=str(self.s.model_path(lexicon)),
                tokens=str(self.s.model_path(tokens)),
                noise_scale=tt.vits_noise_scale,
                noise_scale_w=tt.vits_noise_scale_w,
                length_scale=tt.vits_length_scale,
            )
            model_cfg = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits, num_threads=tt.num_threads, provider=self.s.provider, debug=False)
            rule_fsts = ",".join(str(self.s.model_path(x)) for x in rule_fst_cfg.split(",") if x)
            selected_model = model
        elif engine == "kokoro":
            required = [tt.model, tt.tokens, tt.voices]
            required.extend(x for x in tt.lexicon.split(",") if x)
            missing = [rel for rel in required if not self.s.model_path(rel).is_file()]
            if missing:
                raise FileNotFoundError(f"TTS(kokoro) model file missing: {missing}")
            kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(self.s.model_path(tt.model)),
                lexicon=",".join(str(self.s.model_path(x)) for x in tt.lexicon.split(",") if x),
                tokens=str(self.s.model_path(tt.tokens)),
                data_dir=str(self.s.model_path(tt.data_dir)) if tt.data_dir else "",
                dict_dir=str(self.s.model_path(tt.dict_dir)) if tt.dict_dir else "",
                voices=str(self.s.model_path(tt.voices)) if tt.voices else "",
            )
            model_cfg = sherpa_onnx.OfflineTtsModelConfig(
                kokoro=kokoro, num_threads=tt.num_threads, provider=self.s.provider, debug=False)
            rule_fsts = ",".join(str(self.s.model_path(x)) for x in tt.rule_fsts.split(",") if x)
            selected_model = tt.model
        else:
            raise ValueError(f"unsupported tts.engine {tt.engine!r}; expected melo, aishell3, matcha, or kokoro")
        tts_cfg = sherpa_onnx.OfflineTtsConfig(
            model=model_cfg, rule_fsts=rule_fsts, max_num_sentences=1)
        if not tts_cfg.validate():
            raise RuntimeError("invalid TTS config")
        self.tts = sherpa_onnx.OfflineTts(config=tts_cfg)
        log.info("TTS(%s) loaded in %.1fs, sample_rate=%d, speakers=%d",
                 engine, time.monotonic() - t1, self.tts.sample_rate, self.tts.num_speakers)

        # 降噪/VAD 模型存在性校验（实例按会话创建）
        for rel in (self.s.denoiser.gtcrn_model, self.s.vad.model):
            if not self.s.model_path(rel).is_file():
                raise FileNotFoundError(f"model file missing: {self.s.model_path(rel)}")

        self._load_refine()

        r = self.s.refine
        self.info = {
            "provider": self.s.provider,
            "sherpa_onnx": sherpa_onnx.__version__,
            "asr": {"type": "streaming-zipformer-transducer", "model": Path(a.encoder).parent.name, "language": a.language},
            "tts": {"type": engine, "model": Path(selected_model).parent.name,
                    "sample_rate": self.tts.sample_rate, "num_speakers": self.tts.num_speakers,
                    "default_speaker": self.default_speaker},
            "vad": {"type": "silero", "model": self.s.vad.model},
            "denoiser": {"type": "gtcrn", "model": self.s.denoiser.gtcrn_model},
            "refine": ({"type": "sense-voice", "model": Path(r.sense_voice_model).parent.name,
                        "itn": r.use_itn} if self.refine_recognizer else None),
            "punctuation": ({"type": "ct-transformer", "model": Path(r.punct_model).parent.name}
                            if self.punctuation else None),
        }
        self.ready = True
        log.info("all models ready in %.1fs", time.monotonic() - t0)

    def _load_refine(self) -> None:
        """v2 精修通道：SenseVoice + 标点。模型缺失/加载失败仅告警降级，不阻断启动。"""
        r = self.s.refine
        if not r.enabled:
            log.info("refine disabled by config")
            return
        sv_model = self.s.model_path(r.sense_voice_model)
        sv_tokens = self.s.model_path(r.sense_voice_tokens)
        if not (sv_model.is_file() and sv_tokens.is_file()):
            msg = f"refine models missing ({sv_model.name}), refine channel disabled"
            log.warning(msg)
            self.load_errors.append(msg)
            return
        try:
            t0 = time.monotonic()
            self.refine_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                tokens=str(sv_tokens),
                model=str(sv_model),
                num_threads=r.num_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
                provider=self.s.provider,
                language=r.language,
                use_itn=r.use_itn,
            )
            log.info("refine SenseVoice loaded in %.1fs", time.monotonic() - t0)
        except Exception as e:
            self.refine_recognizer = None
            self.load_errors.append(f"refine SenseVoice load failed: {e}")
            log.exception("refine SenseVoice load failed, channel disabled")
            return
        punct_model = self.s.model_path(r.punct_model)
        if punct_model.is_file():
            try:
                t1 = time.monotonic()
                cfg = sherpa_onnx.OfflinePunctuationConfig()
                cfg.model.ct_transformer = str(punct_model)
                cfg.model.num_threads = r.num_threads
                cfg.model.provider = self.s.provider
                self.punctuation = sherpa_onnx.OfflinePunctuation(config=cfg)
                log.info("punctuation loaded in %.1fs", time.monotonic() - t1)
            except Exception as e:
                self.punctuation = None
                self.load_errors.append(f"punctuation load failed: {e}")
                log.exception("punctuation load failed (refine still works without it)")
        else:
            self.load_errors.append(f"punct model missing ({punct_model.name})")

    # ---- speaker 解析 ----
    @property
    def default_speaker(self) -> int:
        engine = self.tts_engine or (self.s.tts.engine or "melo").lower()
        defaults = self.s.tts.default_speakers or {}
        if engine in defaults:
            return int(defaults[engine])
        if self.s.tts.default_speaker is not None:
            return int(self.s.tts.default_speaker)
        return 0

    def resolve_speaker(self, speaker) -> int:
        """仅支持整数 sid；省略时使用当前引擎的默认中文音色。"""
        if speaker is None:
            sid = self.default_speaker
        elif isinstance(speaker, bool):
            raise ValueError("bad speaker")
        elif isinstance(speaker, int):
            sid = speaker
        elif isinstance(speaker, str) and speaker.isdigit():
            sid = int(speaker)
        else:
            raise ValueError(f"speaker must be an integer sid in v1 (0..{self.tts.num_speakers - 1}), got {speaker!r}")
        if not (0 <= sid < self.tts.num_speakers):
            raise ValueError(f"speaker sid {sid} out of range 0..{self.tts.num_speakers - 1}")
        return sid
