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
        self.info: dict = {}
        self.ready = False
        self.load_errors: list[str] = []

    # ---- 工厂（每会话） ----
    def create_denoiser(self):
        cfg = sherpa_onnx.OnlineSpeechDenoiserConfig()
        cfg.model.gtcrn.model = str(self.s.model_path(self.s.denoiser.gtcrn_model))
        cfg.model.num_threads = self.s.denoiser.num_threads
        cfg.model.provider = "cpu"
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
        cfg.provider = "cpu"
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
            provider="cpu",
        )
        log.info("ASR loaded in %.1fs", time.monotonic() - t0)

        t1 = time.monotonic()
        tt = self.s.tts
        engine = (tt.engine or "kokoro").lower()
        if engine == "matcha":
            missing = [rel for rel in (tt.matcha_model, tt.matcha_tokens, tt.matcha_vocoder)
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
                matcha=matcha, num_threads=tt.num_threads, provider="cpu", debug=False)
            rule_fsts = ",".join(str(self.s.model_path(x)) for x in tt.matcha_rule_fsts.split(",") if x)
        else:
            missing = [rel for rel in (tt.model, tt.tokens) if not self.s.model_path(rel).is_file()]
            if missing:
                raise FileNotFoundError(f"TTS model file missing: {missing}")
            kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(self.s.model_path(tt.model)),
                lexicon=",".join(str(self.s.model_path(x)) for x in tt.lexicon.split(",") if x),
                tokens=str(self.s.model_path(tt.tokens)),
                data_dir=str(self.s.model_path(tt.data_dir)) if tt.data_dir else "",
                dict_dir=str(self.s.model_path(tt.dict_dir)) if tt.dict_dir else "",
                voices=str(self.s.model_path(tt.voices)) if tt.voices else "",
            )
            model_cfg = sherpa_onnx.OfflineTtsModelConfig(
                kokoro=kokoro, num_threads=tt.num_threads, provider="cpu", debug=False)
            rule_fsts = ",".join(str(self.s.model_path(x)) for x in tt.rule_fsts.split(",") if x)
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

        self.info = {
            "provider": "cpu",
            "sherpa_onnx": sherpa_onnx.__version__,
            "asr": {"type": "streaming-zipformer-transducer", "model": Path(a.encoder).parent.name, "language": a.language},
            "tts": {"type": engine, "model": Path(tt.model if engine == "kokoro" else tt.matcha_model).parent.name,
                    "sample_rate": self.tts.sample_rate, "num_speakers": self.tts.num_speakers},
            "vad": {"type": "silero", "model": self.s.vad.model},
            "denoiser": {"type": "gtcrn", "model": self.s.denoiser.gtcrn_model},
            "refine": None, "punctuation": None, "itn": None,  # v1 精简核心，后续阶段接入
        }
        self.ready = True
        log.info("all models ready in %.1fs", time.monotonic() - t0)

    # ---- speaker 解析 ----
    def resolve_speaker(self, speaker, ) -> int:
        """v1：仅支持整数 sid（kokoro voices.bin 顺序）。名称映射待官方表接入。"""
        if speaker is None:
            return int(self.s.tts.default_speaker)
        if isinstance(speaker, bool):
            raise ValueError("bad speaker")
        if isinstance(speaker, int):
            sid = speaker
        elif isinstance(speaker, str) and speaker.isdigit():
            sid = int(speaker)
        else:
            raise ValueError(f"speaker must be an integer sid in v1 (0..{self.tts.num_speakers - 1}), got {speaker!r}")
        if not (0 <= sid < self.tts.num_speakers):
            raise ValueError(f"speaker sid {sid} out of range 0..{self.tts.num_speakers - 1}")
        return sid
