"""配置加载：config.yaml + 环境变量覆盖（VOICE_CONFIG / VOICE_MODELS_DIR / VOICE_API_KEY / VOICE_PORT）。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATHS = [
    Path(os.environ.get("VOICE_CONFIG", "")),
    Path("/opt/voice/config.yaml"),
    Path(__file__).resolve().parent.parent / "config.yaml",
]


@dataclass
class AsrCfg:
    encoder: str = ""
    decoder: str = ""
    joiner: str = ""
    tokens: str = ""
    num_threads: int = 2
    language: str = "zh"
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 1.2
    rule3_min_utterance_length: float = 20.0


@dataclass
class DenoiserCfg:
    gtcrn_model: str = ""
    num_threads: int = 1


@dataclass
class VadCfg:
    model: str = ""
    threshold: float = 0.5
    min_silence: float = 0.5
    min_speech: float = 0.25
    max_speech: float = 20.0
    window_size: int = 512
    sample_rate: int = 16000
    num_threads: int = 1
    buffer_seconds: int = 60


@dataclass
class TtsCfg:
    engine: str = "kokoro"  # kokoro | matcha
    # kokoro（多音色，质量优先）
    model: str = ""
    lexicon: str = ""
    tokens: str = ""
    data_dir: str = ""
    dict_dir: str = ""
    voices: str = ""
    rule_fsts: str = ""
    # matcha-icefall-zh-baker（单女声，低延迟备选；baker 数据集仅限非商用）
    matcha_model: str = "matcha-icefall-zh-baker/model-steps-3.onnx"
    matcha_lexicon: str = "matcha-icefall-zh-baker/lexicon.txt"
    matcha_tokens: str = "matcha-icefall-zh-baker/tokens.txt"
    matcha_dict_dir: str = "matcha-icefall-zh-baker/dict"
    matcha_data_dir: str = ""  # espeak-ng-data（英文词发音）；留空则复用 kokoro 的
    matcha_vocoder: str = "vocos-22khz-univ.onnx"
    matcha_rule_fsts: str = ("matcha-icefall-zh-baker/date.fst,"
                             "matcha-icefall-zh-baker/number.fst,"
                             "matcha-icefall-zh-baker/phone.fst")
    matcha_noise_scale: float = 0.667
    matcha_length_scale: float = 1.0
    # 通用
    num_threads: int = 2
    default_speaker: int = 0
    default_speed: float = 1.0
    sample_rate_choices: list = field(default_factory=lambda: [16000, 24000])


@dataclass
class LimitsCfg:
    asr_sessions: int = 32
    tts_sessions: int = 4
    idle_timeout: float = 60.0
    max_session: float = 300.0
    max_upload_mb: int = 25


@dataclass
class DefaultsCfg:
    denoise: bool = False


@dataclass
class Settings:
    models_dir: Path = Path("/opt/voice/models")
    api_key: str = "devkey-local"
    asr: AsrCfg = field(default_factory=AsrCfg)
    denoiser: DenoiserCfg = field(default_factory=DenoiserCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    tts: TtsCfg = field(default_factory=TtsCfg)
    limits: LimitsCfg = field(default_factory=LimitsCfg)
    defaults: DefaultsCfg = field(default_factory=DefaultsCfg)

    def model_path(self, rel: str) -> Path:
        return self.models_dir / rel


def _fill(cls, data: dict[str, Any] | None):
    obj = cls()
    for k, v in (data or {}).items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj


def load_settings() -> Settings:
    cfg_path = next((p for p in DEFAULT_CONFIG_PATHS if p.is_file()), None)
    raw: dict[str, Any] = {}
    if cfg_path is not None:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    s = Settings()
    s.models_dir = Path(os.environ.get("VOICE_MODELS_DIR", raw.get("models_dir", "/opt/voice/models")))
    s.api_key = os.environ.get("VOICE_API_KEY", "devkey-local")
    s.asr = _fill(AsrCfg, raw.get("asr"))
    s.denoiser = _fill(DenoiserCfg, raw.get("denoiser"))
    s.vad = _fill(VadCfg, raw.get("vad"))
    s.tts = _fill(TtsCfg, raw.get("tts"))
    s.limits = _fill(LimitsCfg, raw.get("limits"))
    s.defaults = _fill(DefaultsCfg, raw.get("defaults"))
    if port := os.environ.get("VOICE_PORT"):
        raw.setdefault("server", {})["port"] = int(port)
    s.server = raw.get("server", {"host": "0.0.0.0", "port": 8090})
    return s
