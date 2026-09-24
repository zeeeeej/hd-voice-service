"""每 WS 会话一份状态的流式 ASR 管线：
GTCRN 降噪(可选) → Silero VAD + streaming zipformer 连续解码
- VAD speech_end（或 recognizer 端点/20s 兜底）触发 final
- 纯静音段不出 final；eof 时 flush 降噪与 VAD
所有方法均为同步阻塞，调用方需放到线程（asyncio.to_thread）执行。
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)

SR = 16000


class AsrSession:
    def __init__(self, hub, denoise: bool = False):
        self.hub = hub
        self.denoise_enabled = denoise
        self._denoiser = hub.create_denoiser() if denoise else None
        self._vad = hub.create_vad()
        self._rec = hub.recognizer
        self._stream = self._rec.create_stream()
        self._den_buf = np.empty(0, dtype=np.float32)
        self._vad_buf = np.empty(0, dtype=np.float32)
        self.total_fed = 0            # 已喂入(VAD/ASR 时间轴)采样数
        self.seg_index = 0
        self.in_speech = False
        self.seg_start_t = 0.0
        self.seg_asr_ms = 0.0
        self.seg_denoise_ms = 0.0
        self.last_partial = ""
        self._speech_start_wall = None
        self._first_partial_wall = None
        self._forced_final = False    # recognizer 端点已强切，等待 VAD 段弹出时跳过重复 final
        self.finished = False

    # ---------- 内部 ----------
    def _run_denoise(self, samples: np.ndarray) -> tuple[np.ndarray, float]:
        if self._denoiser is None:
            return samples, 0.0
        t0 = time.perf_counter()
        self._den_buf = np.concatenate([self._den_buf, samples])
        fs = int(self._denoiser.frame_shift_in_samples)
        out = []
        while len(self._den_buf) >= fs:
            # run(samples, sample_rate) → DenoisedAudio；输出可能短于输入（内部缓存 1 帧延迟）
            denoised = self._denoiser.run(self._den_buf[:fs], SR)
            out.append(np.asarray(denoised.samples, dtype=np.float32))
            self._den_buf = self._den_buf[fs:]
        audio = np.concatenate(out) if out else np.empty(0, dtype=np.float32)
        return audio, (time.perf_counter() - t0) * 1000.0

    def _flush_denoise(self) -> tuple[np.ndarray, float]:
        if self._denoiser is None or len(self._den_buf) == 0:
            self._den_buf = np.empty(0, dtype=np.float32)
            return np.empty(0, dtype=np.float32), 0.0
        t0 = time.perf_counter()
        tail = np.asarray(self._denoiser.flush().samples, dtype=np.float32)
        self._den_buf = np.empty(0, dtype=np.float32)
        return tail, (time.perf_counter() - t0) * 1000.0

    def _feed_vad(self, audio: np.ndarray):
        self._vad_buf = np.concatenate([self._vad_buf, audio])
        w = int(self._vad.config.silero_vad.window_size)
        while len(self._vad_buf) >= w:
            self._vad.accept_waveform(self._vad_buf[:w])
            self._vad_buf = self._vad_buf[w:]

    def _now_t(self) -> float:
        return round(self.total_fed / SR, 2)

    def _current_text(self) -> str:
        # sherpa-onnx 1.13: 结果通过 recognizer.get_result(stream) 获取
        return (self._rec.get_result(self._stream) or "").strip()

    def _emit_final(self, events: list, text: str, start_t: float, end_t: float) -> None:
        dur = max(end_t - start_t, 1e-6)
        events.append({
            "type": "final",
            "text": text,
            "refined": False,          # v1 精简核心：无精修通道
            "segment": self.seg_index,
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "language": self.hub.info.get("asr", {}).get("language", "zh"),
            "confidence": None,
            "timings": {
                "denoise_ms": round(self.seg_denoise_ms, 1),
                "asr_ms": round(self.seg_asr_ms, 1),
                "refine_ms": None,
                "punc_ms": None,
            },
            "rtf": round((self.seg_asr_ms / 1000.0) / dur, 4),
        })
        self.seg_index += 1
        self.seg_asr_ms = 0.0
        self.seg_denoise_ms = 0.0
        self.last_partial = ""
        self._rec.reset(self._stream)

    def _process_audio(self, audio: np.ndarray, events: list) -> None:
        """喂 VAD + recognizer，产出 vad/partial/final 事件（不含降噪本身）。"""
        if len(audio) == 0:
            return
        self._stream.accept_waveform(SR, audio)
        t0 = time.perf_counter()
        # sherpa-onnx 1.13: 每次 decode_stream 只推进一步，须循环到 is_ready 为假
        steps = 0
        while self._rec.is_ready(self._stream) and steps < 100000:
            self._rec.decode_stream(self._stream)
            steps += 1
        self.seg_asr_ms += (time.perf_counter() - t0) * 1000.0
        self._feed_vad(audio)
        self.total_fed += len(audio)

        text = self._current_text()

        # VAD 已完成段弹出 → speech_end + final
        while not self._vad.empty():
            seg = self._vad.front
            self._vad.pop()
            start_t = seg.start / SR
            end_t = (seg.start + len(seg.samples)) / SR
            self.in_speech = False
            events.append({"type": "vad", "state": "speech_end", "t": round(end_t, 2), "segment": self.seg_index})
            if self._forced_final:
                self._forced_final = False
                self.last_partial = ""
                self._rec.reset(self._stream)
                self.seg_asr_ms = 0.0
                self.seg_denoise_ms = 0.0
            elif text:
                self._emit_final(events, text, start_t, end_t)
            else:
                # 空文本段（噪声/静音误触发）：不出 final，仅回收状态
                self.last_partial = ""
                self._rec.reset(self._stream)
                self.seg_asr_ms = 0.0
                self.seg_denoise_ms = 0.0
            text = self._current_text()

        # speech_start
        if self._vad.is_speech_detected() and not self.in_speech:
            self.in_speech = True
            self.seg_start_t = self._now_t()
            self._speech_start_wall = time.monotonic()
            self._first_partial_wall = None
            self._forced_final = False
            events.append({"type": "vad", "state": "speech_start", "t": self._now_t(), "segment": self.seg_index})

        # partial
        if text and text != self.last_partial:
            self.last_partial = text
            if self._first_partial_wall is None and self._speech_start_wall is not None:
                self._first_partial_wall = time.monotonic()
            events.append({"type": "partial", "text": text, "segment": self.seg_index, "t": self._now_t()})

        # recognizer 端点兜底（长句尾静音 / 超 20s 强切）
        if self._rec.is_endpoint(self._stream):
            if text and not self._forced_final:
                self._emit_final(events, text, self.seg_start_t, self._now_t())
                self._forced_final = self.in_speech  # VAD 段稍后弹出时跳过
                if self.in_speech:
                    events.append({"type": "vad", "state": "speech_end", "t": self._now_t(), "segment": self.seg_index - 1})
                    self.in_speech = False
            else:
                self._rec.reset(self._stream)
                self.last_partial = ""

    # ---------- 对外 ----------
    def feed(self, samples: np.ndarray) -> list[dict]:
        """samples: 16k 单声道 float32。返回事件列表（顺序即发送顺序）。"""
        if self.finished:
            return []
        events: list[dict] = []
        audio, den_ms = self._run_denoise(samples)
        self.seg_denoise_ms += den_ms
        self._process_audio(audio, events)
        return events

    def finish(self) -> list[dict]:
        """eof：flush 降噪 + VAD，收尾 final。"""
        if self.finished:
            return []
        self.finished = True
        events: list[dict] = []
        tail, den_ms = self._flush_denoise()
        self.seg_denoise_ms += den_ms
        if len(tail):
            self._process_audio(tail, events)
        # 尾部补 1.28s 静音（zipformer 右侧上下文 ~0.64s×2），吐完尾部 token
        self._process_audio(np.zeros(int(1.28 * SR), dtype=np.float32), events)
        self._vad.flush()
        while not self._vad.empty():
            seg = self._vad.front
            self._vad.pop()
            start_t = seg.start / SR
            end_t = (seg.start + len(seg.samples)) / SR
            self.in_speech = False
            events.append({"type": "vad", "state": "speech_end", "t": round(end_t, 2), "segment": self.seg_index})
            if not self._forced_final:
                self._emit_final(events, self._current_text(), start_t, end_t)
            self._forced_final = False
        # 最终兜底：VAD 从未触发（如超短语音）但识别器已有文本
        text = self._current_text()
        if text:
            self._emit_final(events, text, self.seg_start_t, self._now_t())
        return events

    @property
    def first_partial_latency_ms(self) -> float | None:
        if self._speech_start_wall and self._first_partial_wall:
            return (self._first_partial_wall - self._speech_start_wall) * 1000.0
        return None
