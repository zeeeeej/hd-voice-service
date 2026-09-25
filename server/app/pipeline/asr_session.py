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
    def __init__(self, hub, denoise: bool = False, refine: bool = False,
                 punctuate: bool = True, seg_cap_seconds: float = 25.0):
        self.hub = hub
        self.denoise_enabled = denoise
        self.refine_enabled = refine          # 会话级开关（模型缺失时自动无效果）
        self.punctuate_enabled = punctuate
        self._seg_cap = int(seg_cap_seconds * SR)
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
        # v2 精修：当前语音段音频缓存 + 0.5s pre-roll 环形缓存（补 VAD 触发前的音头）
        self._capturing = False
        self._seg_pcm: list = []
        self._seg_pcm_len = 0
        self._preroll: list = []
        self._preroll_len = 0
        self._preroll_cap = int(0.5 * SR)

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

    def _now_t(self) -> float:
        return round(self.total_fed / SR, 2)

    def _current_text(self) -> str:
        # sherpa-onnx 1.13: 结果通过 recognizer.get_result(stream) 获取
        return (self._rec.get_result(self._stream) or "").strip()

    def _run_refine(self, pcm: np.ndarray) -> tuple[str | None, float | None, float | None]:
        """SenseVoice 精修 + 标点。返回 (refined_text, refine_ms, punc_ms)；失败返回 (None,…)。"""
        try:
            import time as _t
            t0 = _t.perf_counter()
            stream = self.hub.refine_recognizer.create_stream()
            stream.accept_waveform(SR, pcm)
            self.hub.refine_recognizer.decode_stream(stream)
            refined = (stream.result.text or "").strip()
            refine_ms = (_t.perf_counter() - t0) * 1000.0
            if not refined:
                return None, refine_ms, None
            # SenseVoice(use_itn) 输出已含标点/ITN，不再叠加 ct-transformer
            return refined, refine_ms, None
        except Exception:
            log.exception("refine failed, fallback to streaming text")
            return None, None, None

    def _take_seg_pcm(self) -> np.ndarray | None:
        if not self._seg_pcm:
            return None
        pcm = np.concatenate(self._seg_pcm)
        self._seg_pcm = []
        self._seg_pcm_len = 0
        self._capturing = False
        return pcm

    def _emit_final(self, events: list, text: str, start_t: float, end_t: float,
                    pcm: np.ndarray | None = None) -> None:
        dur = max(end_t - start_t, 1e-6)
        refined_flag = False
        refine_ms = punc_ms = None
        if (self.refine_enabled and pcm is not None and len(pcm) >= int(0.25 * SR)
                and getattr(self.hub, "refine_available", False)):
            r_text, refine_ms, punc_ms = self._run_refine(pcm)
            if r_text:
                text = r_text
                refined_flag = True
        if (not refined_flag and text and self.punctuate_enabled
                and self.hub.punctuation is not None):
            # 流式文本（refine 关闭或精修失败回退）→ ct-transformer 补标点
            import time as _t
            t1 = _t.perf_counter()
            try:
                text = self.hub.punctuation.add_punctuation(text)
                punc_ms = (_t.perf_counter() - t1) * 1000.0
            except Exception:
                log.exception("punctuate failed, keep raw streaming text")
        events.append({
            "type": "final",
            "text": text,
            "refined": refined_flag,
            "segment": self.seg_index,
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "language": self.hub.info.get("asr", {}).get("language", "zh"),
            "confidence": None,
            "timings": {
                "denoise_ms": round(self.seg_denoise_ms, 1),
                "asr_ms": round(self.seg_asr_ms, 1),
                "refine_ms": round(refine_ms, 1) if refine_ms is not None else None,
                "punc_ms": round(punc_ms, 1) if punc_ms is not None else None,
            },
            "rtf": round((self.seg_asr_ms / 1000.0) / dur, 4),
        })
        self.seg_index += 1
        self.seg_asr_ms = 0.0
        self.seg_denoise_ms = 0.0
        self.last_partial = ""
        self._rec.reset(self._stream)

    def _append_seg(self, audio: np.ndarray) -> None:
        self._seg_pcm.append(audio)
        self._seg_pcm_len += len(audio)
        if self._seg_pcm_len > self._seg_cap:      # 安全上限，丢弃最旧
            drop = self._seg_pcm_len - self._seg_cap
            while drop > 0 and self._seg_pcm:
                head = self._seg_pcm[0]
                if len(head) <= drop:
                    drop -= len(head)
                    self._seg_pcm.pop(0)
                else:
                    self._seg_pcm[0] = head[drop:]
                    drop = 0
            self._seg_pcm_len = sum(len(x) for x in self._seg_pcm)

    def _push_preroll(self, audio: np.ndarray) -> None:
        self._preroll.append(audio)
        self._preroll_len += len(audio)
        while self._preroll_len > self._preroll_cap and len(self._preroll) > 1:
            self._preroll_len -= len(self._preroll.pop(0))

    def _process_audio(self, audio: np.ndarray, events: list) -> None:
        """喂 VAD + recognizer，产出 vad/partial/final 事件（不含降噪本身）。

        VAD 按 512 样本窗逐个推进，并在每个窗口后即时处理状态转换——
        即使客户端一次发来大块音频（如整句/REST），speech_start/end 与
        段音频捕获也不会丢失。
        """
        if len(audio) == 0:
            return
        self._push_preroll(audio)
        self._stream.accept_waveform(SR, audio)
        t0 = time.perf_counter()
        # sherpa-onnx 1.13: 每次 decode_stream 只推进一步，须循环到 is_ready 为假
        steps = 0
        while self._rec.is_ready(self._stream) and steps < 100000:
            self._rec.decode_stream(self._stream)
            steps += 1
        self.seg_asr_ms += (time.perf_counter() - t0) * 1000.0

        w = int(self._vad.config.silero_vad.window_size)
        self._vad_buf = np.concatenate([self._vad_buf, audio])
        while len(self._vad_buf) >= w:
            win = self._vad_buf[:w]
            self._vad_buf = self._vad_buf[w:]
            self._vad.accept_waveform(win)
            self.total_fed += w
            if self._capturing:
                self._append_seg(win)

            # VAD 已完成段弹出 → speech_end + final
            while not self._vad.empty():
                seg = self._vad.front
                self._vad.pop()
                start_t = seg.start / SR
                end_t = (seg.start + len(seg.samples)) / SR
                self.in_speech = False
                events.append({"type": "vad", "state": "speech_end",
                               "t": round(end_t, 2), "segment": self.seg_index})
                seg_pcm = self._take_seg_pcm()
                text_now = self._current_text()
                if self._forced_final:
                    self._forced_final = False
                    self.last_partial = ""
                    self._rec.reset(self._stream)
                    self.seg_asr_ms = 0.0
                    self.seg_denoise_ms = 0.0
                elif text_now:
                    self._emit_final(events, text_now, start_t, end_t, pcm=seg_pcm)
                else:
                    # 空文本段（噪声/静音误触发）：不出 final，仅回收状态
                    self.last_partial = ""
                    self._rec.reset(self._stream)
                    self.seg_asr_ms = 0.0
                    self.seg_denoise_ms = 0.0

            # speech_start
            if self._vad.is_speech_detected() and not self.in_speech:
                self.in_speech = True
                self.seg_start_t = self._now_t()
                self._speech_start_wall = time.monotonic()
                self._first_partial_wall = None
                self._forced_final = False
                # 开始为精修缓存段音频（含 pre-roll 补音头）
                self._seg_pcm = list(self._preroll)
                self._seg_pcm_len = sum(len(x) for x in self._seg_pcm)
                self._capturing = True
                events.append({"type": "vad", "state": "speech_start",
                               "t": self._now_t(), "segment": self.seg_index})

        text = self._current_text()

        # partial
        if text and text != self.last_partial:
            self.last_partial = text
            if self._first_partial_wall is None and self._speech_start_wall is not None:
                self._first_partial_wall = time.monotonic()
            events.append({"type": "partial", "text": text,
                           "segment": self.seg_index, "t": self._now_t()})

        # recognizer 端点兜底（长句尾静音 / 超 20s 强切）
        if self._rec.is_endpoint(self._stream):
            if text and not self._forced_final:
                self._emit_final(events, text, self.seg_start_t, self._now_t(),
                                 pcm=self._take_seg_pcm())
                self._forced_final = self.in_speech
                if self.in_speech:
                    events.append({"type": "vad", "state": "speech_end",
                                   "t": self._now_t(), "segment": self.seg_index - 1})
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
                self._emit_final(events, self._current_text(), start_t, end_t,
                                 pcm=self._take_seg_pcm())
            else:
                self._take_seg_pcm()
            self._forced_final = False
        # 最终兜底：VAD 从未触发（如超短语音）但识别器已有文本
        text = self._current_text()
        if text:
            self._emit_final(events, text, self.seg_start_t, self._now_t(),
                             pcm=self._take_seg_pcm())
        return events

    @property
    def first_partial_latency_ms(self) -> float | None:
        if self._speech_start_wall and self._first_partial_wall:
            return (self._first_partial_wall - self._speech_start_wall) * 1000.0
        return None
