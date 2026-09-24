"""TTS 输入文本按句切分：
- 遇 。！？；!?\n;… 必切（硬边界）
- 缓冲超过 MAX_BUFFER 字仍无硬边界 → 按最后一个软边界 ，、,：: 切
- 禁止产出单字碎片（flush 时除外），禁止纯标点/空白碎片
"""
from __future__ import annotations

HARD = "。！？；!?;\n…："
SOFT = "，、,—"
MAX_BUFFER = 30
MIN_CHUNK = 2


class TextChunker:
    def __init__(self, max_buffer: int = MAX_BUFFER, min_chunk: int = MIN_CHUNK,
                 fast_first: bool = True):
        self.buf = ""
        self.max_buffer = max_buffer
        self.min_chunk = min_chunk
        self.fast_first = fast_first  # 会话首句在软边界提前切分，压低 TTS 首包延迟
        self._emitted = False

    def push(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        # fast-first：尚未产出任何句子时，在第一个软边界切一次（仅限一次）
        if self.fast_first and not self._emitted:
            idx = next((i for i, ch in enumerate(self.buf) if ch in SOFT), -1)
            if idx >= self.min_chunk - 1 and len(self.buf) > idx + 1:
                chunk = self.buf[: idx + 1]
                self.buf = self.buf[idx + 1:]
                if self._meaningful(chunk):
                    self._emitted = True
                    out.append(chunk)
        while True:
            idx = next((i for i, ch in enumerate(self.buf) if ch in HARD), -1)
            if idx >= 0:
                chunk = self.buf[: idx + 1]
                self.buf = self.buf[idx + 1:]
                if self._meaningful(chunk):
                    self._emitted = True
                    out.append(chunk)
                elif out:
                    out[-1] += chunk  # 标点粘连到前一块
                continue
            if len(self.buf) > self.max_buffer:
                soft_idx = max((self.buf.rfind(c, 0, len(self.buf)) for c in SOFT), default=-1)
                if soft_idx >= self.min_chunk - 1:
                    chunk = self.buf[: soft_idx + 1]
                    self.buf = self.buf[soft_idx + 1:]
                    if self._meaningful(chunk):
                        out.append(chunk)
                    continue
            break
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []

    @staticmethod
    def _meaningful(chunk: str) -> bool:
        stripped = chunk.strip()
        if not stripped:
            return False
        # 至少包含一个非标点字符
        return any(ch not in HARD + SOFT + " \t" for ch in stripped)
