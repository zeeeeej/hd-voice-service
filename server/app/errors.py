"""统一错误类型：REST 映射 HTTP 状态码，WS 映射 error 帧 + close code。"""
from __future__ import annotations


class VoiceError(Exception):
    status = 500
    ws_close_code = 1011
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequest(VoiceError):
    status = 400
    ws_close_code = 1008
    code = "bad_request"


class UnsupportedOption(BadRequest):
    code = "unsupported_option"


class Overloaded(VoiceError):
    status = 429
    ws_close_code = 1013
    code = "overloaded"


class NotReady(VoiceError):
    status = 503
    ws_close_code = 1013
    code = "not_ready"
