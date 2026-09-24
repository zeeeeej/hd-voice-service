"""X-API-Key 鉴权（WS 允许 ?api_key= 查询参数兜底，因浏览器 WS 无法自定义头）。"""
from __future__ import annotations

from fastapi import Header, Query, Request

from ..errors import VoiceError


class Unauthorized(VoiceError):
    status = 401
    ws_close_code = 1008
    code = "unauthorized"


def _ok(provided: str | None, expected: str) -> bool:
    return provided is not None and provided == expected


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    api_key: str | None = Query(default=None),
) -> None:
    expected = request.app.state.settings.api_key
    if not _ok(x_api_key, expected) and not _ok(api_key, expected):
        raise Unauthorized("invalid or missing X-API-Key")


def ws_authorized(headers, query_params, expected: str) -> bool:
    return _ok(headers.get("x-api-key"), expected) or _ok(query_params.get("api_key"), expected)
