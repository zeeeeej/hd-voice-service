"""结构化日志 + X-Request-Id 上下文。"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "rid": request_id_var.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).handlers = [handler]


def log_extra(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})
