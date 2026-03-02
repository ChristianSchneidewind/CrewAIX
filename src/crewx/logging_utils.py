from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crewx.io import ensure_dir


class RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
        }
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            payload.update(extra_data)
        return json.dumps(payload, ensure_ascii=False)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"extra_data": {"event": event, **fields}})


def setup_logging(
    out_dir: str,
    *,
    verbose: bool,
    run_id: str,
    json_logs: bool = True,
    log_dir: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger("crewx")
    if logger.handlers:
        return logger

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    log_root = Path(log_dir) if log_dir else Path(out_dir) / "logs"
    ensure_dir(log_root)

    base_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(run_id)s]: %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(base_formatter)

    text_log_path = log_root / f"run_{run_id}.log"
    file_handler = logging.FileHandler(text_log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(base_formatter)

    run_filter = RunIdFilter(run_id)
    stream_handler.addFilter(run_filter)
    file_handler.addFilter(run_filter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    if json_logs:
        json_log_path = log_root / f"run_{run_id}.jsonl"
        json_handler = logging.FileHandler(json_log_path, encoding="utf-8")
        json_handler.setLevel(level)
        json_handler.setFormatter(JsonFormatter())
        json_handler.addFilter(run_filter)
        logger.addHandler(json_handler)

    logger.propagate = False
    return logger
