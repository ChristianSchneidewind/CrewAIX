from __future__ import annotations

import logging
from pathlib import Path

from crewx.io import ensure_dir


def setup_logging(out_dir: str, *, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("crewx")
    if logger.handlers:
        return logger

    ensure_dir(out_dir)
    level = logging.DEBUG if verbose else logging.INFO

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    log_path = Path(out_dir) / "run.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
