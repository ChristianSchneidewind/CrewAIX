from __future__ import annotations

import re
import time
from pathlib import Path

from crewx.io import ensure_dir


class RateLimitHit(RuntimeError):
    pass


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "rate_limit" in message or "429" in message


def is_request_too_large(exc: Exception) -> bool:
    message = str(exc).lower()
    return "request too large" in message or "must be reduced" in message


def is_connection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "connection error" in message or "no route to host" in message or "connecterror" in message


def parse_retry_after_seconds(message: str) -> float | None:
    lower = (message or "").lower()
    match = re.search(r"try again in\s+([0-9.]+)\s*(ms|s)", lower)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ms":
        return value / 1000.0
    return value


def _append_text(path: str, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(text)


def kickoff_with_retry(
    crew,
    *,
    max_retries: int = 6,
    base_delay: float = 2.0,
    fail_fast_on_rate_limit: bool = False,
    debug_path: str | None = None,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return str(crew.kickoff() or "")
        except Exception as exc:
            if is_rate_limit_error(exc):
                if fail_fast_on_rate_limit:
                    raise RateLimitHit(str(exc))
                if attempt < max_retries:
                    message = str(exc)
                    retry_after = parse_retry_after_seconds(message)
                    delay = retry_after if retry_after is not None else base_delay * (attempt + 1)
                    time.sleep(delay + 0.25)
                    last_exc = exc
                    continue
            if is_connection_error(exc) and attempt < max_retries:
                delay = min(60.0, base_delay * (attempt + 1) * 3)
                if debug_path:
                    _append_text(
                        debug_path,
                        f"CONNECTION ERROR\nattempt={attempt + 1}\ndelay={delay}s\nerror={exc}\n\n",
                    )
                time.sleep(delay)
                last_exc = exc
                continue
            raise
    if last_exc:
        raise last_exc
    return ""
