from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def list_recent_tweet_texts(out_dir: str, *, limit: int) -> list[str]:
    """
    Collect tweet texts from newest JSON outputs. Best-effort; ignores broken files.
    """
    p = Path(out_dir)
    if not p.exists():
        return []

    # newest first
    json_files = sorted(p.glob("tweets_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    texts: list[str] = []
    for jf in json_files:
        if len(texts) >= limit:
            break
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            for t in data.get("tweets", []):
                txt = (t.get("text") or "").strip()
                if txt:
                    texts.append(txt)
                    if len(texts) >= limit:
                        break
        except Exception:
            continue
    return texts
