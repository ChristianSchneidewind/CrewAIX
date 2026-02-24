#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from atproto import Client
from dotenv import load_dotenv


def _load_queue(latest_queue_path: Path) -> list[dict[str, Any]]:
    data = json.loads(latest_queue_path.read_text(encoding="utf-8"))
    queue = data.get("queue") or []
    if not isinstance(queue, list):
        raise ValueError("Queue JSON must contain a list under 'queue'.")
    return queue


def _find_latest_queue(out_dir: Path) -> Path:
    candidates = list(out_dir.glob("post_queue_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No post_queue_*.json files found in {out_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    load_dotenv()

    handle = os.getenv("BSKY_HANDLE")
    app_password = os.getenv("BSKY_APP_PASSWORD")
    service_url = os.getenv("BSKY_SERVICE_URL", "https://bsky.social")
    out_dir = Path(os.getenv("OUT_DIR", "out"))

    if not handle or not app_password:
        raise SystemExit("Missing BSKY_HANDLE or BSKY_APP_PASSWORD in .env")

    latest_queue = _find_latest_queue(out_dir)
    queue = _load_queue(latest_queue)
    if not queue:
        raise SystemExit(f"Queue is empty in {latest_queue}")

    client = Client(base_url=service_url)
    client.login(handle, app_password)

    for item in queue:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        language = (item.get("language") or "").strip()
        langs = [language] if language else None
        client.send_post(text=text, langs=langs)

    print(f"Posted {len(queue)} item(s) from {latest_queue}")


if __name__ == "__main__":
    main()
