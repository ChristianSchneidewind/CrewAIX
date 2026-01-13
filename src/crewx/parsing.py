import json
import re
from pathlib import Path
from crewx.io import write_text


def save_raw(out_dir: str, raw: str) -> Path:
    p = Path(out_dir) / "last_raw_output.txt"
    write_text(str(p), (raw or ""))
    return p


def extract_json_object_from_text(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None

    extracted = None

    # ```json ... ```
    if "```" in raw:
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
        if m:
            extracted = m.group(1).strip()

    # outermost {...}
    if extracted is None:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            extracted = raw[start : end + 1].strip()

    if extracted is None:
        return None

    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return None


def fallback_parse_tweets_from_text(raw: str, n_tweets: int) -> dict | None:
    """
    If the model doesn't return JSON, try to parse common list formats into tweets.
    Supports:
      - 'Tweet 1: ...'
      - '1) ...' / '1. ...'
      - '- ...' / '* ...'
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    tweets: list[str] = []

    # A) "Tweet 1: ..."
    matches = re.findall(
        r"(?:^|\n)\s*(?:Tweet|TWEET)\s*\d+\s*[:\-]\s*(.+?)(?=\n\s*(?:Tweet|TWEET)\s*\d+\s*[:\-]|\Z)",
        raw,
        flags=re.DOTALL,
    )
    for m in matches:
        txt = " ".join(m.strip().splitlines()).strip()
        if txt:
            tweets.append(txt)

    # B) Numbered list: 1) ... or 1. ...
    if not tweets:
        matches = re.findall(
            r"(?:^|\n)\s*\d+\s*[.)]\s+(.+?)(?=\n\s*\d+\s*[.)]\s+|\Z)",
            raw,
            flags=re.DOTALL,
        )
        for m in matches:
            txt = " ".join(m.strip().splitlines()).strip()
            if txt:
                tweets.append(txt)

    # C) Bullets: - ... or * ...
    if not tweets:
        matches = re.findall(
            r"(?:^|\n)\s*[-*]\s+(.+?)(?=\n\s*[-*]\s+|\Z)",
            raw,
            flags=re.DOTALL,
        )
        for m in matches:
            txt = " ".join(m.strip().splitlines()).strip()
            if txt:
                tweets.append(txt)

    tweets = tweets[:n_tweets]
    if not tweets:
        return None

    return {
        "tweets": [
            {"text": t[:240], "language": "de", "tags": [], "intent": "other"}
            for t in tweets
        ]
    }
