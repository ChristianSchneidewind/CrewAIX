from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TweetType:
    name: str
    description: str


def load_tweet_types(path: str) -> List[TweetType]:
    """
    Format in tweet_types.md (simple + tolerant):

    # Tweet Types
    ## marketing
    - Description...
    ## educational
    - Description...
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on headings level 2 "## "
    parts = re.split(r"(?m)^##\s+", content)
    if len(parts) <= 1:
        return []

    tweet_types: List[TweetType] = []
    for chunk in parts[1:]:
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        desc = "\n".join(lines[1:]).strip()
        # remove leading bullets in description
        desc = re.sub(r"(?m)^\s*[-*]\s+", "", desc).strip()
        tweet_types.append(TweetType(name=name, description=desc))

    return tweet_types


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_value(text: str) -> Any:
    """
    Robust-ish extraction: find first {...} block that parses as JSON.
    """
    text = _strip_code_fences(text)

    # Quick path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Greedy scan for object
    start_indices = [m.start() for m in re.finditer(r"\{", text)]
    for start in start_indices:
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            candidate = text[start:end]
            try:
                return json.loads(candidate)
            except Exception:
                continue

    raise ValueError("Could not parse JSON from output.")


def parse_tweets_output(raw: str, expected_count: Optional[int] = None) -> Dict[str, Any]:
    if raw is None or not str(raw).strip():
        raise ValueError("Empty LLM output (raw is blank).")

    data = _extract_first_json_value(raw)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")

    tweets = data.get("tweets")
    if not isinstance(tweets, list):
        raise ValueError('JSON must contain key "tweets" as a list.')

    norm: List[Dict[str, Any]] = []
    for i, t in enumerate(tweets):
        if not isinstance(t, dict):
            raise ValueError(f"Tweet #{i} must be an object.")
        text = str(t.get("text", "")).strip()
        ttype = str(t.get("tweet_type", "")).strip()
        if not text:
            raise ValueError(f"Tweet #{i} missing text.")
        if not ttype:
            raise ValueError(f"Tweet #{i} missing tweet_type.")
        norm.append({"tweet_type": ttype, "text": text})

    if expected_count is not None and len(norm) != expected_count:
        raise ValueError(f"Expected {expected_count} tweets, got {len(norm)}.")

    return {"tweets": norm}
