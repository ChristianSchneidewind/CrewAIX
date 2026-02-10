from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TweetType:
    name: str
    goal: str
    style: list[str]
    rules: list[str]

    @property
    def description(self) -> str:
        # For compatibility with earlier prompts.py versions that used tt.description
        style = "\n".join(f"- {s}" for s in self.style) if self.style else "- (none)"
        rules = "\n".join(f"- {r}" for r in self.rules) if self.rules else "- (none)"
        return f"TYPE GOAL:\n{self.goal}\n\nSTYLE GUIDELINES:\n{style}\n\nCONTENT RULES:\n{rules}"


def parse_tweet_types_md(md: str) -> list[TweetType]:
    """
    Parses your content/tweet_types.md structure:
    - # Tweet Types
    - ## marketing
    - Goal: ...
    - Style:
      - ...
    - Rules:
      - ...
    """
    # Split on "## <name>"
    blocks = re.split(r"(?m)^\s*##\s+", md)
    if not blocks:
        return []

    # First block is header before first type
    types: list[TweetType] = []
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        rest = "\n".join(lines[1:]).strip()

        goal = ""
        style: list[str] = []
        rules: list[str] = []

        # Goal: line
        m_goal = re.search(r"(?mi)^\s*Goal:\s*(.+)\s*$", rest)
        if m_goal:
            goal = m_goal.group(1).strip()

        # Style section bullets
        m_style = re.search(r"(?ms)^\s*Style:\s*\n(.*?)(?:\n\s*Rules:\s*\n|\Z)", rest)
        if m_style:
            style_block = m_style.group(1)
            style = [re.sub(r"^\s*-\s*", "", l).strip() for l in style_block.splitlines() if l.strip().startswith("-")]

        # Rules section bullets
        m_rules = re.search(r"(?ms)^\s*Rules:\s*\n(.*?)(?:\Z)", rest)
        if m_rules:
            rules_block = m_rules.group(1)
            rules = [re.sub(r"^\s*-\s*", "", l).strip() for l in rules_block.splitlines() if l.strip().startswith("-")]

        types.append(TweetType(name=name, goal=goal, style=style, rules=rules))

    return types


def _extract_first_json_object(raw: str) -> str | None:
    """Best-effort: find first JSON object or array even if model prints extra text.

    Strategy:
    1) If the whole stripped string is valid JSON -> use it.
    2) Otherwise, try to find the first valid JSON array "[...]".
    3) If that fails, try to find the first valid JSON object "{...}".
    """
    if not raw:
        return None

    raw_strip = raw.strip()

    # 1) Quick path: whole string is valid JSON (object or array)
    if (raw_strip.startswith("{") and raw_strip.endswith("}")) or (
        raw_strip.startswith("[") and raw_strip.endswith("]")
    ):
        try:
            json.loads(raw_strip)
            return raw_strip
        except json.JSONDecodeError:
            pass

    # Helper to scan for first valid JSON by delimiter
    def _scan_for_valid_segment(text: str, open_char: str, close_char: str) -> str | None:
        start = text.find(open_char)
        if start == -1:
            return None
        for end in range(len(text) - 1, start, -1):
            if text[end] != close_char:
                continue
            candidate = text[start : end + 1].strip()
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return candidate
        return None

    # 2) Prefer arrays if present
    segment = _scan_for_valid_segment(raw, "[", "]")
    if segment is not None:
        return segment

    # 3) Fallback: try objects
    segment = _scan_for_valid_segment(raw, "{", "}")
    if segment is not None:
        return segment

    return None


def _normalize_tag(tag: str) -> str:
    t = tag.strip().lower()
    if t.startswith("#"):
        t = t[1:]
    mapping = {
        "myth": "myth_vs_fact",
        "mythos": "myth_vs_fact",
        "myth_vs_fact": "myth_vs_fact",
        "fact": "myth_vs_fact",
        "fakt": "myth_vs_fact",
    }
    return mapping.get(t, t)


def parse_tweets_response(raw: str, *, n_tweets: int) -> dict[str, Any]:
    """Parse model output into a normalized tweets structure.

    Returns: {"tweets": [ {tweet_type,text,language,tags}, ... ] }
    Ensures at most n_tweets tweets, and normalizes fields.
    Raises ValueError on failure.
    """
    json_str = _extract_first_json_object(raw)
    if not json_str:
        raise ValueError("No JSON object found in model output.")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Give a slightly more helpful error including a snippet
        snippet = json_str[:200].replace("\n", " ")
        raise ValueError(f"Invalid JSON in model output: {e} | snippet: {snippet!r}") from e

    # Accept either a plain list of tweets OR an object with "tweets" key
    if isinstance(data, list):
        tweets = data
    else:
        tweets = data.get("tweets")

    if not isinstance(tweets, list):
        raise ValueError('JSON must be a list or contain key "tweets" as a list.')

    norm: list[dict[str, Any]] = []
    for t in tweets:
        if isinstance(t, str):
            text = t.strip()
            if not text:
                continue
            norm.append(
                {
                    "tweet_type": "unknown",
                    "text": text,
                    "language": "de",
                    "tags": [],
                }
            )
            continue

        if not isinstance(t, dict):
            continue
        tweet_type = str(t.get("tweet_type") or "").strip()
        text = str(t.get("text") or "").strip()
        language = str(t.get("language") or "de").strip() or "de"
        tags_raw = t.get("tags")
        if isinstance(tags_raw, list):
            tags_list = tags_raw
        else:
            tags_list = []

        tags_norm = [_normalize_tag(str(x)) for x in tags_list if str(x).strip()]
        tags_norm = [t for t in tags_norm if t]

        norm.append(
            {
                "tweet_type": tweet_type,
                "text": text,
                "language": language,
                "tags": tags_norm,
            }
        )

    if not norm:
        raise ValueError("Parsed JSON but no usable tweets were found.")

    return {"tweets": norm[:n_tweets]}
