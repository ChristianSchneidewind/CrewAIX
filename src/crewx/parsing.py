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
    """
    Best-effort: find first {...} object even if model prints extra text.
    """
    if not raw:
        return None
    # Quick path: raw is already JSON
    raw_strip = raw.strip()
    if raw_strip.startswith("{") and raw_strip.endswith("}"):
        return raw_strip

    # Search for balanced-ish JSON object by locating first '{' and last '}'.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw[start : end + 1].strip()
    return candidate


def parse_tweets_response(raw: str, *, n_tweets: int) -> dict[str, Any]:
    """
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
        raise ValueError(f"Invalid JSON in model output: {e}") from e

    tweets = data.get("tweets")
    if not isinstance(tweets, list):
        raise ValueError('JSON must contain key "tweets" as a list.')

    norm: list[dict[str, Any]] = []
    for t in tweets:
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

        norm.append(
        {
            "tweet_type": tweet_type,
            "text": text,
            "language": language,
            "tags": [str(x).strip() for x in tags_list if str(x).strip()],
        }
    )


    if not norm:
        raise ValueError("Parsed JSON but no usable tweets were found.")

    return {"tweets": norm[:n_tweets]}
