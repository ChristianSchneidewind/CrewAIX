from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Iterable, List, Optional

from crewai import Agent, Crew, Task

from crewx.config import Settings
from crewx.io import ensure_dir, read_text, write_json, write_text
from crewx.llm import build_llm_from_env
from crewx.parsing import TweetType, load_tweet_types, parse_tweets_output
from crewx.prompts import (
    build_generate_task_description,
    build_regen_task_description,
    build_repair_task_description,
)

# -----------------------------
# Helpers
# -----------------------------
def now_str() -> str:
    # e.g. 2026-01-19_14-46-19
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = " ".join(s.split())
    return s


def is_near_duplicate(candidate: str, existing: Iterable[str], threshold: float = 0.90) -> bool:
    cand = norm_text(candidate)
    if not cand:
        return True
    for ex in existing:
        ratio = SequenceMatcher(None, cand, norm_text(ex)).ratio()
        if ratio >= threshold:
            return True
    return False


def _save_raw(out_dir: str, raw: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "last_raw_output.txt")
    write_text(path, raw or "")
    return path


def _kickoff_single(agent: Agent, task: Task) -> str:
    # CrewAI returns various shapes sometimes; normalize to str
    result = Crew(agents=[agent], tasks=[task]).kickoff()
    if result is None:
        return ""
    return str(result)


def _parse_raw_with_optional_repair(
    agent: Agent,
    raw: str,
    out_dir: str,
    n_tweets: int,
    allow_repair: bool = True,
) -> dict[str, Any]:
    """
    1) Save raw
    2) Try parse locally (robust)
    3) If parse fails and allow_repair -> run repair task once, parse again
    """
    _save_raw(out_dir, raw)

    try:
        return parse_tweets_output(raw, expected_count=n_tweets)
    except Exception:
        if not allow_repair:
            raise

    # Repair only once (prevents infinite loops + long timeouts)
    repair_task = Task(
        description=build_repair_task_description(raw, n_tweets),
        expected_output="Valid JSON only.",
    )
    repaired_raw = _kickoff_single(agent, repair_task)
    _save_raw(out_dir, repaired_raw)

    return parse_tweets_output(repaired_raw, expected_count=n_tweets)


# -----------------------------
# Main pipeline
# -----------------------------
def run_generate_tweets() -> None:
    settings = Settings()

    out_dir = settings.out_dir
    ensure_dir(out_dir)

    tweets_md = read_text(settings.tweets_md_path)

    # tweet_types.md is expected next to tweets.md (content/)
    tweet_types_path = os.path.join(os.path.dirname(settings.tweets_md_path), "tweet_types.md")
    tweet_types: List[TweetType] = load_tweet_types(tweet_types_path)

    if not tweet_types:
        raise RuntimeError(
            f"No tweet types found in {tweet_types_path}. "
            "Add at least one '## <TypeName>' section."
        )

    n_tweets = settings.n_tweets

    llm = build_llm_from_env()
    agent = Agent(
        role="X/Twitter Copywriter",
        goal="Generate high-quality, varied tweets for the given company context and tweet types.",
        backstory="You write concise, punchy tweets with strong clarity and no fluff.",
        llm=llm,
        verbose=False,
    )

    # Distribute requested tweets across types
    per_type_target = int(math.ceil(n_tweets / len(tweet_types)))

    all_tweets: List[dict[str, Any]] = []
    existing_texts: List[str] = []

    # -------------
    # Generate per type
    # -------------
    for tt in tweet_types:
        target_for_this_type = per_type_target

        # Attempt 1: fresh generation
        gen_task = Task(
            description=build_generate_task_description(
                tweets_md=tweets_md,
                tweet_type_name=tt.name,
                tweet_type_description=tt.description,
                n_tweets=target_for_this_type,
            ),
            expected_output="Valid JSON only.",
        )
        raw = _kickoff_single(agent, gen_task)

        data = _parse_raw_with_optional_repair(
            agent=agent,
            raw=raw,
            out_dir=out_dir,
            n_tweets=target_for_this_type,
            allow_repair=True,
        )

        tweets = data.get("tweets", [])
        # Deduplicate (near-duplicate)
        kept: List[dict[str, Any]] = []
        for t in tweets:
            text = str(t.get("text", "")).strip()
            if not text:
                continue
            if is_near_duplicate(text, existing_texts, threshold=0.90):
                continue
            t["tweet_type"] = tt.name
            kept.append(t)
            existing_texts.append(text)

        all_tweets.extend(kept)

        # If we still need more for this type, do a regen loop (bounded)
        attempts = 0
        while attempts < 2 and sum(1 for t in all_tweets if t.get("tweet_type") == tt.name) < target_for_this_type:
            missing = target_for_this_type - sum(1 for t in all_tweets if t.get("tweet_type") == tt.name)

            regen_task = Task(
                description=build_regen_task_description(
                    tweets_md=tweets_md,
                    tweet_type_name=tt.name,
                    tweet_type_description=tt.description,
                    n_tweets=missing,
                    already_generated_texts=existing_texts,
                ),
                expected_output="Valid JSON only.",
            )
            regen_raw = _kickoff_single(agent, regen_task)

            regen_data = _parse_raw_with_optional_repair(
                agent=agent,
                raw=regen_raw,
                out_dir=out_dir,
                n_tweets=missing,
                allow_repair=True,
            )

            regen_tweets = regen_data.get("tweets", [])
            for t in regen_tweets:
                text = str(t.get("text", "")).strip()
                if not text:
                    continue
                if is_near_duplicate(text, existing_texts, threshold=0.90):
                    continue
                t["tweet_type"] = tt.name
                all_tweets.append(t)
                existing_texts.append(text)

            attempts += 1

    # Trim to requested count (in case we generated extra)
    all_tweets = all_tweets[:n_tweets]

    # --- Write outputs ---
    timestamp = now_str()
    out_json_path = os.path.join(out_dir, f"tweets_{timestamp}.json")
    out_md_path = os.path.join(out_dir, f"tweets_{timestamp}.md")

    payload = {"tweets": all_tweets}
    write_json(out_json_path, payload)

    # Markdown output
    md_lines: List[str] = []
    md_lines.append(f"# Tweets ({timestamp})")
    md_lines.append("")
    for i, t in enumerate(all_tweets, start=1):
        md_lines.append(f"## {i}. ({t.get('tweet_type', 'unknown')})")
        md_lines.append(t.get("text", "").strip())
        md_lines.append("")

    write_text(out_md_path, "\n".join(md_lines).rstrip() + "\n")

    print(f"Wrote: {out_json_path}")
    print(f"Wrote: {out_md_path}")
