from __future__ import annotations

from pathlib import Path
import json
import time
import logging
import re

from crewai import Agent, Crew, Process, Task

from crewx.config import apply_litellm_env, load_settings
from crewx.embeddings import build_embedding_map, embed_texts, is_embedding_auth_error
from crewx.filters import (
    accept_relaxed_candidate,
    assign_missing_types,
    filter_crewai_tweets,
    normalize_candidate_fields,
)
from crewx.io import (
    append_jsonl,
    ensure_dir,
    list_recent_tweet_texts,
    now_timestamp,
    read_text,
    write_json,
    write_text,
)
from crewx.llm import build_llm
from crewx.logging_utils import setup_logging
from crewx.parsing import TweetType, parse_tweet_types_md, parse_tweets_response
from crewx.prompts_pipeline import (
    build_generator_prompt,
    build_post_prompt,
    build_review_prompt,
    format_types_md,
)
from crewx.retry import (
    RateLimitHit,
    kickoff_with_retry,
    is_request_too_large,
    parse_retry_after_seconds,
)
from crewx.rules import extract_bucket, infer_bucket_from_text

LOGGER_NAME = "crewx.pipeline"


def _append_text(path: str, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _parse_roles_md(md: str) -> dict[str, dict[str, str]]:
    blocks = re.split(r"(?m)^\s*##\s+", md)
    roles: dict[str, dict[str, str]] = {}
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        key = lines[0].strip().lower()
        rest = "\n".join(lines[1:])
        role = ""
        goal = ""
        backstory = ""
        m_role = re.search(r"(?mi)^\s*Role:\s*(.+)\s*$", rest)
        if m_role:
            role = m_role.group(1).strip()
        m_goal = re.search(r"(?ms)^\s*Goal:\s*(.+?)(?:\n\s*Backstory:|\Z)", rest)
        if m_goal:
            goal = m_goal.group(1).strip()
        m_backstory = re.search(r"(?ms)^\s*Backstory:\s*(.+?)\s*$", rest)
        if m_backstory:
            backstory = m_backstory.group(1).strip()
        roles[key] = {"role": role, "goal": goal, "backstory": backstory}
    return roles


def _load_roles(path: str) -> dict[str, dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return {}
    return _parse_roles_md(p.read_text(encoding="utf-8"))


def _rotation_start_index(out_dir: str, *, total_types: int) -> int:
    """Deterministic rotation based on history length.

    Uses number of non-empty history lines to pick the next start index.
    """
    if total_types <= 0:
        return 0
    history_path = Path(out_dir) / "history.jsonl"
    if not history_path.exists():
        return 0
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return 0
    non_empty = sum(1 for line in lines if line.strip())
    return non_empty % total_types


def fix_history_unknown_types(out_dir: str, fallback_type: str = "educational") -> int:
    history_path = Path(out_dir) / "history.jsonl"
    if not history_path.exists():
        return 0
    lines = history_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    changed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except Exception:
            updated.append(line)
            continue
        t = (data.get("tweet_type") or "").strip().lower()
        if t == "unknown":
            data["tweet_type"] = fallback_type
            changed += 1
        updated.append(json.dumps(data, ensure_ascii=False))
    history_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return changed


def _build_crews(
    *,
    generator_agent: Agent,
    reviewer_agent: Agent,
    poster_agent: Agent,
    company_md: str,
    ideas_md: str | None,
    recent_context: list[str],
    active_types: list[TweetType],
    forced_types: bool,
    n_tweets: int,
) -> tuple[Crew, Crew]:
    required_types = [t.name.strip() for t in active_types]
    effective_n_tweets = len(required_types) if forced_types else n_tweets
    types_md = format_types_md(active_types)

    generate_task = Task(
        description=build_generator_prompt(
            company_md=company_md,
            types_md=types_md,
            ideas_md=ideas_md,
            n_tweets=effective_n_tweets,
            recent=recent_context,
            required_types=required_types if forced_types else None,
        ),
        expected_output="A JSON array of tweet objects.",
        agent=generator_agent,
    )

    review_task = Task(
        description=build_review_prompt(n_tweets=effective_n_tweets),
        expected_output="A JSON array of compliant tweet objects.",
        agent=reviewer_agent,
        context=[generate_task],
    )

    post_task = Task(
        description=build_post_prompt(),
        expected_output="A JSON array of tweets ready to queue.",
        agent=poster_agent,
        context=[review_task],
    )

    crew = Crew(
        agents=[generator_agent, reviewer_agent, poster_agent],
        tasks=[generate_task, review_task, post_task],
        process=Process.sequential,
        verbose=generator_agent.verbose,
    )

    review_only_task = Task(
        description=build_post_prompt(),
        expected_output="A JSON array of tweets ready to queue.",
        agent=poster_agent,
        context=[review_task],
    )

    review_only_crew = Crew(
        agents=[generator_agent, reviewer_agent, poster_agent],
        tasks=[generate_task, review_task, review_only_task],
        process=Process.sequential,
        verbose=generator_agent.verbose,
    )

    return crew, review_only_crew


def run_generate_tweets_crewai() -> None:
    apply_litellm_env()
    settings = load_settings()
    ensure_dir(settings.out_dir)
    setup_logging(settings.out_dir, verbose=settings.verbose)
    pipeline_logger = logging.getLogger(LOGGER_NAME)
    pipeline_logger.info("Starting tweet generation")

    company_md = read_text(settings.tweets_md_path)
    types_md = read_text(settings.tweet_types_md_path)
    roles = _load_roles(settings.crew_roles_md_path)

    ideas_md = None
    ideas_path = Path(settings.ideas_md_path)
    if ideas_path.exists():
        ideas_md = ideas_path.read_text(encoding="utf-8")

    all_types = parse_tweet_types_md(types_md)
    if not all_types:
        raise RuntimeError(f"No tweet types found in {settings.tweet_types_md_path}")

    forced_types = [t.strip().lower() for t in settings.forced_tweet_types if t.strip()]
    if forced_types:
        active_types = [
            t for t in all_types if t.name.strip().lower() in set(forced_types)
        ]
    else:
        max_types = min(settings.n_tweets, len(all_types))
        start_idx = _rotation_start_index(settings.out_dir, total_types=len(all_types))
        active_types = [
            all_types[(start_idx + i) % len(all_types)] for i in range(max_types)
        ]
        forced_types = [t.name.strip().lower() for t in active_types]

    fix_history_unknown_types(settings.out_dir, fallback_type="educational")
    recent = list_recent_tweet_texts(settings.out_dir, limit=settings.recent_tweets_max)

    llm = build_llm(settings)

    generator_role = roles.get("generator", {})
    reviewer_role = roles.get("reviewer", {})
    poster_role = roles.get("poster", {})

    generator_agent = Agent(
        role=generator_role.get("role") or "Tweet Generator",
        goal=generator_role.get("goal")
        or "Generate varied German tweets that follow the provided constraints.",
        backstory=generator_role.get("backstory")
        or "You are an expert social media writer for travel and passenger rights.",
        llm=llm,
        verbose=settings.verbose,
    )

    reviewer_agent = Agent(
        role=reviewer_role.get("role") or "X Compliance Reviewer",
        goal=reviewer_role.get("goal")
        or "Ensure tweets comply with X constraints and style rules.",
        backstory=reviewer_role.get("backstory")
        or "You are a strict reviewer who fixes or removes non-compliant tweets.",
        llm=llm,
        verbose=settings.verbose,
    )

    poster_agent = Agent(
        role=poster_role.get("role") or "Tweet Poster",
        goal=poster_role.get("goal")
        or "Prepare final tweets for the posting queue without altering content.",
        backstory=poster_role.get("backstory")
        or "You only prepare a queue; you never call external APIs.",
        llm=llm,
        verbose=settings.verbose,
    )

    base_active_types = active_types

    last_raw_path = f"{settings.out_dir}/last_raw_output.txt"
    write_text(last_raw_path, "")
    tweets: list[dict] = []
    max_attempts = 3
    embedding_disabled = False

    def _build_context_limits(force_minimal: bool) -> list[int]:
        if force_minimal:
            return [0]
        limits: list[int] = []
        for size in [settings.recent_tweets_max, 10, 5, 0]:
            if size is None:
                continue
            size = min(size, len(recent))
            if size not in limits:
                limits.append(size)
        return limits

    def _build_n_tweet_levels(force_minimal: bool) -> list[int]:
        if force_minimal:
            return [1]
        levels: list[int] = []
        for size in [settings.n_tweets, 5, 3, 1]:
            if size <= 0:
                continue
            if size not in levels:
                levels.append(size)
        return levels

    force_minimal = False

    while True:
        context_limits = _build_context_limits(force_minimal)
        n_tweet_levels = _build_n_tweet_levels(force_minimal)
        rate_limit_triggered = False

        for context_limit in context_limits:
            recent_context = recent[-context_limit:] if context_limit > 0 else []

            for n_tweets in n_tweet_levels:
                if forced_types:
                    active_types = base_active_types
                else:
                    active_types = base_active_types[
                        : max(1, min(n_tweets, len(base_active_types)))
                    ]

                effective_n_tweets = len(active_types) if forced_types else n_tweets
                pipeline_logger.info(
                    "Run context: force_minimal=%s recent_context=%s n_tweets=%s types=%s",
                    force_minimal,
                    len(recent_context),
                    effective_n_tweets,
                    ",".join([t.name for t in active_types]),
                )

                _append_text(
                    last_raw_path,
                    f"RUN CONTEXT\nforce_minimal={force_minimal}\nrecent_context={len(recent_context)}\n"
                    f"n_tweets={effective_n_tweets}\nactive_types={','.join([t.name for t in active_types])}\n\n",
                )

                crew, review_only_crew = _build_crews(
                    generator_agent=generator_agent,
                    reviewer_agent=reviewer_agent,
                    poster_agent=poster_agent,
                    company_md=company_md,
                    ideas_md=ideas_md,
                    recent_context=recent_context,
                    active_types=active_types,
                    forced_types=bool(forced_types),
                    n_tweets=effective_n_tweets,
                )

                for attempt in range(max_attempts):
                    try:
                        raw_str = kickoff_with_retry(
                            crew, fail_fast_on_rate_limit=True, debug_path=last_raw_path
                        )
                    except RateLimitHit as exc:
                        retry_after = parse_retry_after_seconds(str(exc))
                        delay = max(retry_after or 0, 60.0)
                        pipeline_logger.warning(
                            "Rate limit hit. Sleeping for %s seconds.", delay
                        )
                        time.sleep(delay + 0.25)
                        rate_limit_triggered = True
                        break
                    except Exception as exc:
                        if is_request_too_large(exc):
                            pipeline_logger.warning(
                                "Request too large; retrying with smaller context"
                            )
                            break
                        raise

                    _append_text(last_raw_path, "RAW OUTPUT\n" + raw_str + "\n\n")

                    try:
                        default_type = (
                            active_types[0].name.strip() if active_types else None
                        )
                        data = parse_tweets_response(
                            raw_str,
                            n_tweets=effective_n_tweets,
                            default_tweet_type=default_type,
                        )
                    except ValueError:
                        try:
                            raw_str = kickoff_with_retry(
                                review_only_crew,
                                fail_fast_on_rate_limit=True,
                                debug_path=last_raw_path,
                            )
                        except RateLimitHit as exc:
                            retry_after = parse_retry_after_seconds(str(exc))
                            delay = max(retry_after or 0, 60.0)
                            pipeline_logger.warning(
                                "Rate limit hit. Sleeping for %s seconds.", delay
                            )
                            time.sleep(delay + 0.25)
                            rate_limit_triggered = True
                            break
                        except Exception as exc:
                            if is_request_too_large(exc):
                                pipeline_logger.warning(
                                    "Request too large; retrying with smaller context"
                                )
                                break
                            raise
                        _append_text(last_raw_path, "RAW OUTPUT\n" + raw_str + "\n\n")
                        try:
                            default_type = (
                                active_types[0].name.strip() if active_types else None
                            )
                            data = parse_tweets_response(
                                raw_str,
                                n_tweets=effective_n_tweets,
                                default_tweet_type=default_type,
                            )
                        except ValueError:
                            continue

                    required_types = [t.name.strip() for t in active_types]
                    data["tweets"] = assign_missing_types(
                        data["tweets"], required_types
                    )
                    data["tweets"] = [
                        normalize_candidate_fields(t) for t in data["tweets"]
                    ]

                    max_travel_hack = 1
                    allowed_types = {t.name.strip().lower() for t in active_types}
                    type_limits = (
                        {t.name.strip().lower(): 1 for t in active_types}
                        if forced_types
                        else None
                    )

                    embedding_threshold = (
                        settings.embedding_similarity_threshold
                        if settings.embedding_model_name
                        and (settings.embedding_api_key or settings.openai_api_key)
                        else None
                    )
                    recent_for_embeddings = recent_context[
                        : settings.embedding_history_max
                    ]
                    recent_embeddings = None
                    candidate_embeddings = None
                    candidate_texts = [
                        (t.get("text") or "").strip()
                        for t in data["tweets"]
                        if (t.get("text") or "").strip()
                    ]
                    embedding_error = None
                    if embedding_threshold and not embedding_disabled:
                        try:
                            recent_embeddings = embed_texts(
                                recent_for_embeddings, settings
                            )
                            candidate_embeddings = build_embedding_map(
                                candidate_texts, settings
                            )
                        except Exception as exc:
                            embedding_error = str(exc)
                            if is_embedding_auth_error(exc):
                                embedding_disabled = True
                                embedding_error += " (embedding disabled)"
                            pipeline_logger.warning(
                                "Embedding error: %s", embedding_error
                            )

                        _append_text(
                            last_raw_path,
                            f"EMBEDDING DEBUG\nrecent={len(recent_for_embeddings)} emb_recent={'yes' if recent_embeddings else 'no'}\n"
                            f"candidates={len(candidate_texts)} emb_candidates={'yes' if candidate_embeddings else 'no'}\n"
                            f"error={embedding_error}\n\n",
                        )

                    tweets = filter_crewai_tweets(
                        data["tweets"],
                        recent,
                        max_travel_hack=max_travel_hack,
                        allowed_types=allowed_types,
                        type_limits=type_limits,
                        embedding_threshold=embedding_threshold,
                        recent_embeddings=recent_embeddings,
                        candidate_embeddings=candidate_embeddings,
                    )
                    if not tweets:
                        fallback = []
                        for t in data["tweets"]:
                            if accept_relaxed_candidate(
                                t, allowed_types=allowed_types, type_limits=type_limits
                            ):
                                fallback = [t]
                                break
                        tweets = fallback

                    if tweets:
                        tweets = filter_crewai_tweets(
                            tweets,
                            recent,
                            max_travel_hack=max_travel_hack,
                            allowed_types=allowed_types,
                            type_limits=type_limits,
                            embedding_threshold=None,
                            recent_embeddings=None,
                            candidate_embeddings=None,
                        )

                    if tweets:
                        pipeline_logger.info("Accepted %s tweets", len(tweets))
                        break

                if tweets or rate_limit_triggered:
                    break

            if tweets or rate_limit_triggered:
                break

        if tweets:
            break
        if rate_limit_triggered and not force_minimal:
            force_minimal = True
            pipeline_logger.warning(
                "Rate limit triggered; retrying with minimal settings"
            )
            continue
        break

    if not tweets:
        pipeline_logger.warning("No tweets produced")
        return

    timestamp = now_timestamp()
    out_queue_path = f"{settings.out_dir}/post_queue_{timestamp}.json"

    output_candidates = tweets[: min(effective_n_tweets, 5)]
    seen_buckets: set[str] = set()
    seen_types: set[str] = set()
    deduped_output: list[dict] = []
    for t in output_candidates:
        tags = t.get("tags") if isinstance(t.get("tags"), list) else []
        text = t.get("text", "")
        bucket = extract_bucket(text, tags) or infer_bucket_from_text(text)
        t_type = (t.get("tweet_type") or "").strip().lower()
        if bucket and bucket in seen_buckets:
            continue
        if t_type and t_type in seen_types:
            continue
        if bucket:
            seen_buckets.add(bucket)
        if t_type:
            seen_types.add(t_type)
        deduped_output.append(t)

    if not deduped_output and output_candidates:
        deduped_output = [output_candidates[0]]

    payload = {"tweets": deduped_output}
    write_json(out_queue_path, {"queue": payload["tweets"]})
    pipeline_logger.info("Wrote post queue: %s", out_queue_path)

    history_path = f"{settings.out_dir}/history.jsonl"
    for t in payload["tweets"]:
        append_jsonl(history_path, t)
