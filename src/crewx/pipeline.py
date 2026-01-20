from __future__ import annotations

from crewai import Crew, Task

from crewx.config import apply_litellm_env, load_settings
from crewx.io import ensure_dir, list_recent_tweet_texts, now_timestamp, read_text, write_json, write_text
from crewx.llm import build_agent
from crewx.parsing import parse_tweet_types_md, parse_tweets_response
from crewx.prompts import build_generation_prompt


def _pick_active_types(all_types, active_names: list[str] | None) -> list:
    if not all_types:
        return []
    if not active_names:
        return all_types
    wanted = {n.strip().lower() for n in active_names if n.strip()}
    picked = [t for t in all_types if t.name.strip().lower() in wanted]
    return picked or all_types


def run_generate_tweets() -> None:
    apply_litellm_env()
    settings = load_settings()
    ensure_dir(settings.out_dir)

    company_md = read_text(settings.tweets_md_path)
    types_md = read_text(settings.tweet_types_md_path)

    all_types = parse_tweet_types_md(types_md)
    if not all_types:
        raise RuntimeError(f"No tweet types found in {settings.tweet_types_md_path}")

    # For now: use all types in file order (you can later add ACTIVE_TWEET_TYPES env)
    active_types = all_types

    # Use last N tweets from previous outputs to reduce repeats
    recent = list_recent_tweet_texts(settings.out_dir, limit=settings.recent_tweets_max)

    agent = build_agent(settings)

    # Round-robin generation across types until we have n_tweets
    remaining = settings.n_tweets
    all_tweets: list[dict] = []

    # always store last raw (debug)
    last_raw_path = f"{settings.out_dir}/last_raw_output.txt"

    type_idx = 0
    while remaining > 0:
        tt = active_types[type_idx % len(active_types)]
        type_idx += 1

        # Generate in small batches per type to keep the model responsive
        batch_n = min(remaining, 6)

        prompt = build_generation_prompt(
            company_md=company_md,
            tweet_type=tt,
            recent_tweets=recent,
            n_tweets=batch_n,
        )

        task = Task(
            description=prompt,
            agent=agent,  # IMPORTANT: fixes "Agent is missing in the task"
            expected_output="VALID JSON ONLY.",
        )

        raw = Crew(agents=[agent], tasks=[task]).kickoff()
        raw_str = str(raw or "")

        # always write debug raw
        write_text(last_raw_path, raw_str)

        data = parse_tweets_response(raw_str, n_tweets=batch_n)
        tweets = data["tweets"]

        # Update recent list (so the next batch avoids repeating the new ones too)
        for t in tweets:
            txt = (t.get("text") or "").strip()
            if txt:
                recent.insert(0, txt)
        recent = recent[: settings.recent_tweets_max]

        all_tweets.extend(tweets)
        remaining = settings.n_tweets - len(all_tweets)

        if len(all_tweets) >= settings.n_tweets:
            break

    # --- Write outputs ---
    timestamp = now_timestamp()
    out_json_path = f"{settings.out_dir}/tweets_{timestamp}.json"
    out_md_path = f"{settings.out_dir}/tweets_{timestamp}.md"

    payload = {"tweets": all_tweets[: settings.n_tweets]}
    write_json(out_json_path, payload)

    # Simple md rendering
    md_lines = []
    for t in payload["tweets"]:
        md_lines.append(f"- [{t.get('tweet_type','')}] {t.get('text','').strip()}")
    write_text(out_md_path, "\n".join(md_lines).strip() + "\n")
