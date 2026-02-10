from __future__ import annotations

from textwrap import dedent
from pathlib import Path
import re
import json

from crewai import Agent, Crew, Process, Task

from crewx.config import apply_litellm_env, load_settings
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
from crewx.parsing import parse_tweet_types_md, parse_tweets_response

DOCUMENT_PATTERNS = [
    "boardingpass",
    "buchungsbestätigung",
    "buchungsdetails",
    "airline-nachricht",
    "airline-nachrichten",
    "nachrichten der airline",
    "flugunterlagen",
    "reisedokumente",
    "e-mail",
    "e-mails",
    "email",
    "e mail",
    "sms",
]

KEYWORD_QUOTAS = {
    "gate_changes": {
        "needles": ["gate-änder", "gate ändern", "gate-wechsel", "anzeigetafel", "anzeigen"],
        "max_per_batch": 1,
    },
    "connections": {
        "needles": ["umsteig", "umsteigezeit", "umsteigezeiten", "anschlussflug", "zubringer", "puffer"],
        "max_per_batch": 1,
    },
    "eu261": {
        "needles": ["eu-verordnung 261/2004", "verordnung 261/2004", "eu261", "eu-261"],
        "max_per_batch": 1,
    },
}

KEYWORD_HISTORY_LIMITS = {
    "gate_changes": 0,
    "connections": 1,
    "eu261": 0,
}

TIP_LANGUAGE = [
    "tipp",
    "empfiehlt",
    "hilft",
    "vermeiden",
    "behalten sie",
    "prüfen sie",
    "planen sie",
    "sollten",
    "anspruchs-check",
    "anspruchscheck",
    "lohnt sich",
    "wichtig ist",
    "sofort",
    "am besten",
    "trinken sie",
    "packen sie",
    "bleiben sie",
    "achten sie",
]

MAX_TYPES_PER_BATCH = {
    "travel_hack": 1,
    "passenger_rights_quick": 1,
}

TOPIC_KEYWORDS = [
    "flug",
    "airline",
    "boarding",
    "gate",
    "flughafen",
    "passagier",
    "flugninja",
    "flugversp",
    "annull",
    "überbuch",
]

FORBIDDEN_CLAIM_PHRASES = [
    "abflugort in der eu",
    "in der eu startet",
    "in der eu startet oder landet",
    "startet oder landet",
    "start oder landung in der eu",
    "start oder landung",
    "abflug oder ankunft in österreich",
    "abflug oder ankunft",
    "ab start",
    "ab landung",
    "ab start oder landung",
    "eu-airline",
    "eu airline",
    "eu-airlines",
    "eu airlines",
    "österreich",
    "überbuchungen sind",
    "überbuchung ist üblich",
    "überbuchungen sind üblich",
    "außergewöhnlich",
    "außergewöhnliche umstände",
    "absichtlich überbuch",
    "um leerplätze zu",
]


def _is_doc_tip(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in DOCUMENT_PATTERNS)


def _count_keyword_hits(texts: list[str], needles: list[str]) -> int:
    hits = 0
    for t in texts:
        lower = t.lower()
        if any(n in lower for n in needles):
            hits += 1
    return hits


def _violates_hard_rules(text: str, *, strict: bool = True) -> bool:
    lower_text = (text or "").lower()

    if any(p in lower_text for p in FORBIDDEN_CLAIM_PHRASES):
        return True

    if "3 stunden" in lower_text or "3h" in lower_text:
        return True

    if "3\u00a0stunden" in lower_text:
        return True

    if "mehr als 3" in lower_text or "über 3" in lower_text or "ab 3" in lower_text:
        return True

    if "drei stunden" in lower_text:
        return True

    if "eu-verordnung 261/2004" in lower_text or "verordnung 261/2004" in lower_text or "eu261" in lower_text or "eu-261" in lower_text:
        return True

    if "bis zu 600" in lower_text or "600 €" in lower_text or "600€" in lower_text:
        return True

    if "gate-änder" in lower_text or "gate ändern" in lower_text or "gate-wechsel" in lower_text:
        return True
    if "anzeigetafel" in lower_text or "anzeigen" in lower_text:
        return True
    if "e-mail" in lower_text or "email" in lower_text or "sms" in lower_text:
        return True
    if "buchungsbestätigung" in lower_text or "buchungsdetails" in lower_text or "nachrichten der airline" in lower_text:
        return True
    if "umsteigezeit" in lower_text or "umsteigezeiten" in lower_text or "umsteig" in lower_text:
        return True
    if "anschlussflug" in lower_text or "pufferzeit" in lower_text:
        return True
    if "checkliste" in lower_text or "drei tipps" in lower_text:
        return True

    if strict:
        return False

    return False


def _assign_missing_types(tweets: list[dict], required_types: list[str]) -> list[dict]:
    required_queue = [t.strip().lower() for t in required_types if t.strip()]
    if not required_queue:
        return tweets
    used = {((t.get("tweet_type") or "").strip().lower()) for t in tweets}
    remaining = [t for t in required_queue if t and t not in used]

    for t in tweets:
        tweet_type = (t.get("tweet_type") or "").strip().lower()
        if not tweet_type or tweet_type == "unknown":
            if remaining:
                t["tweet_type"] = remaining.pop(0)
            else:
                t["tweet_type"] = required_queue[0]
    return tweets


def _filter_crewai_tweets(
    tweets: list[dict],
    recent_texts: list[str],
    *,
    max_travel_hack: int,
    allowed_types: set[str] | None = None,
    type_limits: dict[str, int] | None = None,
) -> list[dict]:
    filtered: list[dict] = []
    type_counts: dict[str, int] = {}
    doc_tip_in_recent = any(_is_doc_tip(t) for t in recent_texts)
    recent_scope = recent_texts[:50] if recent_texts else []
    doc_tip_recent_hits = _count_keyword_hits(recent_scope, DOCUMENT_PATTERNS)

    for t in tweets:
        text = (t.get("text") or "").strip()
        if not text:
            continue

        tweet_type = (t.get("tweet_type") or "").strip().lower()
        if allowed_types and tweet_type not in allowed_types:
            continue

        lower_text = text.lower()
        if not any(k in lower_text for k in TOPIC_KEYWORDS):
            continue

        if tweet_type == "industry_insight":
            if any(p in lower_text for p in TIP_LANGUAGE):
                continue

        if tweet_type == "fun_fact":
            if any(p in lower_text for p in TIP_LANGUAGE):
                continue

        type_counts.setdefault(tweet_type, 0)

        effective_limits = type_limits or MAX_TYPES_PER_BATCH
        max_for_type = effective_limits.get(tweet_type)
        if max_for_type is not None and type_counts[tweet_type] >= max_for_type:
            continue

        if tweet_type == "travel_hack":
            if type_counts[tweet_type] >= max_travel_hack:
                continue

        if _is_doc_tip(text):
            doc_tip_batch_hits = _count_keyword_hits([u.get("text", "") for u in filtered], DOCUMENT_PATTERNS)
            if doc_tip_batch_hits >= 1 or doc_tip_recent_hits >= 1:
                continue

        if _violates_hard_rules(text):
            continue

        keyword_blocked = False
        for key, quota in KEYWORD_QUOTAS.items():
            needles = quota["needles"]
            max_per_batch = quota["max_per_batch"]
            if any(n in text.lower() for n in needles):
                batch_hits = _count_keyword_hits([u.get("text", "") for u in filtered], needles)
                recent_hits = _count_keyword_hits(recent_scope, needles)
                history_limit = KEYWORD_HISTORY_LIMITS.get(key)
                if batch_hits >= max_per_batch:
                    keyword_blocked = True
                    break
                if history_limit is not None and recent_hits >= history_limit:
                    keyword_blocked = True
                    break
                if history_limit is None and recent_hits >= max_per_batch:
                    keyword_blocked = True
                    break
        if keyword_blocked:
            continue

        filtered.append(t)
        type_counts[tweet_type] += 1

    return filtered


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


def _build_generator_prompt(
    *,
    company_md: str,
    types_md: str,
    n_tweets: int,
    recent: list[str],
    required_types: list[str] | None = None,
) -> str:
    recent_block = "\n".join(f"- {t}" for t in recent[-5:]) if recent else "(none)"
    return dedent(
        f"""
        You write German tweets for the company described below.
        Output MUST be a JSON array only (no markdown, no explanation).

        COMPANY CONTEXT (Markdown):
        ---
        {company_md}
        ---

        TWEET TYPES (Markdown):
        ---
        {types_md}
        ---

        REQUIRED TYPES:
        {", ".join(required_types) if required_types else "(none)"}

        DIVERSITY RULES:
        - Produce exactly {n_tweets} tweets with varied angles and different openings.
        - Use at least 3 different tweet_type values in the batch.
        - Max 1 travel_hack per batch.
        - If a list of required tweet types is provided, generate EXACTLY one tweet per type.
        - Avoid repetitive openings like "Wussten Sie/Wissen Sie/Haben Sie gewusst".
        - Avoid "Mythos/Fakt/Irrtum/Falsch" openings.
        - Avoid "Checkliste/Schritte" wording.
        - Avoid repeating the same scenario (e.g., Anschlussflug/Zubringer) more than once.
        - If you generate travel_hack tweets, avoid document-keeping tips (Boardingpass/Buchungsbestätigung/Airline-Nachrichten).
          Use other angles like security prep, gate-change checks, carry-on limits, connection buffer, seat choice, hydration.

        X CONSTRAINTS:
        - Max 240 characters per tweet.
        - Max 2 hashtags total per tweet.
        - Emojis 0–2.

        RECENT TWEETS (do not repeat):
        {recent_block}

        Output format:
        [
          {{"tweet_type": "...", "opening_style": "question|tip|scenario|condition|mistake_fix|checklist|myth_vs_fact", "text": "...", "language": "de", "tags": ["..."]}}
        ]
        """
    ).strip()


def _build_review_prompt(*, n_tweets: int) -> str:
    return dedent(
        f"""
        You are a strict X (Twitter) compliance reviewer.
        Check the tweets provided in the context and fix or remove any tweet that violates these rules:
        - Max 240 characters
        - Max 2 hashtags
        - No "Wussten Sie/Wissen Sie/Haben Sie gewusst/Wusstest du" openings
        - No "Mythos/Fakt/Irrtum/Falsch" openings
        - No "Checkliste/Schritte" wording
        - No legal advice or guarantees
        - HARD BAN: 3-hour thresholds ("3 Stunden", "über 3", "ab 3", "3h")
        - HARD BAN: EU start/landing claims ("ab Start/Landung in der EU", "EU-Airlines")
        - Avoid repeated EU-261 mentions across the batch

        If a tweet violates the rules, rewrite it to comply while keeping the meaning.
        If it cannot be fixed, remove it.

        Output MUST be a JSON array only.
        Return up to {n_tweets} tweets in the SAME OBJECT FORMAT:
        [
          {{"tweet_type": "...", "opening_style": "question|tip|scenario|condition|mistake_fix|checklist|myth_vs_fact", "text": "...", "language": "de", "tags": ["..."]}}
        ]
        Always preserve or set the correct tweet_type for each item.
        If the input is a list of strings, convert each string into this object format
        and assign a suitable tweet_type from the REQUIRED TYPES list.
        """
    ).strip()


def _build_quality_prompt(*, n_tweets: int) -> str:
    return dedent(
        f"""
        You are a quality reviewer for German tweets.
        Keep only tweets that are clear, concrete, and non-redundant.

        Quality rules:
        - Remove generic or vague statements.
        - Remove marketing fluff or empty phrases.
        - Prefer specific details or situations.
        - Remove near-duplicates or overly similar tweets.
        - Remove tweets dominated by calls to action ("prüfen lassen", "kostenlose Prüfung", "lohnt sich").
        - Remove tip-style tweets only if they are vague or repetitive.
        - For fun_fact: no imperatives or CTA (e.g., "Ruhe bewahren", "Geduld haben", "achten Sie").
        - Do NOT add new facts.
        - Do NOT change tweet_type unless required.

        Output MUST be a JSON array only.
        Return up to {n_tweets} tweets in the SAME OBJECT FORMAT:
        [
          {{"tweet_type": "...", "opening_style": "question|tip|scenario|condition|mistake_fix|checklist|myth_vs_fact", "text": "...", "language": "de", "tags": ["..."]}}
        ]
        """
    ).strip()


def _build_post_prompt() -> str:
    return dedent(
        """
        Prepare the approved tweets for posting.
        Do NOT change the text. Do NOT add new content.
        Output MUST be a JSON array of the tweets to be queued.
        """
    ).strip()


def run_generate_tweets_crewai() -> None:
    apply_litellm_env()
    settings = load_settings()
    ensure_dir(settings.out_dir)

    company_md = read_text(settings.tweets_md_path)
    types_md = read_text(settings.tweet_types_md_path)
    roles = _load_roles(settings.crew_roles_md_path)

    all_types = parse_tweet_types_md(types_md)
    if not all_types:
        raise RuntimeError(f"No tweet types found in {settings.tweet_types_md_path}")

    forced_types = [t.strip().lower() for t in settings.forced_tweet_types if t.strip()]
    if forced_types:
        active_types = [t for t in all_types if t.name.strip().lower() in set(forced_types)]
    else:
        # Deterministic rotation based on history length (stable across runs).
        max_types = min(settings.n_tweets, len(all_types))
        start_idx = _rotation_start_index(settings.out_dir, total_types=len(all_types))
        active_types = [all_types[(start_idx + i) % len(all_types)] for i in range(max_types)]
        forced_types = [t.name.strip().lower() for t in active_types]

    fix_history_unknown_types(settings.out_dir, fallback_type="educational")
    recent = list_recent_tweet_texts(settings.out_dir, limit=settings.recent_tweets_max)

    llm = build_llm(settings)

    generator_role = roles.get("generator", {})
    reviewer_role = roles.get("reviewer", {})
    quality_role = roles.get("quality_reviewer", {})
    poster_role = roles.get("poster", {})

    generator_agent = Agent(
        role=generator_role.get("role") or "Tweet Generator",
        goal=generator_role.get("goal") or "Generate varied German tweets that follow the provided constraints.",
        backstory=generator_role.get("backstory") or "You are an expert social media writer for travel and passenger rights.",
        llm=llm,
        verbose=settings.verbose,
    )

    reviewer_agent = Agent(
        role=reviewer_role.get("role") or "X Compliance Reviewer",
        goal=reviewer_role.get("goal") or "Ensure tweets comply with X constraints and style rules.",
        backstory=reviewer_role.get("backstory") or "You are a strict reviewer who fixes or removes non-compliant tweets.",
        llm=llm,
        verbose=settings.verbose,
    )

    quality_agent = Agent(
        role=quality_role.get("role") or "Quality Reviewer",
        goal=quality_role.get("goal") or "Filter out low-quality tweets and keep only clear, concrete outputs.",
        backstory=quality_role.get("backstory") or "You are a senior editor focused on clarity and usefulness.",
        llm=llm,
        verbose=settings.verbose,
    )

    poster_agent = Agent(
        role=poster_role.get("role") or "Tweet Poster",
        goal=poster_role.get("goal") or "Prepare final tweets for the posting queue without altering content.",
        backstory=poster_role.get("backstory") or "You only prepare a queue; you never call external APIs.",
        llm=llm,
        verbose=settings.verbose,
    )

    required_types = [t.name.strip() for t in active_types]
    effective_n_tweets = len(required_types) if forced_types else settings.n_tweets

    generate_task = Task(
        description=_build_generator_prompt(
            company_md=company_md,
            types_md=types_md,
            n_tweets=effective_n_tweets,
            recent=recent,
            required_types=required_types if forced_types else None,
        ),
        expected_output="A JSON array of tweet objects.",
        agent=generator_agent,
    )

    review_task = Task(
        description=_build_review_prompt(n_tweets=effective_n_tweets),
        expected_output="A JSON array of compliant tweet objects.",
        agent=reviewer_agent,
        context=[generate_task],
    )

    quality_task = Task(
        description=_build_quality_prompt(n_tweets=effective_n_tweets),
        expected_output="A JSON array of high-quality tweet objects.",
        agent=quality_agent,
        context=[review_task],
    )

    post_task = Task(
        description=_build_post_prompt(),
        expected_output="A JSON array of tweets ready to queue.",
        agent=poster_agent,
        context=[quality_task],
    )

    crew = Crew(
        agents=[generator_agent, reviewer_agent, quality_agent, poster_agent],
        tasks=[generate_task, review_task, quality_task, post_task],
        process=Process.sequential,
        verbose=settings.verbose,
    )

    review_only_task = Task(
        description=_build_post_prompt(),
        expected_output="A JSON array of tweets ready to queue.",
        agent=poster_agent,
        context=[review_task],
    )

    review_only_crew = Crew(
        agents=[generator_agent, reviewer_agent, poster_agent],
        tasks=[generate_task, review_task, review_only_task],
        process=Process.sequential,
        verbose=settings.verbose,
    )

    last_raw_path = f"{settings.out_dir}/last_raw_output.txt"
    tweets: list[dict] = []
    max_attempts = 3

    for attempt in range(max_attempts):
        raw_result = crew.kickoff()
        raw_str = str(raw_result or "")
        write_text(last_raw_path, raw_str)

        try:
            data = parse_tweets_response(raw_str, n_tweets=effective_n_tweets)
        except ValueError:
            # fallback to review-only flow if quality stage yields nothing
            raw_result = review_only_crew.kickoff()
            raw_str = str(raw_result or "")
            write_text(last_raw_path, raw_str)
            try:
                data = parse_tweets_response(raw_str, n_tweets=effective_n_tweets)
            except ValueError:
                continue

        required_types = [t.name.strip() for t in active_types]
        data["tweets"] = _assign_missing_types(data["tweets"], required_types)

        max_travel_hack = max(1, effective_n_tweets // 5)
        allowed_types = {t.name.strip().lower() for t in active_types}
        type_limits = {t.name.strip().lower(): 1 for t in active_types} if forced_types else None
        tweets = _filter_crewai_tweets(
            data["tweets"],
            recent,
            max_travel_hack=max_travel_hack,
            allowed_types=allowed_types,
            type_limits=type_limits,
        )
        if not tweets:
            fallback = []
            for t in data["tweets"]:
                text = (t.get("text") or "").strip()
                if not text:
                    continue
                if attempt < max_attempts - 1 and _violates_hard_rules(text):
                    continue
                if attempt == max_attempts - 1 and _violates_hard_rules(text, strict=False):
                    continue
                fallback = [t]
                break
            tweets = fallback

        if tweets:
            break

    if not tweets:
        return

    timestamp = now_timestamp()
    out_queue_path = f"{settings.out_dir}/post_queue_{timestamp}.json"

    payload = {"tweets": tweets[:effective_n_tweets]}
    write_json(out_queue_path, {"queue": payload["tweets"]})

    history_path = f"{settings.out_dir}/history.jsonl"
    for t in payload["tweets"]:
        append_jsonl(history_path, t)
