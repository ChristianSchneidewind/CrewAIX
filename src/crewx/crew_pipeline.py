from __future__ import annotations

from textwrap import dedent
from pathlib import Path
import re
import json
import math
import time
from typing import Iterable

from litellm import embedding as litellm_embedding

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
    "annullierung": {
        "needles": ["annull", "gestrichen"],
        "max_per_batch": 1,
    },
    "verspaetung": {
        "needles": ["verspät", "verspaet"],
        "max_per_batch": 1,
    },
    "ueberbuchung": {
        "needles": ["überbuch", "ueberbuch", "boarding verweigert", "nicht mitfliegen"],
        "max_per_batch": 1,
    },
    "entschaedigung": {
        "needles": ["entschäd", "entschaed", "kompensation", "ausgleichszahlung"],
        "max_per_batch": 1,
    },
}

KEYWORD_HISTORY_LIMITS = {
    "gate_changes": 0,
    "connections": 1,
    "eu261": 0,
    "annullierung": 999,
    "verspaetung": 999,
    "ueberbuchung": 999,
    "entschaedigung": 999,
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

TOPIC_BUCKETS = {
    "boarding_gate": ["boarding", "gate", "einsteigen", "boarden", "boarding-gruppe", "boardinggruppe"],
    "gepaeck_handgepaeck": ["gepäck", "gepaeck", "handgepäck", "handgepaeck", "koffer", "gepäckband"],
    "gepaeckverlust": ["gepäckverlust", "gepaeckverlust", "gepäck verspätet", "gepaeck verspätet", "verloren", "beschädigt"],
    "checkin_sitzplatz": ["check-in", "checkin", "sitzplatz", "sitz", "boardingpass", "mobile boarding"],
    "anschlussflug": ["anschlussflug", "zubringer", "umsteig", "umsteigen", "connection", "transit"],
    "wetter_irrops": ["wetter", "sturm", "schnee", "gewitter", "nebel", "vereisung"],
    "streik": ["streik", "arbeitskampf", "gewerkschaft"],
    "codeshare": ["codeshare", "code-share", "allianz", "operating carrier", "durchführende airline"],
    "sicherheit": ["sicherheitskontrolle", "security", "flughafensicherheit", "kontrolle", "flüssigkeiten", "liquids"],
    "reiseruecktritt_kulanz": ["reiserücktritt", "reiseruecktritt", "storno", "umbuchung", "erstattung", "gutschein", "kulanz"],
    "vielflieger": ["vielflieger", "status", "meilen", "bonusprogramm", "frequent flyer"],
    "betreuung": ["betreuung", "verpflegung", "hotel", "transfer", "ersatzbeförderung"],
}

BUCKET_TAGS = set(TOPIC_BUCKETS.keys())

BUCKET_HISTORY_WINDOW = 15
BUCKET_HISTORY_MAX = 1
IDEA_BANK_MAX_ITEMS = 10
ACTIVE_BUCKETS = [
    "boarding_gate",
    "gepaeck_handgepaeck",
    "checkin_sitzplatz",
    "wetter_irrops",
    "streik",
]

BRAND_TERMS = [
    "flugninja",
    "flugninja.at",
    "https://www.flugninja.at/",
    "#flugninja",
]

CTA_TERMS = [
    "mehr infos",
    "mehr erfahren",
    "jetzt prüfen",
    "anspruch prüfen",
    "kostenlos prüfen",
    "kostenlose prüfung",
    "kostenloser anspruchs-check",
    "anspruchs-check",
    "anspruchscheck",
]

DETAIL_KEYWORDS = [
    "wenn",
    "falls",
    "sobald",
    "vor",
    "nach",
    "bei",
    "am gate",
    "am band",
    "sicherheitskontrolle",
    "check-in",
    "checkin",
    "boarding",
    "koffer",
    "handgepäck",
    "handgepaeck",
    "gepäck",
    "gepaeck",
    "schalter",
    "formular",
    "ersatz",
    "umbuch",
    "gutschein",
    "verpflegung",
    "hotel",
]

DETAIL_NUMBER_PATTERN = re.compile(r"\d")
HASHTAG_PATTERN = re.compile(r"#\w+")

URL_PATTERN = re.compile(r"https?://\S+")

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


def _extract_bucket(text: str, tags: list[str] | None) -> str | None:
    tags = tags or []
    bucket_tags = []
    for tag in tags:
        norm = (tag or "").strip().lower().lstrip("#")
        if norm in BUCKET_TAGS:
            bucket_tags.append(norm)
    if len(bucket_tags) != 1:
        return None
    return bucket_tags[0]


def _contains_brand_or_cta(text: str, tags: list[str] | None) -> bool:
    lower = (text or "").lower()
    tag_values = [str(t).strip().lower().lstrip("#") for t in (tags or []) if str(t).strip()]

    if URL_PATTERN.search(lower):
        return True
    if any(term in lower for term in BRAND_TERMS):
        return True
    if any(term in lower for term in CTA_TERMS):
        return True
    if any(term.replace("#", "").strip() in tag_values for term in BRAND_TERMS if term.startswith("#")):
        return True
    if "flugninja" in tag_values:
        return True
    return False


def _bucket_matches_text(bucket: str, text: str) -> bool:
    needles = TOPIC_BUCKETS.get(bucket, [])
    lower = (text or "").lower()
    return any(n in lower for n in needles)


def _infer_bucket_from_text(text: str) -> str | None:
    lower = (text or "").lower()
    for bucket, needles in TOPIC_BUCKETS.items():
        if any(n in lower for n in needles):
            return bucket
    return None


def _count_recent_bucket_hits(texts: list[str], bucket: str) -> int:
    hits = 0
    for t in texts:
        if _infer_bucket_from_text(t) == bucket:
            hits += 1
    return hits


def _is_allowed_bucket(bucket: str) -> bool:
    return bucket in ACTIVE_BUCKETS


def _trim_idea_bank(ideas_md: str | None) -> str:
    if not ideas_md:
        return "(none)"
    lines = [line.strip() for line in ideas_md.splitlines()]
    bullets = [line for line in lines if line.startswith("- ")]
    if bullets:
        trimmed = bullets[:IDEA_BANK_MAX_ITEMS]
        return "\n".join(trimmed)
    return "\n".join(lines[:IDEA_BANK_MAX_ITEMS])


def _trim_company_context(company_md: str) -> str:
    allowed_sections = {
        "company",
        "product / offer",
        "target audience",
        "tone & voice",
        "proof / facts (only use these)",
        "content pillars",
    }
    blocks = re.split(r"(?m)^##\s+", company_md)
    out: list[str] = []
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].strip().lower()
        if heading in allowed_sections:
            out.append("## " + lines[0].strip())
            out.extend(lines[1:])
            out.append("")
    return "\n".join(out).strip() or company_md.strip()


def _infer_opening_style(text: str) -> str:
    lower = (text or "").lower().strip()
    if "?" in lower:
        return "question"
    if lower.startswith(("wenn ", "falls ", "sobald ", "sofern ")):
        return "condition"
    if any(p in lower for p in ["ich ", "mein ", "meine ", "wir ", "uns ", "mich "]):
        return "scenario"
    return "tip"


def _normalize_candidate_fields(t: dict) -> dict:
    text = (t.get("text") or "").strip()
    if not text:
        return t

    if not (t.get("language") or "").strip():
        t["language"] = "de"

    if not (t.get("opening_style") or "").strip():
        t["opening_style"] = _infer_opening_style(text)

    tags = t.get("tags") if isinstance(t.get("tags"), list) else []
    bucket = _infer_bucket_from_text(text)
    if bucket and bucket not in tags:
        tags.append(bucket)

    if t.get("opening_style") and len(tags) < 2 and t["opening_style"] not in tags:
        tags.append(t["opening_style"])

    t["tags"] = tags
    return t


def _has_concrete_detail(text: str) -> bool:
    lower = (text or "").lower()
    if DETAIL_NUMBER_PATTERN.search(lower):
        return True
    return any(k in lower for k in DETAIL_KEYWORDS)


def _has_hashtag(text: str) -> bool:
    return bool(HASHTAG_PATTERN.search(text or ""))


def _accept_relaxed_candidate(
    t: dict,
    *,
    allowed_types: set[str] | None,
    type_limits: dict[str, int] | None,
) -> bool:
    text = (t.get("text") or "").strip()
    if not text:
        return False

    tweet_type = (t.get("tweet_type") or "").strip().lower()
    if allowed_types and tweet_type not in allowed_types:
        return False
    if type_limits and tweet_type in type_limits and type_limits[tweet_type] <= 0:
        return False

    tags = t.get("tags") if isinstance(t.get("tags"), list) else []

    if _violates_hard_rules(text):
        return False
    if not _has_concrete_detail(text):
        return False

    bucket = _extract_bucket(text, tags)
    if not bucket:
        return False

    if _has_hashtag(text) and tweet_type != "marketing":
        return False

    if _contains_brand_or_cta(text, tags):
        return tweet_type == "marketing"

    return True


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "rate_limit" in message or "429" in message


def _is_request_too_large(exc: Exception) -> bool:
    message = str(exc).lower()
    return "request too large" in message or "must be reduced" in message


def _is_connection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "connection error" in message or "no route to host" in message or "connecterror" in message


class _RateLimitHit(RuntimeError):
    pass


def _parse_retry_after_seconds(message: str) -> float | None:
    lower = (message or "").lower()
    match = re.search(r"try again in\s+([0-9.]+)\s*(ms|s)", lower)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ms":
        return value / 1000.0
    return value


def _kickoff_with_retry(
    crew: Crew,
    *,
    max_retries: int = 6,
    base_delay: float = 2.0,
    fail_fast_on_rate_limit: bool = False,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return str(crew.kickoff() or "")
        except Exception as exc:
            if _is_rate_limit_error(exc):
                if fail_fast_on_rate_limit:
                    raise _RateLimitHit(str(exc))
                if attempt < max_retries:
                    message = str(exc)
                    retry_after = _parse_retry_after_seconds(message)
                    delay = retry_after if retry_after is not None else base_delay * (attempt + 1)
                    time.sleep(delay + 0.25)
                    last_exc = exc
                    continue
            if _is_connection_error(exc) and attempt < max_retries:
                delay = min(60.0, base_delay * (attempt + 1) * 3)
                _append_text(
                    f"{Path.cwd()}/out/last_raw_output.txt",
                    f"CONNECTION ERROR\nattempt={attempt + 1}\ndelay={delay}s\nerror={exc}\n\n",
                )
                time.sleep(delay)
                last_exc = exc
                continue
            raise
    if last_exc:
        raise last_exc
    return ""


def _append_text(path: str, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _is_embedding_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "invalid_api_key" in message or "incorrect api key" in message or "401" in message


def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_texts(texts: list[str], settings) -> list[list[float]]:
    if not texts or not settings.embedding_model_name:
        raise ValueError("Embedding disabled or empty input")
    response = litellm_embedding(
        model=settings.embedding_model_name,
        input=texts,
        api_key=settings.embedding_api_key or settings.openai_api_key,
        base_url=settings.embedding_api_base or settings.openai_api_base,
    )

    if isinstance(response, dict) and response.get("error"):
        raise ValueError(f"Embedding error: {response.get('error')}")

    data = None
    if isinstance(response, dict) and isinstance(response.get("data"), list):
        data = response.get("data")
    elif isinstance(response, list):
        data = response
    elif isinstance(response, dict) and isinstance(response.get("embedding"), list):
        return [response.get("embedding")]
    else:
        response_data = getattr(response, "data", None)
        if isinstance(response_data, list):
            data = response_data
        elif hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict) and isinstance(dumped.get("data"), list):
                data = dumped.get("data")

    if not data:
        keys = list(response.keys()) if isinstance(response, dict) else type(response).__name__
        raise ValueError(f"No embedding data returned (response={keys})")

    embeddings: list[list[float]] = []
    for item in data:
        emb = item.get("embedding") if isinstance(item, dict) else None
        if isinstance(emb, list):
            embeddings.append(emb)
    if not embeddings:
        raise ValueError("No embeddings parsed")
    return embeddings


def _build_embedding_map(texts: list[str], settings) -> dict[str, list[float]]:
    embeddings = _embed_texts(texts, settings)
    mapping: dict[str, list[float]] = {}
    for text, emb in zip(texts, embeddings):
        if text and text not in mapping:
            mapping[text] = emb
    if not mapping:
        raise ValueError("Empty embedding map")
    return mapping


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
    embedding_threshold: float | None = None,
    recent_embeddings: list[list[float]] | None = None,
    candidate_embeddings: dict[str, list[float]] | None = None,
) -> list[dict]:
    filtered: list[dict] = []
    type_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    brand_hits = 0
    accepted_embeddings: list[list[float]] = []
    doc_tip_in_recent = any(_is_doc_tip(t) for t in recent_texts)
    recent_scope = recent_texts[:50] if recent_texts else []
    recent_bucket_scope = recent_texts[:BUCKET_HISTORY_WINDOW] if recent_texts else []
    doc_tip_recent_hits = _count_keyword_hits(recent_scope, DOCUMENT_PATTERNS)

    for t in tweets:
        text = (t.get("text") or "").strip()
        if not text:
            continue

        tweet_type = (t.get("tweet_type") or "").strip().lower()
        if allowed_types and tweet_type not in allowed_types:
            continue

        tags = t.get("tags") if isinstance(t.get("tags"), list) else []
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

        if not _has_concrete_detail(text):
            continue

        bucket = _extract_bucket(text, tags)
        if not bucket:
            continue
        if not _is_allowed_bucket(bucket):
            continue
        if not _bucket_matches_text(bucket, text):
            continue
        if bucket_counts.get(bucket, 0) >= 1:
            continue
        if _count_recent_bucket_hits(recent_bucket_scope, bucket) >= BUCKET_HISTORY_MAX:
            continue

        if _has_hashtag(text) and tweet_type != "marketing":
            continue

        if _contains_brand_or_cta(text, tags):
            if tweet_type != "marketing":
                continue
            if brand_hits >= 1:
                continue
            brand_hits += 1

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

        if embedding_threshold and candidate_embeddings:
            candidate_embedding = candidate_embeddings.get(text)
            if candidate_embedding:
                pool = (recent_embeddings or []) + accepted_embeddings
                if any(_cosine_similarity(candidate_embedding, emb) >= embedding_threshold for emb in pool):
                    continue
                accepted_embeddings.append(candidate_embedding)

        filtered.append(t)
        type_counts[tweet_type] += 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    if not filtered:
        return []

    unique: list[dict] = []
    seen_buckets: set[str] = set()
    for t in filtered:
        tags = t.get("tags") if isinstance(t.get("tags"), list) else []
        bucket = _extract_bucket(t.get("text", ""), tags)
        if not bucket or bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        unique.append(t)

    return unique


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


def _format_types_md(types: list[TweetType]) -> str:
    lines = ["# Tweet Types (compact)"]
    for tt in types:
        lines.append(f"## {tt.name}")
        if tt.goal:
            lines.append(f"Goal: {tt.goal}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_generator_prompt(
    *,
    company_md: str,
    types_md: str,
    ideas_md: str | None,
    n_tweets: int,
    recent: list[str],
    required_types: list[str] | None = None,
) -> str:
    recent_block = "\n".join(f"- {t}" for t in recent[-3:]) if recent else "(none)"
    ideas_block = _trim_idea_bank(ideas_md)
    company_block = _trim_company_context(company_md)
    return dedent(
        f"""
        Write German tweets for the company below.
        Output MUST be a JSON array only. Each item is an object with keys:
        tweet_type, opening_style, text, language, tags. No plain strings.

        COMPANY CONTEXT:
        ---
        {company_block}
        ---

        IDEA BANK (subset):
        ---
        {ideas_block}
        ---

        TWEET TYPES:
        ---
        {types_md}
        ---

        REQUIRED TYPES: {", ".join(required_types) if required_types else "(none)"}

        BUCKET TAGS (pick a DIFFERENT one per tweet and include it in tags):
        boarding_gate, gepaeck_handgepaeck, checkin_sitzplatz, wetter_irrops, streik

        DIVERSITY:
        - Exactly {n_tweets} tweets, varied angles and openings.
        - At least 2 different tweet_type values.
        - Max 1 travel_hack.
        - If REQUIRED TYPES given: exactly one per type.
        - New concrete detail per tweet; no repeated scenario/claim in batch.
        - Brand/CTA at most ONE tweet.
        - Avoid "Wussten Sie/Wissen Sie/Haben Sie gewusst", "Mythos/Fakt/Irrtum/Falsch", "Checkliste/Schritte".
        - travel_hack: avoid document-keeping tips.

        TAGS:
        - Exactly ONE bucket tag + optional ONE angle tag (tip/scenario/condition/etc.).

        X:
        - Max 240 chars, no hashtags unless marketing, emojis 0–2.

        RECENT (avoid repeats):
        {recent_block}

        Output:
        [{{"tweet_type":"...","opening_style":"question|tip|scenario|condition|mistake_fix|checklist|myth_vs_fact","text":"...","language":"de","tags":["..."]}}]
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

        Output MUST be a JSON array only. No strings.
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

        Output MUST be a JSON array only. No strings.
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

    ideas_md = None
    ideas_path = Path(settings.ideas_md_path)
    if ideas_path.exists():
        ideas_md = ideas_path.read_text(encoding="utf-8")

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

    poster_agent = Agent(
        role=poster_role.get("role") or "Tweet Poster",
        goal=poster_role.get("goal") or "Prepare final tweets for the posting queue without altering content.",
        backstory=poster_role.get("backstory") or "You only prepare a queue; you never call external APIs.",
        llm=llm,
        verbose=settings.verbose,
    )

    base_active_types = active_types
    base_required_types = [t.name.strip() for t in base_active_types]

    def _build_crews(*, recent_context: list[str], active_types: list[TweetType], n_tweets: int) -> tuple[Crew, Crew]:
        required_types = [t.name.strip() for t in active_types]
        effective_n_tweets = len(required_types) if forced_types else n_tweets
        types_md = _format_types_md(active_types)

        generate_task = Task(
            description=_build_generator_prompt(
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
            description=_build_review_prompt(n_tweets=effective_n_tweets),
            expected_output="A JSON array of compliant tweet objects.",
            agent=reviewer_agent,
            context=[generate_task],
        )

        post_task = Task(
            description=_build_post_prompt(),
            expected_output="A JSON array of tweets ready to queue.",
            agent=poster_agent,
            context=[review_task],
        )

        crew = Crew(
            agents=[generator_agent, reviewer_agent, poster_agent],
            tasks=[generate_task, review_task, post_task],
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

        return crew, review_only_crew

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
                    active_types = base_active_types[: max(1, min(n_tweets, len(base_active_types)))]

                effective_n_tweets = len(active_types) if forced_types else n_tweets
                _append_text(
                    last_raw_path,
                    f"RUN CONTEXT\nforce_minimal={force_minimal}\nrecent_context={len(recent_context)}\n"
                    f"n_tweets={effective_n_tweets}\nactive_types={','.join([t.name for t in active_types])}\n\n",
                )

                crew, review_only_crew = _build_crews(
                    recent_context=recent_context,
                    active_types=active_types,
                    n_tweets=effective_n_tweets,
                )

                for attempt in range(max_attempts):
                    try:
                        raw_str = _kickoff_with_retry(crew, fail_fast_on_rate_limit=False)
                    except _RateLimitHit as exc:
                        retry_after = _parse_retry_after_seconds(str(exc))
                        delay = max(retry_after or 0, 60.0)
                        time.sleep(delay + 0.25)
                        rate_limit_triggered = True
                        break
                    except Exception as exc:
                        if _is_request_too_large(exc):
                            break
                        raise

                    _append_text(last_raw_path, "RAW OUTPUT\n" + raw_str + "\n\n")

                    try:
                        default_type = active_types[0].name.strip() if active_types else None
                        data = parse_tweets_response(
                            raw_str,
                            n_tweets=effective_n_tweets,
                            default_tweet_type=default_type,
                        )
                    except ValueError:
                        # fallback to review-only flow if quality stage yields nothing
                        try:
                            raw_str = _kickoff_with_retry(review_only_crew, fail_fast_on_rate_limit=False)
                        except _RateLimitHit as exc:
                            retry_after = _parse_retry_after_seconds(str(exc))
                            delay = max(retry_after or 0, 60.0)
                            time.sleep(delay + 0.25)
                            rate_limit_triggered = True
                            break
                        except Exception as exc:
                            if _is_request_too_large(exc):
                                break
                            raise
                        _append_text(last_raw_path, "RAW OUTPUT\n" + raw_str + "\n\n")
                        try:
                            default_type = active_types[0].name.strip() if active_types else None
                            data = parse_tweets_response(
                                raw_str,
                                n_tweets=effective_n_tweets,
                                default_tweet_type=default_type,
                            )
                        except ValueError:
                            continue

                    required_types = [t.name.strip() for t in active_types]
                    data["tweets"] = _assign_missing_types(data["tweets"], required_types)
                    data["tweets"] = [_normalize_candidate_fields(t) for t in data["tweets"]]
                    data["tweets"] = [_normalize_candidate_fields(t) for t in data["tweets"]]

                    max_travel_hack = 1
                    allowed_types = {t.name.strip().lower() for t in active_types}
                    type_limits = {t.name.strip().lower(): 1 for t in active_types} if forced_types else None

                    embedding_threshold = (
                        settings.embedding_similarity_threshold
                        if settings.embedding_model_name and (settings.embedding_api_key or settings.openai_api_key)
                        else None
                    )
                    recent_for_embeddings = recent_context[: settings.embedding_history_max]
                    recent_embeddings = None
                    candidate_embeddings = None
                    candidate_texts = [(t.get("text") or "").strip() for t in data["tweets"] if (t.get("text") or "").strip()]
                    embedding_error = None
                    if embedding_threshold and not embedding_disabled:
                        try:
                            recent_embeddings = _embed_texts(recent_for_embeddings, settings)
                            candidate_embeddings = _build_embedding_map(candidate_texts, settings)
                        except Exception as exc:
                            embedding_error = str(exc)
                            if _is_embedding_auth_error(exc):
                                embedding_disabled = True
                                embedding_error += " (embedding disabled)"

                        _append_text(
                            last_raw_path,
                            f"EMBEDDING DEBUG\nrecent={len(recent_for_embeddings)} emb_recent={'yes' if recent_embeddings else 'no'}\n"
                            f"candidates={len(candidate_texts)} emb_candidates={'yes' if candidate_embeddings else 'no'}\n"
                            f"error={embedding_error}\n\n",
                        )

                    tweets = _filter_crewai_tweets(
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
                            if _accept_relaxed_candidate(t, allowed_types=allowed_types, type_limits=type_limits):
                                fallback = [t]
                                break
                        tweets = fallback

                    if tweets:
                        tweets = _filter_crewai_tweets(
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
                        break

                if tweets or rate_limit_triggered:
                    break

            if tweets or rate_limit_triggered:
                break

        if tweets:
            break
        if rate_limit_triggered and not force_minimal:
            force_minimal = True
            continue
        break

    if not tweets:
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
        bucket = _extract_bucket(text, tags) or _infer_bucket_from_text(text)
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

    history_path = f"{settings.out_dir}/history.jsonl"
    for t in payload["tweets"]:
        append_jsonl(history_path, t)
