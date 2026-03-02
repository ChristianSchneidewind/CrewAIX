from __future__ import annotations

from typing import Any

from crewx.embeddings import cosine_similarity
from crewx.rules import (
    BUCKET_HISTORY_MAX,
    BUCKET_HISTORY_WINDOW,
    DOCUMENT_PATTERNS,
    KEYWORD_HISTORY_LIMITS,
    KEYWORD_QUOTAS,
    MAX_TYPES_PER_BATCH,
    TIP_LANGUAGE,
    TOPIC_KEYWORDS,
    bucket_matches_text,
    contains_brand_or_cta,
    count_keyword_hits,
    count_recent_bucket_hits,
    extract_bucket,
    has_concrete_detail,
    has_hashtag,
    infer_bucket_from_text,
    infer_opening_style,
    is_allowed_bucket,
    is_doc_tip,
    violates_hard_rules,
)


def _coerce_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            tags.append(text)
    return tags


def normalize_candidate_fields(t: dict) -> dict:
    text = (t.get("text") or "").strip()
    if not text:
        return t

    if not (t.get("language") or "").strip():
        t["language"] = "de"

    if not (t.get("opening_style") or "").strip():
        t["opening_style"] = infer_opening_style(text)

    tags = _coerce_tags(t.get("tags"))
    bucket = infer_bucket_from_text(text)
    if bucket and bucket not in tags:
        tags.append(bucket)

    if t.get("opening_style") and len(tags) < 2 and t["opening_style"] not in tags:
        tags.append(t["opening_style"])

    t["tags"] = tags
    return t


def accept_relaxed_candidate(
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

    tags = _coerce_tags(t.get("tags"))

    if violates_hard_rules(text):
        return False
    if not has_concrete_detail(text):
        return False

    bucket = extract_bucket(text, tags)
    if not bucket:
        return False

    if has_hashtag(text) and tweet_type != "marketing":
        return False

    if contains_brand_or_cta(text, tags):
        return tweet_type == "marketing"

    return True


def assign_missing_types(tweets: list[dict], required_types: list[str]) -> list[dict]:
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


def filter_crewai_tweets(
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
    recent_scope = recent_texts[:50] if recent_texts else []
    recent_bucket_scope = recent_texts[:BUCKET_HISTORY_WINDOW] if recent_texts else []
    doc_tip_recent_hits = count_keyword_hits(recent_scope, DOCUMENT_PATTERNS)

    for t in tweets:
        text = (t.get("text") or "").strip()
        if not text:
            continue

        tweet_type = (t.get("tweet_type") or "").strip().lower()
        if allowed_types and tweet_type not in allowed_types:
            continue

        tags = _coerce_tags(t.get("tags"))
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

        if is_doc_tip(text):
            doc_tip_batch_hits = count_keyword_hits(
                [u.get("text", "") for u in filtered], DOCUMENT_PATTERNS
            )
            if doc_tip_batch_hits >= 1 or doc_tip_recent_hits >= 1:
                continue

        if violates_hard_rules(text):
            continue

        if not has_concrete_detail(text):
            continue

        bucket = extract_bucket(text, tags)
        if not bucket:
            continue
        if not is_allowed_bucket(bucket):
            continue
        if not bucket_matches_text(bucket, text):
            continue
        if bucket_counts.get(bucket, 0) >= 1:
            continue
        if count_recent_bucket_hits(recent_bucket_scope, bucket) >= BUCKET_HISTORY_MAX:
            continue

        if has_hashtag(text) and tweet_type != "marketing":
            continue

        if contains_brand_or_cta(text, tags):
            if tweet_type != "marketing":
                continue
            if brand_hits >= 1:
                continue
            brand_hits += 1

        keyword_blocked = False
        for key, quota in KEYWORD_QUOTAS.items():
            if not isinstance(quota, dict):
                continue
            needles = quota.get("needles")
            max_per_batch = quota.get("max_per_batch")
            if not isinstance(needles, list) or not isinstance(max_per_batch, int):
                continue
            needles = [str(n) for n in needles]
            if any(n in text.lower() for n in needles):
                batch_hits = count_keyword_hits([u.get("text", "") for u in filtered], needles)
                recent_hits = count_keyword_hits(recent_scope, needles)
                history_limit = KEYWORD_HISTORY_LIMITS.get(key)
                if history_limit is not None and not isinstance(history_limit, int):
                    history_limit = None
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
                if any(
                    cosine_similarity(candidate_embedding, emb) >= embedding_threshold
                    for emb in pool
                ):
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
        tags = _coerce_tags(t.get("tags"))
        bucket = extract_bucket(t.get("text", ""), tags)
        if not bucket or bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        unique.append(t)

    return unique
