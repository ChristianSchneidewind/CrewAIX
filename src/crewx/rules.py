from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_rules() -> dict[str, Any]:
    path = _project_root() / "config" / "rules.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Rules file must contain a mapping at the top level.")
    return data


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


_rules = _load_rules()

DOCUMENT_PATTERNS = _as_list(_rules.get("document_patterns"))
KEYWORD_QUOTAS = _as_dict(_rules.get("keyword_quotas"))
KEYWORD_HISTORY_LIMITS = _as_dict(_rules.get("keyword_history_limits"))
TIP_LANGUAGE = _as_list(_rules.get("tip_language"))
MAX_TYPES_PER_BATCH = _as_dict(_rules.get("max_types_per_batch"))
TOPIC_KEYWORDS = _as_list(_rules.get("topic_keywords"))
TOPIC_BUCKETS = _as_dict(_rules.get("topic_buckets"))

BUCKET_TAGS = set(TOPIC_BUCKETS.keys())

BUCKET_HISTORY_WINDOW = int(_rules.get("bucket_history_window", 15))
BUCKET_HISTORY_MAX = int(_rules.get("bucket_history_max", 1))
IDEA_BANK_MAX_ITEMS = int(_rules.get("idea_bank_max_items", 10))
ACTIVE_BUCKETS = _as_list(_rules.get("active_buckets"))

BRAND_TERMS = _as_list(_rules.get("brand_terms"))
CTA_TERMS = _as_list(_rules.get("cta_terms"))
DETAIL_KEYWORDS = _as_list(_rules.get("detail_keywords"))

DETAIL_NUMBER_PATTERN = re.compile(r"\d")
HASHTAG_PATTERN = re.compile(r"#\w+")
URL_PATTERN = re.compile(r"https?://\S+")

FORBIDDEN_CLAIM_PHRASES = _as_list(_rules.get("forbidden_claim_phrases"))


def is_doc_tip(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in DOCUMENT_PATTERNS)


def count_keyword_hits(texts: list[str], needles: list[str]) -> int:
    hits = 0
    for t in texts:
        lower = t.lower()
        if any(n in lower for n in needles):
            hits += 1
    return hits


def extract_bucket(text: str, tags: list[str] | None) -> str | None:
    tags = tags or []
    bucket_tags = []
    for tag in tags:
        norm = (tag or "").strip().lower().lstrip("#")
        if norm in BUCKET_TAGS:
            bucket_tags.append(norm)
    if len(bucket_tags) != 1:
        return None
    return bucket_tags[0]


def contains_brand_or_cta(text: str, tags: list[str] | None) -> bool:
    lower = (text or "").lower()
    tag_values = [str(t).strip().lower().lstrip("#") for t in (tags or []) if str(t).strip()]

    if URL_PATTERN.search(lower):
        return True
    if any(term in lower for term in BRAND_TERMS):
        return True
    if any(term in lower for term in CTA_TERMS):
        return True
    if any(
        term.replace("#", "").strip() in tag_values for term in BRAND_TERMS if term.startswith("#")
    ):
        return True
    if "flugninja" in tag_values:
        return True
    return False


def bucket_matches_text(bucket: str, text: str) -> bool:
    needles = TOPIC_BUCKETS.get(bucket, [])
    lower = (text or "").lower()
    return any(n in lower for n in needles)


def infer_bucket_from_text(text: str) -> str | None:
    lower = (text or "").lower()
    for bucket, needles in TOPIC_BUCKETS.items():
        if any(n in lower for n in needles):
            return bucket
    return None


def count_recent_bucket_hits(texts: list[str], bucket: str) -> int:
    hits = 0
    for t in texts:
        if infer_bucket_from_text(t) == bucket:
            hits += 1
    return hits


def is_allowed_bucket(bucket: str) -> bool:
    return bucket in ACTIVE_BUCKETS


def infer_opening_style(text: str) -> str:
    lower = (text or "").lower().strip()
    if "?" in lower:
        return "question"
    if lower.startswith(("wenn ", "falls ", "sobald ", "sofern ")):
        return "condition"
    if any(p in lower for p in ["ich ", "mein ", "meine ", "wir ", "uns ", "mich "]):
        return "scenario"
    return "tip"


def has_concrete_detail(text: str) -> bool:
    lower = (text or "").lower()
    if DETAIL_NUMBER_PATTERN.search(lower):
        return True
    return any(k in lower for k in DETAIL_KEYWORDS)


def has_hashtag(text: str) -> bool:
    return bool(HASHTAG_PATTERN.search(text or ""))


def violates_hard_rules(text: str, *, strict: bool = True) -> bool:
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

    if any(p in lower_text for p in _as_list(_rules.get("forbidden_legal_claims"))):
        return True

    if any(p in lower_text for p in _as_list(_rules.get("forbidden_compensation_claims"))):
        return True

    if any(p in lower_text for p in _as_list(_rules.get("forbidden_patterns"))):
        return True

    if strict:
        return False

    return False
