from __future__ import annotations

import re

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
    if any(term.replace("#", "").strip() in tag_values for term in BRAND_TERMS if term.startswith("#")):
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
