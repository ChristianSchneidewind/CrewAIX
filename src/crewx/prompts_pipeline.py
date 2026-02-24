from __future__ import annotations

import re
from textwrap import dedent

from crewx.parsing import TweetType
from crewx.rules import IDEA_BANK_MAX_ITEMS


def trim_idea_bank(ideas_md: str | None) -> str:
    if not ideas_md:
        return "(none)"
    lines = [line.strip() for line in ideas_md.splitlines()]
    bullets = [line for line in lines if line.startswith("- ")]
    if bullets:
        trimmed = bullets[:IDEA_BANK_MAX_ITEMS]
        return "\n".join(trimmed)
    return "\n".join(lines[:IDEA_BANK_MAX_ITEMS])


def trim_company_context(company_md: str) -> str:
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


def format_types_md(types: list[TweetType]) -> str:
    lines = ["# Tweet Types (compact)"]
    for tt in types:
        lines.append(f"## {tt.name}")
        if tt.goal:
            lines.append(f"Goal: {tt.goal}")
        lines.append("")
    return "\n".join(lines).strip()


def build_generator_prompt(
    *,
    company_md: str,
    types_md: str,
    ideas_md: str | None,
    n_tweets: int,
    recent: list[str],
    required_types: list[str] | None = None,
) -> str:
    recent_block = "\n".join(f"- {t}" for t in recent[-3:]) if recent else "(none)"
    ideas_block = trim_idea_bank(ideas_md)
    company_block = trim_company_context(company_md)
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


def build_review_prompt(*, n_tweets: int) -> str:
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


def build_quality_prompt(*, n_tweets: int) -> str:
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


def build_post_prompt() -> str:
    return dedent(
        """
        Prepare the approved tweets for posting.
        Do NOT change the text. Do NOT add new content.
        Output MUST be a JSON array of the tweets to be queued.
        """
    ).strip()
