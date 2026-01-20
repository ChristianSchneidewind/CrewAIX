from __future__ import annotations

from textwrap import dedent
from typing import Iterable, Sequence

from crewx.parsing import TweetType


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {x}" for x in items if (x or "").strip())


def tweet_task_prompt_for_type(
    md_context: str,
    tt: TweetType,
    n_candidates: int,
    history_texts: Sequence[str],
) -> str:
    """
    We intentionally ask the model to generate MORE tweets than needed,
    so we can filter near-duplicates without extra regeneration calls.
    """

    # Keep history short to avoid giant prompts (and Ollama timeouts)
    recent = list(history_texts)[-50:]
    recent_block = "\n".join(f"- {t}" for t in recent) if recent else "(none)"

    goal_block = tt.goal.strip() if tt.goal else "(no specific goal provided)"
    style_block = _bullets(tt.style) if tt.style else "(no style rules)"
    rules_block = _bullets(tt.rules) if tt.rules else "(no extra rules)"

    return dedent(
        f"""
        You write German tweets for the company described below.

        Output MUST be VALID JSON ONLY.
        No markdown. No explanation.

        COMPANY CONTEXT (Markdown):
        ---
        {md_context}
        ---

        ACTIVE TWEET TYPE: {tt.name}

        TYPE GOAL:
        {goal_block}

        STYLE GUIDELINES:
        {style_block}

        CONTENT RULES:
        {rules_block}

        PROMPT DIVERSITY (VERY IMPORTANT):
        - Every tweet must feel meaningfully different from the others (not just reworded).
        - Use a DIFFERENT opening style for each tweet (rotate through these):
          1) Question
          2) “Myth vs Fact”
          3) Micro-tip (imperative)
          4) Short scenario (“Stell dir vor…”)
          5) Checklist / steps (2–3 short items)
          6) Common mistake + fix
          7) Constraint/condition highlight (e.g., “bis zu …”, “wenn … dann …”)
        - Do NOT reuse the same sentence template across tweets.
          (Example of forbidden repetition: “Kostenloser X bei Y. Link.” in multiple tweets.)
        - Avoid generic filler / ad-speak like:
          "Unsere Experten", "Jetzt starten", "Wir sind für dich da", "Check unsere Website",
          "Schnell und einfach", "Hol dir dein Geld", "Garantiert", "in wenigen Minuten"
        - Across the batch, each of these phrases may appear MAX ONCE:
          "Kostenloser Anspruchs-Check", "Keine Vorleistung", "keine Gebühren ohne Erfolg"
        - Each tweet must include at least ONE concrete detail:
          a condition, a tip, a typical scenario, a limit (“bis zu …”), or a clear misconception + correction.
        - Tags: add 1–2 short tags that describe the angle (e.g. ["tip"], ["myth"], ["scenario"], ["checklist"]).

        RECENT TWEETS (do NOT repeat, do NOT paraphrase too closely):
        {recent_block}

        HARD REQUIREMENTS (technical):
        - Return EXACTLY {n_candidates} tweet objects in JSON.
        - Each tweet object MUST include:
          - "tweet_type": "{tt.name}"
          - "text": string (max 240 chars)
          - "language": "de"
          - "tags": array of short strings (may be empty)
        - No hashtag spam. Max 2 hashtags total if any.
        - Avoid repeating the same sentence pattern.
        - Each tweet must contain at least ONE concrete useful detail, tip, or condition.

        OUTPUT JSON SCHEMA:
        {{
          "tweets": [
            {{
              "tweet_type": "{tt.name}",
              "text": "...",
              "language": "de",
              "tags": []
            }}
          ]
        }}
        """
    ).strip()


def build_generation_prompt(
    *,
    company_md: str,
    tweet_type,
    recent_tweets: list[str],
    n_tweets: int,
) -> str:
    return tweet_task_prompt_for_type(
        md_context=company_md,
        tt=tweet_type,
        n_candidates=n_tweets,
        history_texts=recent_tweets,
    )

