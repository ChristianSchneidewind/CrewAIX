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
    """Build a focused prompt for generating varied German tweets for one type.

    The prompt is intentionally compact so that even smaller local models
    can follow the most important diversity rules.
    """

    # Keep history short to avoid giant prompts (and slow cloud responses)
    recent = list(history_texts)[-3:]
    recent_block = "\n".join(f"- {t}" for t in recent) if recent else "(none)"

    goal_block = tt.goal.strip() if tt.goal else "(no specific goal provided)"
    style_block = _bullets(tt.style) if tt.style else "(no style rules)"
    rules_block = _bullets(tt.rules) if tt.rules else "(no extra rules)"

    return dedent(
        f"""
        You write German tweets for the company described below.

        Output MUST be in CrewAI final answer format.
        Start with "Final Answer:" followed by a JSON array only.
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

        DIVERSITY RULES (VERY IMPORTANT):
        - Every tweet must feel clearly different from the others (nicht nur leicht umformuliert).
        - For each tweet, set an "opening_style" field with one of:
          "question", "myth_vs_fact", "tip", "scenario", "checklist", "mistake_fix", "condition".
        - As long as you generate <= 7 tweets, use every opening_style höchstens einmal.
        - Do NOT reuse exactly the same sentence template across tweets.
        - Avoid generic filler like: "Unsere Experten", "Jetzt starten", "Wir sind für dich da",
          "Check unsere Website", "Schnell und einfach", "Hol dir dein Geld", "Garantiert",
          "in wenigen Minuten".
        - Avoid repeating the same brand-slogan or call-to-action in multiple tweets.
          For example, these phrases may appear AT MOST ONCE per batch:
          "FlugNinja hilft Ihnen", "Mehr erfahren: https://www.flugninja.at/", "#FlugNinja".
        - Do NOT mention "EU-Abflughafen" or "EU-Abflughäfen".
        - Do NOT invent time thresholds like "ab 3 Stunden" (not in the brief).
        - Only mention "Österreich" or "ab/nach Österreich" when it is truly relevant to the tweet.
        - Across the batch, avoid repeating the same key fact or phrase (e.g., "bis zu 600 €",
          "kostenloser Anspruchs-Check", "ohne Vorleistung", "nur bei Erfolg"), unless you have no other option.
        - Do NOT reuse the same advice. In particular, avoid repeating documentation tips like
          "Flugnummer notieren", "geplante Ankunftszeit", "Verspätungsgrund dokumentieren",
          "Boardingpass/Buchungsbestätigung aufbewahren", or "Airline-Nachrichten sichern".
        - HARD BAN openings like "Wussten Sie", "Wissen Sie", "Haben Sie gewusst", "Wusstest du".
          HARD BAN "Checkliste"/"Schritte" formats.
          HARD BAN openings like "Tipp:", "Fehler:", "Mythos:", "Fakt:", "Irrtum:", "Falsch".
          Vary the openings.
        - Across the batch, mention "EU-261" or "EU-Verordnung 261/2004" at most ONCE.
          Mention "bis zu 600 €" at most ONCE.
        - Avoid focusing on Annullierung repeatedly; mix in Verspätung and Überbuchung too.
        - Ensure topic variety across the batch: mix scenarios (Verspätung, Annullierung, Überbuchung),
          process/steps, myths vs facts, and practical traveler tips.
        - Mention Anschlussflug/Zubringer nur höchstens EINMAL pro Batch.
        - Each tweet must include at least ONE concrete detail:
          a condition, a tip, a typical scenario, a limit ("bis zu …"),
          or a misconception + correction.
        - Tags: add 1–2 short tags that describe the angle (e.g. ["tip"], ["myth"], ["scenario"], ["checklist"]).

        RECENT TWEETS (do NOT repeat, do NOT paraphrase closely):
        {recent_block}

        HARD REQUIREMENTS (technical):
        - Return EXACTLY {n_candidates} tweet objects in JSON.
        - Output format MUST be a single JSON ARRAY of tweet objects. No wrapper object.
          Example:
          [
            {{
              "tweet_type": "{tt.name}",
              "opening_style": "question",
              "text": "...",
              "language": "de",
              "tags": []
            }}
          ]
        - Each tweet object MUST include:
          - "tweet_type": "{tt.name}"
          - "opening_style": one of
            ["question", "myth_vs_fact", "tip", "scenario", "checklist", "mistake_fix", "condition"]
          - "text": string (max 240 chars)
          - "language": "de"
          - "tags": array of short strings (may be empty)
        - No hashtag spam. Max 2 hashtags total if any.
        - Each tweet must contain at least ONE concrete useful detail, tip, or condition.

        IF YOU CANNOT FOLLOW THESE RULES, OUTPUT:
        Final Answer: []
        DO NOT WRITE ANY OTHER TEXT OUTSIDE OF "Final Answer:" + JSON.
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

