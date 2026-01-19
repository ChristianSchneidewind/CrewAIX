from __future__ import annotations

from typing import Iterable, List
from textwrap import dedent

from crewx.parsing import TweetType


def _render_tweet_types(tweet_types: List[TweetType]) -> str:
    # Compact list, model-friendly
    lines = []
    for tt in tweet_types:
        # allow empty description
        desc = (tt.description or "").strip()
        if desc:
            lines.append(f"- {tt.name}: {desc}")
        else:
            lines.append(f"- {tt.name}")
    return "\n".join(lines)


def tweet_task_prompt(*, md: str, tweet_types: List[TweetType], n_tweets: int) -> str:
    types_block = _render_tweet_types(tweet_types)

    return dedent(
        f"""
        You are a copywriter for a company. Generate exactly {n_tweets} tweets for X (Twitter) based ONLY on the provided markdown content.
        The tweets must be varied, useful, and not repetitive.

        Allowed tweet types (pick EXACTLY ONE per tweet from the list):
        {types_block}

        Rules:
        - Output MUST be valid JSON only (no markdown fences, no commentary).
        - JSON schema:
          {{
            "tweets": [
              {{
                "tweet_type": "<one of the allowed types>",
                "text": "<tweet text>"
              }}
            ]
          }}
        - Exactly {n_tweets} items in "tweets".
        - Keep each tweet under 280 characters.
        - No duplicated ideas, hooks, or sentence structures.
        - If you mention links, keep them generic (no tracking params).
        - Write in German, tone: locker.

        Markdown input:
        ---
        {md}
        ---
        """
    ).strip()


def repair_json_prompt(*, raw: str, n_tweets: int, tweet_types: List[TweetType]) -> str:
    types_block = _render_tweet_types(tweet_types)

    return dedent(
        f"""
        Fix the following output into VALID JSON ONLY.

        Requirements:
        - Output ONLY JSON (no markdown fences).
        - Must match schema:
          {{
            "tweets": [
              {{
                "tweet_type": "<one of the allowed types>",
                "text": "<tweet text>"
              }}
            ]
          }}
        - Exactly {n_tweets} tweets.
        - tweet_type must be one of:
        {types_block}

        Raw output to fix:
        ---
        {raw}
        ---
        """
    ).strip()


def regen_missing_prompt(
    *,
    md: str,
    tweet_types: List[TweetType],
    n_missing: int,
    existing_texts: Iterable[str],
) -> str:
    types_block = _render_tweet_types(tweet_types)
    existing_block = "\n".join([f"- {t}" for t in existing_texts])

    return dedent(
        f"""
        Generate exactly {n_missing} ADDITIONAL tweets for X (Twitter).
        Use ONLY the markdown input.
        Avoid near-duplicates of the existing tweets.

        Allowed tweet types (pick EXACTLY ONE per tweet):
        {types_block}

        Existing tweets (do NOT repeat or paraphrase these):
        {existing_block}

        Output MUST be valid JSON only:
        {{
          "tweets": [
            {{
              "tweet_type": "<one of the allowed types>",
              "text": "<tweet text>"
            }}
          ]
        }}

        Markdown input:
        ---
        {md}
        ---
        """
    ).strip()
