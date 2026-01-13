import textwrap


def tweet_task_prompt(md: str, n_tweets: int) -> str:
    return textwrap.dedent(f"""
Read the following markdown brief and generate exactly {n_tweets} tweets for X.

Return ONLY a JSON object. No prose. No markdown. No code fences.
The first character of your response must be {{ and the last character must be }}.

JSON schema:
{{
  "tweets": [
    {{
      "text": "string",
      "language": "de|en",
      "tags": ["string", "..."],
      "intent": "announcement|tip|thread_hook|question|hot_take|cta|other"
    }}
  ]
}}

Rules:
- Each tweet text <= 240 characters
- 0–2 emojis per tweet
- Max 2 hashtags per tweet
- Avoid corporate/marketing tone
- Use info ONLY from the markdown brief (no hallucinated facts)
- Vary style across tweets (question, tip, short hot take, CTA, etc.)
- Do NOT repeat the same call-to-action wording across tweets
- Prefer concrete situations over generic phrases

Markdown brief:
----
{md}
----
""").strip()


def repair_prompt(raw_output: str, n_tweets: int) -> str:
    return textwrap.dedent(f"""
Convert the following text into VALID JSON ONLY.

Rules:
- Output ONLY JSON (no prose, no markdown, no code fences)
- First char must be {{ and last char must be }}
- Must match schema exactly
- If you cannot find {n_tweets} tweets, return as many as you can (at least 1 if possible)

Schema:
{{
  "tweets": [
    {{
      "text": "string",
      "language": "de|en",
      "tags": ["string", "..."],
      "intent": "announcement|tip|thread_hook|question|hot_take|cta|other"
    }}
  ]
}}

Text to convert:
----
{raw_output}
----
""").strip()
