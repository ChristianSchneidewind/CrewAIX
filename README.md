# CrewAiX

Generate German tweets for **FlugNinja** with a lightweight CrewAI pipeline. The project focuses on high-variance, bucketed tweet generation with strict format rules and dedupe logic.

## Requirements

- Python 3.11+
- `uv` (recommended)
- OpenAI API key (cloud only)

## Quick Start

```bash
# install deps
uv sync

# set env
cp .env.example .env  # or create your own .env

# run
uv run python src/main.py
```

## Company Inputs

Update company-specific inputs in:

- `content/tweets.md` (company context, tone, constraints)
- `content/tweet_types.md` (tweet type definitions)
- `content/ideas.md` (idea bank)

## Configuration (.env)

Minimal setup:

```env
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4.1

TWEETS_MD_PATH=content/tweets.md
TWEET_TYPES_MD_PATH=content/tweet_types.md
IDEAS_MD_PATH=content/ideas.md
OUT_DIR=out

N_TWEETS=3
RECENT_TWEETS_MAX=5

TEMPERATURE=0.7
VERBOSE=false
```

Optional embedding dedupe (recommended if available):

```env
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_API_KEY=sk-...
EMBEDDING_SIMILARITY_THRESHOLD=0.85
EMBEDDING_HISTORY_MAX=30
```

## How It Works

- **Generator → Reviewer → Poster** (quality stage removed for lower token usage).
- **Buckets**: each tweet must include exactly one bucket tag.
- **Dedupe**:
  - one tweet per bucket in a batch
  - bucket cannot repeat within the last `BUCKET_HISTORY_WINDOW`
  - optional embedding-based similarity filter
- **Output**:
  - queued tweets saved to `out/post_queue_*.json`
  - history appended to `out/history.jsonl`

## Bucket Set (active)

Currently restricted to 5 buckets for lower prompt size:

- `boarding_gate`
- `gepaeck_handgepaeck`
- `checkin_sitzplatz`
- `wetter_irrops`
- `streik`

## Idea Bank

The prompt uses a trimmed subset of the idea bank (top 10 items) to reduce tokens while keeping semantic variety.

Generate/refresh the idea bank:

```bash
python3 scripts/prepare_ideas_source.py
# run skill in agent:
# /tweet-ideas-generator
python3 scripts/update_ideas_from_skill.py --source /tmp/skills-*/tweet-ideas/tweets-*.md
```

## Notes

- `.env` is loaded via `load_dotenv`.
- Rate limits are handled with automatic retries (uses `retry_after` if provided).
- If you see frequent 429s, reduce `N_TWEETS`, `RECENT_TWEETS_MAX`, or trim the idea bank further.

## License

MIT

## Project Structure

```
content/           # source content + idea bank
src/crewx/         # pipeline + parsing + config
out/               # outputs + history
scripts/           # idea bank helpers
```
