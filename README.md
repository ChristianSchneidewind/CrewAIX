# CrewAiX

Generate German X (Twitter) posts for **FlugNinja** with a lightweight CrewAI pipeline. The project focuses on structured tweet generation, strict rule-based filtering, and de-duplication (including optional embedding similarity).

## Requirements

- Python 3.12+
- `uv` (recommended)
- OpenAI API key

## Quick Start

```bash
# install deps
uv sync

# set env
cp .env.example .env

# run generator
uv run python src/main.py run
```

See all CLI options:

```bash
uv run python src/main.py --help
uv run python src/main.py run --help
```

## CLI Usage

### Generate tweets

```bash
uv run python src/main.py run \
  --n-tweets 3 \
  --recent 10 \
  --temperature 0.7 \
  --json
```

Common flags:

- `--n-tweets`: number of tweets to generate
- `--recent`: recent tweets to consider for de-duplication
- `--model`: override model name
- `--temperature`: sampling temperature
- `--out-dir`: output directory
- `--tweets`, `--tweet-types`, `--crew-roles`, `--ideas`: override content paths
- `--force-types`: comma-separated list of tweet types to enforce
- `--embedding-model`, `--embedding-threshold`, `--embedding-history-max`: embedding dedupe controls
- `--dry-run`: generate but do not write queue/history
- `--json` or `--plain`: stdout output format
- `--log-json/--no-log-json`: enable/disable JSONL logging
- `--log-dir`: custom log directory
- `-v/--verbose`: verbose logging

### Fix history entries

```bash
uv run python src/main.py fix-history --fallback-type educational
```

This replaces `tweet_type=unknown` entries in `out/history.jsonl`.

## Configuration (.env)

Minimal setup:

```env
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4.1-mini

TWEETS_MD_PATH=content/tweets.md
TWEET_TYPES_MD_PATH=content/tweet_types.md
CREW_ROLES_MD_PATH=content/crew_roles.md
IDEAS_MD_PATH=content/ideas.md
OUT_DIR=out

N_TWEETS=3
RECENT_TWEETS_MAX=5

TEMPERATURE=0.7
VERBOSE=false
LOG_JSON=true
```

Embedding de-duplication (defaults to OpenAI embeddings when `OPENAI_API_KEY` is set):

```env
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_API_BASE=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-...   # optional; falls back to OPENAI_API_KEY
EMBEDDING_SIMILARITY_THRESHOLD=0.85
EMBEDDING_HISTORY_MAX=30
```

## Company Inputs

Update company-specific inputs in:

- `content/tweets.md` (company context, tone, constraints)
- `content/tweet_types.md` (tweet type definitions)
- `content/crew_roles.md` (CrewAI agent roles)
- `content/ideas.md` (idea bank, optional)

## How It Works

- **Generator → Reviewer → Poster** CrewAI pipeline.
- **Tweet type rotation**: if no `FORCE_TWEET_TYPES`, types are rotated based on history length.
- **Rules & buckets**: constraints and active buckets are in `config/rules.yaml`.
- **De-duplication**:
  - recent-history text filtering
  - one bucket and one tweet type per output
  - keyword quotas and history limits
  - optional embedding similarity filter
- **Output**:
  - queue saved to `out/post_queue_<timestamp>.json`
  - history appended to `out/history.jsonl`
  - raw LLM output saved to `out/last_raw_output.txt`
  - logs stored in `out/logs` (text + optional JSONL)

## Active Buckets

Default active buckets (from `config/rules.yaml`):

- `boarding_gate`
- `gepaeck_handgepaeck`
- `checkin_sitzplatz`
- `wetter_irrops`
- `streik`

You can adjust the active list and keywords in `config/rules.yaml`.

## Idea Bank

Only a trimmed subset of the idea bank is passed into prompts (see `idea_bank_max_items` in `config/rules.yaml`).

Update ideas via:

```bash
python3 scripts/prepare_ideas_source.py
# run skill in agent:
# /tweet-ideas-generator
python3 scripts/update_ideas_from_skill.py --source /tmp/skills-*/tweet-ideas/tweets-*.md
```

## Project Structure

```
content/           # source content + idea bank
config/            # rules + bucket configuration
src/crewx/         # pipeline + parsing + config
out/               # outputs + history + logs
scripts/           # idea bank helpers
```

## License

MIT
