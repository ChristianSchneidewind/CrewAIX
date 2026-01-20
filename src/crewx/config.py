from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # LLM / provider
    openai_api_base: str
    openai_api_key: str
    openai_model_name: str
    temperature: float = 0.7
    verbose: bool = False

    # Content inputs
    tweets_md_path: str = "content/tweets.md"
    tweet_types_md_path: str = "content/tweet_types.md"

    # Output
    out_dir: str = "out"
    n_tweets: int = 10

    # Dedup scope (for prompt “recent tweets”)
    recent_tweets_max: int = 50


def _get_env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def load_settings() -> Settings:
    # Defaults are chosen to match your .env conventions
    openai_api_base = _get_env("OPENAI_API_BASE", "http://localhost:11434") or "http://localhost:11434"
    openai_api_key = _get_env("OPENAI_API_KEY", "ollama") or "ollama"
    openai_model_name = _get_env("OPENAI_MODEL_NAME", "ollama/llama3:latest") or "ollama/llama3:latest"

    tweets_md_path = _get_env("TWEETS_MD_PATH", "content/tweets.md") or "content/tweets.md"
    tweet_types_md_path = _get_env("TWEET_TYPES_MD_PATH", "content/tweet_types.md") or "content/tweet_types.md"
    out_dir = _get_env("OUT_DIR", "out") or "out"

    # Optional knobs
    n_tweets = int(_get_env("N_TWEETS", "10") or "10")
    recent_tweets_max = int(_get_env("RECENT_TWEETS_MAX", "50") or "50")

    # Temperature/verbose are optional; keep safe defaults
    temperature = float(_get_env("TEMPERATURE", "0.7") or "0.7")
    verbose = (_get_env("VERBOSE", "false") or "false").lower() in {"1", "true", "yes", "y", "on"}

    return Settings(
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        openai_model_name=openai_model_name,
        temperature=temperature,
        verbose=verbose,
        tweets_md_path=tweets_md_path,
        tweet_types_md_path=tweet_types_md_path,
        out_dir=out_dir,
        n_tweets=n_tweets,
        recent_tweets_max=recent_tweets_max,
    )


def apply_litellm_env() -> None:
    """
    Keep LiteLLM noise down and avoid pulling proxy dependencies.
    This is env-only config; it does NOT start any proxy.
    """
    os.environ.setdefault("LITELLM_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("LITELLM_SUPPRESS_DEBUG_INFO", "true")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "ERROR") or "ERROR")

    # Avoid weird "logging.NONE" / invalid log levels
    if os.getenv("LITELLM_LOGGING", "").strip().lower() in {"false", "0", "no"}:
        os.environ["LITELLM_LOGGING"] = "false"
