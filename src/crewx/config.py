import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_base: str
    openai_api_key: str
    openai_model_name: str
    tweets_md_path: str
    out_dir: str
    n_tweets: int


def load_settings() -> Settings:
    # Loads .env from CWD (project root)
    load_dotenv()

    return Settings(
        openai_api_base=os.getenv("OPENAI_API_BASE", "http://localhost:11434"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        openai_model_name=os.getenv("OPENAI_MODEL_NAME", "ollama/llama3:latest"),
        tweets_md_path=os.getenv("TWEETS_MD_PATH", "content/tweets.md"),
        out_dir=os.getenv("OUT_DIR", "out"),
        n_tweets=int(os.getenv("N_TWEETS", "10")),
    )


def apply_litellm_env() -> None:
    """
    Set LiteLLM-related env vars EARLY (before importing crewai / litellm indirectly).
    This reduces noise + avoids proxy-related imports where possible.
    """
    # Must be set from environment if present, otherwise defaults:
    os.environ["LITELLM_LOG"] = os.getenv("LITELLM_LOG", "ERROR")
    os.environ["LITELLM_DISABLE_TELEMETRY"] = os.getenv("LITELLM_DISABLE_TELEMETRY", "true")
    os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = os.getenv("LITELLM_SUPPRESS_DEBUG_INFO", "true")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    os.environ["LITELLM_LOGGING"] = os.getenv("LITELLM_LOGGING", "False")

    # Optional flags (safe even if LiteLLM ignores them)
    os.environ["LITELLM_ENABLE_PROXY"] = os.getenv("LITELLM_ENABLE_PROXY", "false")
    os.environ["LITELLM_USE_PROXY"] = os.getenv("LITELLM_USE_PROXY", "false")
