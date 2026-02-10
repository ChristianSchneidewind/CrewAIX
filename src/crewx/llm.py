from __future__ import annotations

from crewai import LLM

from crewx.config import Settings


def build_llm(settings: Settings) -> LLM:
    return LLM(
        model=settings.openai_model_name,
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
        temperature=settings.temperature,
        max_tokens=250,  # keep responses fast and small
        timeout=60,
        max_retries=0,
    )
