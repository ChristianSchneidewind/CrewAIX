from __future__ import annotations

from crewai import Agent, LLM

from crewx.config import Settings


def build_agent(settings: Settings) -> Agent:
    llm = LLM(
        model=settings.openai_model_name,
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
        temperature=settings.temperature,
    )

    return Agent(
        role="X/Twitter Copywriter",
        goal="Generate high-quality, varied tweets for a company based on the provided markdown brief.",
        backstory="You are an expert social media copywriter with strong editorial standards.",
        llm=llm,
        verbose=settings.verbose,
    )
