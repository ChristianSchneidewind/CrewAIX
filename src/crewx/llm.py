from crewai import Agent, LLM
from crewx.config import Settings


def build_llm(settings: Settings) -> LLM:
    return LLM(
        model=settings.openai_model_name,
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
    )


def build_tweet_writer(llm: LLM) -> Agent:
    return Agent(
        role="X/Twitter Copywriter",
        goal="Generate high-quality X tweets from a markdown brief.",
        backstory="You write concise, punchy, non-cringe tweets and follow constraints strictly.",
        llm=llm,
    )
