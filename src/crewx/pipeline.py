from datetime import datetime
from pathlib import Path

from crewai import Task, Crew

from crewx.config import load_settings, apply_litellm_env
from crewx.io import read_text, write_json, write_text
from crewx.llm import build_llm, build_tweet_writer
from crewx.prompts import tweet_task_prompt, repair_prompt
from crewx.parsing import extract_json_object_from_text, fallback_parse_tweets_from_text, save_raw


def run_generate_tweets() -> None:
    # Must happen before crewai import triggers litellm usage in some environments
    apply_litellm_env()
    settings = load_settings()

    md = read_text(settings.tweets_md_path)

    llm = build_llm(settings)
    agent = build_tweet_writer(llm)

    task = Task(
        description=tweet_task_prompt(md, settings.n_tweets),
        expected_output=f"A JSON object with exactly {settings.n_tweets} tweets in the specified schema.",
        agent=agent,
    )

    result = Crew(agents=[agent], tasks=[task]).kickoff()
    raw = result if isinstance(result, str) else str(result)

    save_raw(settings.out_dir, raw)

    data = extract_json_object_from_text(raw)

    if data is None:
        fb = fallback_parse_tweets_from_text(raw, settings.n_tweets)
        if fb is not None:
            data = fb
        else:
            # Repair pass
            repair_task = Task(
                description=repair_prompt(raw, settings.n_tweets),
                expected_output="Valid JSON matching the schema.",
                agent=agent,
            )
            repair_result = Crew(agents=[agent], tasks=[repair_task]).kickoff()
            repair_raw = repair_result if isinstance(repair_result, str) else str(repair_result)
            data = extract_json_object_from_text(repair_raw)

    if data is None:
        raise RuntimeError(f"Could not parse tweets. See {Path(settings.out_dir) / 'last_raw_output.txt'}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_json = Path(settings.out_dir) / f"tweets_{timestamp}.json"
    out_md = Path(settings.out_dir) / f"tweets_{timestamp}.md"

    write_json(str(out_json), data)

    md_lines = [f"# Generated Tweets ({timestamp})", ""]
    for i, t in enumerate(data.get("tweets", []), start=1):
        md_lines.append(f"## Tweet {i}")
        md_lines.append(t.get("text", "").strip())
        md_lines.append("")
    write_text(str(out_md), "\n".join(md_lines))

    print(f"✅ Wrote: {out_json}")
    print(f"✅ Wrote: {out_md}")
