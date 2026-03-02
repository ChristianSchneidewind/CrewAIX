from __future__ import annotations

from pathlib import Path

from crewx.parsing import TweetType
from crewx.prompts_pipeline import (
    build_generator_prompt,
    format_types_md,
    trim_company_context,
    trim_idea_bank,
)
from crewx.rules import IDEA_BANK_MAX_ITEMS


def test_trim_idea_bank_limits_items():
    ideas_md = "\n".join(f"- idee {i}" for i in range(IDEA_BANK_MAX_ITEMS + 2))
    trimmed = trim_idea_bank(ideas_md)
    lines = [line for line in trimmed.splitlines() if line.strip()]
    assert len(lines) == IDEA_BANK_MAX_ITEMS


def test_trim_idea_bank_without_bullets():
    ideas_md = "\n".join(f"Idee {i}" for i in range(IDEA_BANK_MAX_ITEMS + 2))
    trimmed = trim_idea_bank(ideas_md)
    lines = [line for line in trimmed.splitlines() if line.strip()]
    assert len(lines) == IDEA_BANK_MAX_ITEMS
    assert lines[0] == "Idee 0"


def test_trim_company_context_filters_sections():
    company_md = """
# Company

## Company
- A

## Product / Offer
- B

## Legal
- Should be removed

## Tone & Voice
- C
"""
    trimmed = trim_company_context(company_md)
    assert "## Legal" not in trimmed
    assert "## Company" in trimmed
    assert "## Tone & Voice" in trimmed


def test_trim_company_context_fallback():
    company_md = """
# Company

## Legal
- Should remain
""".strip()
    trimmed = trim_company_context(company_md)
    assert trimmed == company_md


def test_format_types_md_compact():
    types = [TweetType(name="marketing", goal="Mehr Leads", style=[], rules=[])]
    result = format_types_md(types)
    assert "# Tweet Types (compact)" in result
    assert "## marketing" in result
    assert "Goal: Mehr Leads" in result


def test_build_generator_prompt_recent_and_required_types():
    prompt = build_generator_prompt(
        company_md="## Company\n- A",
        types_md="## marketing\nGoal: Mehr",
        ideas_md="- Idee",
        n_tweets=3,
        recent=["alt 1", "alt 2", "alt 3", "alt 4", "alt 5"],
        required_types=["marketing", "tip"],
    )
    assert "REQUIRED TYPES: marketing, tip" in prompt
    assert "alt 5" in prompt
    assert "alt 1" not in prompt


def test_build_generator_prompt_snapshot():
    prompt = build_generator_prompt(
        company_md="""
## Company
- A

## Product / Offer
- B

## Legal
- Should be removed
""".strip(),
        types_md="## marketing\nGoal: Mehr",
        ideas_md="- Idee 1\n- Idee 2",
        n_tweets=2,
        recent=["alt 1", "alt 2", "alt 3", "alt 4"],
        required_types=None,
    )
    snapshot_path = Path(__file__).parent / "snapshots" / "generator_prompt.txt"
    expected = snapshot_path.read_text(encoding="utf-8")
    assert prompt == expected
