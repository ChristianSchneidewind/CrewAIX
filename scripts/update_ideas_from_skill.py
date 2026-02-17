#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

IDEA_LINE_RE = re.compile(r"^\s*\d+\.\s+\"([^\"]+)\"")


def _find_default_source_dir() -> Path:
    candidates = [Path("/tweet-ideas"), Path("tweet-ideas")]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    tmp_root = Path("/tmp")
    if tmp_root.exists():
        tmp_matches = sorted(tmp_root.glob("skills-*/tweet-ideas"), key=lambda p: p.stat().st_mtime, reverse=True)
        for match in tmp_matches:
            if match.exists() and match.is_dir():
                return match

    raise FileNotFoundError(
        "No tweet ideas directory found. Expected /tweet-ideas, ./tweet-ideas, or /tmp/skills-*/tweet-ideas."
    )


def _find_latest_file(source_dir: Path) -> Path:
    files = list(source_dir.glob("tweets-*.md"))
    if not files:
        raise FileNotFoundError(f"No tweets-*.md files found in {source_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


BUCKET_KEYWORDS = {
    "boarding_gate": ["boarding", "gate", "einsteigen", "boarden", "boarding-gruppe"],
    "gepaeck_handgepaeck": ["gepäck", "gepaeck", "handgepäck", "handgepaeck", "koffer", "gepäckband"],
    "gepaeckverlust": ["gepäckverlust", "gepaeckverlust", "verloren", "beschädigt", "beschädigung"],
    "checkin_sitzplatz": ["check-in", "checkin", "sitzplatz", "boardingpass"],
    "anschlussflug": ["anschlussflug", "umsteigen", "zubringer", "umsteig"],
    "wetter_irrops": ["wetter", "sturm", "schnee", "gewitter", "nebel"],
    "streik": ["streik", "arbeitskampf", "gewerkschaft"],
    "codeshare": ["codeshare", "code-share", "ausführende airline", "durchführende airline"],
    "sicherheit": ["sicherheitskontrolle", "security", "flüssig", "laptop"],
    "reiseruecktritt_kulanz": ["reiserücktritt", "reiseruecktritt", "storno", "umbuch", "erstattung", "kulanz"],
    "vielflieger": ["vielflieger", "status", "meilen", "bonusprogramm"],
    "betreuung": ["betreuung", "verpflegung", "hotel", "transfer", "ersatzbeförderung"],
}


def _infer_bucket(text: str) -> str | None:
    lower = text.lower()
    for bucket, needles in BUCKET_KEYWORDS.items():
        if any(n in lower for n in needles):
            return bucket
    return None


def _extract_ideas(text: str) -> list[str]:
    ideas: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = IDEA_LINE_RE.match(line)
        if not match:
            continue
        idea = match.group(1).strip()
        if idea and idea not in seen:
            ideas.append(idea)
            seen.add(idea)
    return ideas


def _balance_ideas(ideas: list[str], *, limit: int) -> list[str]:
    per_bucket = max(1, limit // max(1, len(BUCKET_KEYWORDS)))
    buckets: dict[str, list[str]] = {k: [] for k in BUCKET_KEYWORDS}
    misc: list[str] = []

    for idea in ideas:
        bucket = _infer_bucket(idea)
        if bucket:
            buckets[bucket].append(idea)
        else:
            misc.append(idea)

    balanced: list[str] = []
    for bucket, items in buckets.items():
        balanced.extend(items[:per_bucket])

    if len(balanced) < limit:
        remaining = [i for i in ideas if i not in balanced]
        balanced.extend(remaining[: max(0, limit - len(balanced))])

    return balanced[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract tweet ideas from tweet-ideas-generator output and write content/ideas.md",
    )
    parser.add_argument(
        "--source",
        help="Path to a tweets-*.md file. If omitted, uses the latest in /tweet-ideas or ./tweet-ideas.",
    )
    parser.add_argument(
        "--out",
        default="content/ideas.md",
        help="Output file path (default: content/ideas.md)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Max number of ideas to write (default: 60)",
    )

    args = parser.parse_args()

    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
    else:
        source_dir = _find_default_source_dir()
        source_path = _find_latest_file(source_dir)

    ideas_text = source_path.read_text(encoding="utf-8")
    ideas = _extract_ideas(ideas_text)

    if not ideas:
        print("No ideas found in source file.", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    limited = _balance_ideas(ideas, limit=max(1, args.limit))

    content = ["# Tweet Ideas", "", f"**Generated:** {timestamp}", f"**Source:** {source_path}", "", "---", ""]
    content.extend([f"- {idea}" for idea in limited])
    content.append("")

    out_path.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote {len(limited)} ideas to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
