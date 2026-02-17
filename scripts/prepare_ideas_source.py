#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build reference material for tweet-ideas-generator from project content.",
    )
    parser.add_argument(
        "--out",
        default="content/ideas_source.md",
        help="Output file path (default: content/ideas_source.md)",
    )
    args = parser.parse_args()

    tweets_md = Path("content/tweets.md").read_text(encoding="utf-8")
    types_md = Path("content/tweet_types.md").read_text(encoding="utf-8")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = (
        "# Reference Material for Tweet Ideas\n\n"
        "Use this as the source input for /tweet-ideas-generator.\n\n"
        "---\n\n"
        "## Project Content (tweets.md)\n\n"
        f"{tweets_md.strip()}\n\n"
        "---\n\n"
        "## Tweet Types (tweet_types.md)\n\n"
        f"{types_md.strip()}\n"
    )

    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote reference material to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
