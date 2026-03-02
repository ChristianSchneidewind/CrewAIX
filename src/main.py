from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from typing import cast

from crewx.config import Settings, load_settings
from crewx.crew_pipeline import fix_history_unknown_types, run_generate_tweets_crewai

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_NO_TWEETS = 3
EXIT_UNKNOWN_ERROR = 1


def _exit_with_error(code: int, message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _resolve_version() -> str:
    try:
        return version("crewaix")
    except PackageNotFoundError:
        return "0.0.0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crewaix",
        description="Generate tweet queues for CrewAiX.",
    )
    parser.add_argument("--version", action="version", version=_resolve_version())

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Generate a tweet queue")
    run_parser.add_argument("--n-tweets", type=int, help="Number of tweets to generate")
    run_parser.add_argument("--recent", type=int, help="Recent tweets to consider")
    run_parser.add_argument("--model", help="Model name override")
    run_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    run_parser.add_argument("--out-dir", help="Output directory")
    run_parser.add_argument("--tweets", dest="tweets_md_path", help="Path to tweets.md")
    run_parser.add_argument(
        "--tweet-types", dest="tweet_types_md_path", help="Path to tweet_types.md"
    )
    run_parser.add_argument("--crew-roles", dest="crew_roles_md_path", help="Path to crew_roles.md")
    run_parser.add_argument("--ideas", dest="ideas_md_path", help="Path to ideas.md")
    run_parser.add_argument(
        "--force-types",
        help="Comma-separated list of tweet types to force",
    )
    run_parser.add_argument("--embedding-model", help="Embedding model name")
    run_parser.add_argument(
        "--embedding-threshold",
        type=float,
        help="Embedding similarity threshold",
    )
    run_parser.add_argument("--embedding-history-max", type=int, help="Embedding history max size")
    run_parser.add_argument(
        "--log-json",
        dest="log_json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable JSONL logs",
    )
    run_parser.add_argument("--log-dir", help="Custom log directory")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate but do not write queue/history",
    )
    run_parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="JSON output to stdout",
    )
    run_parser.add_argument(
        "--plain",
        dest="output_plain",
        action="store_true",
        help="Plain output to stdout",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    history_parser = subparsers.add_parser("fix-history", help="Fix unknown tweet types")
    history_parser.add_argument("--out-dir", help="Output directory")
    history_parser.add_argument(
        "--fallback-type",
        default="educational",
        help="Fallback type for unknown entries",
    )
    history_parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="JSON output to stdout",
    )

    return parser


def _apply_run_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    if args.n_tweets is not None:
        settings = replace(settings, n_tweets=args.n_tweets)
    if args.recent is not None:
        settings = replace(settings, recent_tweets_max=args.recent)
    if args.model:
        settings = replace(settings, openai_model_name=args.model)
    if args.temperature is not None:
        settings = replace(settings, temperature=args.temperature)
    if args.out_dir:
        settings = replace(settings, out_dir=args.out_dir)
    if args.tweets_md_path:
        settings = replace(settings, tweets_md_path=args.tweets_md_path)
    if args.tweet_types_md_path:
        settings = replace(settings, tweet_types_md_path=args.tweet_types_md_path)
    if args.crew_roles_md_path:
        settings = replace(settings, crew_roles_md_path=args.crew_roles_md_path)
    if args.ideas_md_path:
        settings = replace(settings, ideas_md_path=args.ideas_md_path)
    if args.force_types:
        settings = replace(
            settings,
            forced_tweet_types=tuple([t.strip() for t in args.force_types.split(",") if t.strip()]),
        )
    if args.embedding_model:
        settings = replace(settings, embedding_model_name=args.embedding_model)
    if args.embedding_threshold is not None:
        settings = replace(
            settings,
            embedding_similarity_threshold=args.embedding_threshold,
        )
    if args.embedding_history_max is not None:
        settings = replace(settings, embedding_history_max=args.embedding_history_max)
    if args.log_json is not None:
        settings = replace(settings, log_json=args.log_json)
    if args.log_dir:
        settings = replace(settings, log_dir=args.log_dir)
    if args.verbose:
        settings = replace(settings, verbose=True)
    return settings


def _format_run_output(
    result: dict[str, object],
    *,
    output_json: bool,
    output_plain: bool,
) -> str:
    if output_json:
        return json.dumps(result, ensure_ascii=False)
    if output_plain:
        return str(result.get("out_queue_path", ""))
    count = result.get("output_count", 0)
    path = result.get("out_queue_path", "")
    return f"Wrote {count} tweets to {path}"


def _format_history_output(changed: int, *, output_json: bool) -> str:
    if output_json:
        return json.dumps({"updated": changed}, ensure_ascii=False)
    return f"Updated {changed} history entries"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return EXIT_CONFIG_ERROR

    if args.command == "fix-history":
        settings = load_settings()
        out_dir = args.out_dir or settings.out_dir
        changed = fix_history_unknown_types(out_dir, fallback_type=args.fallback_type)
        print(_format_history_output(changed, output_json=args.output_json))
        return EXIT_OK

    if args.command == "run":
        settings = _apply_run_overrides(load_settings(), args)
        result = run_generate_tweets_crewai(settings, dry_run=args.dry_run)
        if result is None:
            _exit_with_error(EXIT_NO_TWEETS, "No tweets generated")
        result_data = cast(dict[str, object], result)
        print(
            _format_run_output(
                result_data,
                output_json=args.output_json,
                output_plain=args.output_plain,
            )
        )
        return EXIT_OK

    parser.print_help()
    return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        _exit_with_error(EXIT_CONFIG_ERROR, f"Config error: {exc}")
    except Exception as exc:
        _exit_with_error(EXIT_UNKNOWN_ERROR, f"CrewX error: {exc}")
