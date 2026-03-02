import sys

from crewx.crew_pipeline import run_generate_tweets_crewai
from crewx.errors import ConfigurationError, CrewXError, NoTweetsGeneratedError, RateLimitError

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_NO_TWEETS = 3
EXIT_RATE_LIMIT = 4
EXIT_UNKNOWN_ERROR = 1


def _exit_with_error(code: int, message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


if __name__ == "__main__":
    try:
        run_generate_tweets_crewai()
    except ConfigurationError as exc:
        _exit_with_error(EXIT_CONFIG_ERROR, f"Config error: {exc}")
    except RateLimitError as exc:
        _exit_with_error(EXIT_RATE_LIMIT, f"Rate limit: {exc}")
    except NoTweetsGeneratedError as exc:
        _exit_with_error(EXIT_NO_TWEETS, f"No tweets generated: {exc}")
    except CrewXError as exc:
        _exit_with_error(EXIT_UNKNOWN_ERROR, f"CrewX error: {exc}")
