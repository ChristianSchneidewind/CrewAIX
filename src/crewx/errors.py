from __future__ import annotations


class CrewXError(RuntimeError):
    """Base class for CrewX pipeline errors."""


class ConfigurationError(CrewXError):
    """Configuration is missing or invalid."""


class NoTweetTypesError(CrewXError):
    """No tweet types were parsed from the input file."""


class NoTweetsGeneratedError(CrewXError):
    """Pipeline finished without producing any tweets."""


class RateLimitError(CrewXError):
    """Rate limit encountered and no fallback succeeded."""
