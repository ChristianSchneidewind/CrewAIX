from __future__ import annotations

from pathlib import Path

import pytest

from crewx.retry import RateLimitHit, kickoff_with_retry, parse_retry_after_seconds


class DummyCrew:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def kickoff(self):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_kickoff_with_retry_rate_limit(monkeypatch):
    crew = DummyCrew([Exception("Rate limit, try again in 0.01 s"), "ok"])
    monkeypatch.setattr("crewx.retry.time.sleep", lambda *_: None)
    result = kickoff_with_retry(crew, max_retries=1, base_delay=0.0)
    assert result == "ok"


def test_kickoff_with_retry_rate_limit_fail_fast(monkeypatch):
    crew = DummyCrew([Exception("rate_limit"), "ok"])
    monkeypatch.setattr("crewx.retry.time.sleep", lambda *_: None)
    with pytest.raises(RateLimitHit):
        kickoff_with_retry(crew, max_retries=1, base_delay=0.0, fail_fast_on_rate_limit=True)


def test_kickoff_with_retry_connection_error_logs(tmp_path, monkeypatch):
    debug_path = tmp_path / "retry.log"
    crew = DummyCrew([Exception("Connection error"), "ok"])
    monkeypatch.setattr("crewx.retry.time.sleep", lambda *_: None)

    result = kickoff_with_retry(
        crew,
        max_retries=1,
        base_delay=0.0,
        debug_path=str(debug_path),
    )

    assert result == "ok"
    assert "CONNECTION ERROR" in Path(debug_path).read_text(encoding="utf-8")


def test_parse_retry_after_seconds_units():
    assert parse_retry_after_seconds("Try again in 500 ms") == 0.5
    assert parse_retry_after_seconds("try again in 2 s") == 2.0
    assert parse_retry_after_seconds("no hint") is None
