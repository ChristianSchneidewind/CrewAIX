from __future__ import annotations

import json

import pytest

from crewx.parsing import parse_tweet_types_md, parse_tweets_response


def test_parse_tweet_types_md_basic():
    md = """
# Tweet Types

## marketing
Goal: Mehr Conversions
Style:
- kurz
- klar
Rules:
- keine hashtags

## tip
Goal: Hilfreiche Tipps
Style:
- freundlich
"""
    types = parse_tweet_types_md(md)
    assert len(types) == 2
    assert types[0].name == "marketing"
    assert types[0].goal == "Mehr Conversions"
    assert types[0].style == ["kurz", "klar"]
    assert types[0].rules == ["keine hashtags"]
    assert "TYPE GOAL" in types[0].description
    assert types[1].name == "tip"


def test_parse_tweets_response_with_wrapped_json():
    raw = (
        "prefix text\n"
        + json.dumps(
            [
                {
                    "tweet_type": "fun_fact",
                    "opening_style": "question",
                    "text": "Wusstest du, dass ein Gate-Wechsel am Flughafen vorkommt?",
                    "language": "de",
                    "tags": ["#Myth"],
                },
                {
                    "text": "Wenn dein Flug am Gate verspätet ist, frag nach Betreuung.",
                    "tags": ["boarding_gate"],
                },
                "Nur ein String",
            ]
        )
        + "\ntrailing"
    )
    parsed = parse_tweets_response(raw, n_tweets=2, default_tweet_type="tip")
    assert len(parsed["tweets"]) == 2
    first = parsed["tweets"][0]
    assert first["tags"] == ["myth_vs_fact"]
    second = parsed["tweets"][1]
    assert second["tweet_type"] == "tip"
    assert second["language"] == "de"


def test_parse_tweets_response_requires_list():
    with pytest.raises(ValueError):
        parse_tweets_response('{"foo": 1}', n_tweets=1)


def test_parse_tweets_response_invalid_json():
    with pytest.raises(ValueError):
        parse_tweets_response("{invalid json", n_tweets=1)


def test_parse_tweets_response_mixed_types_and_default_type():
    raw = json.dumps(
        [
            {"text": "Wenn dein Flug am Gate ist, frag nach Betreuung."},
            123,
            {"tweet_type": "", "text": "Wenn dein Flug am Gate ist 2.", "tags": []},
        ]
    )
    parsed = parse_tweets_response(raw, n_tweets=5, default_tweet_type="service")
    assert len(parsed["tweets"]) == 2
    assert parsed["tweets"][0]["tweet_type"] == "service"
    assert parsed["tweets"][0]["language"] == "de"
    assert parsed["tweets"][1]["tweet_type"] == "service"
