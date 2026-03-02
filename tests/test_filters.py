from __future__ import annotations

from crewx.filters import accept_relaxed_candidate, filter_crewai_tweets, normalize_candidate_fields


def test_normalize_candidate_fields_adds_defaults():
    tweet = {"text": "Wenn dein Flug am Gate verspätet ist, frag nach Betreuung."}
    normalized = normalize_candidate_fields(tweet)
    assert normalized["language"] == "de"
    assert normalized["opening_style"] == "condition"
    assert "boarding_gate" in normalized["tags"]
    assert "condition" in normalized["tags"]


def test_accept_relaxed_candidate_rejects_hashtag():
    tweet = {
        "tweet_type": "service",
        "text": "Wenn du am Gate bist #hashtag, frag nach Betreuung.",
        "tags": ["boarding_gate"],
    }
    assert accept_relaxed_candidate(tweet, allowed_types=None, type_limits=None) is False


def test_accept_relaxed_candidate_allows_valid():
    tweet = {
        "tweet_type": "service",
        "text": "Wenn du am Gate bist, frag nach Betreuung.",
        "tags": ["boarding_gate"],
    }
    assert accept_relaxed_candidate(tweet, allowed_types=None, type_limits=None) is True


def test_filter_crewai_tweets_bucket_dedupe_and_cta_rules():
    tweets = [
        {
            "tweet_type": "industry_insight",
            "text": "Wenn du am Gate wartest, frag nach einer Umbuchung.",
            "tags": ["boarding_gate"],
        },
        {
            "tweet_type": "industry_insight",
            "text": "Wenn du am Gate bist, erkundige dich nach der Boardinggruppe.",
            "tags": ["boarding_gate"],
        },
        {
            "tweet_type": "service",
            "text": "Wenn du am Gate bist, jetzt prüfen bei Flugninja.",
            "tags": ["boarding_gate"],
        },
        {
            "tweet_type": "marketing",
            "text": "Wenn dein Fluggepäck am Band fehlt, jetzt prüfen bei Flugninja.",
            "tags": ["gepaeck_handgepaeck"],
        },
    ]

    filtered = filter_crewai_tweets(
        tweets,
        recent_texts=[],
        max_travel_hack=1,
        allowed_types=None,
        type_limits=None,
        embedding_threshold=None,
        recent_embeddings=None,
        candidate_embeddings=None,
    )

    assert len(filtered) == 2
    texts = {t["text"] for t in filtered}
    assert "Wenn du am Gate wartest, frag nach einer Umbuchung." in texts
    assert "Wenn dein Fluggepäck am Band fehlt, jetzt prüfen bei Flugninja." in texts


def test_filter_crewai_tweets_keyword_quota_blocks_second():
    tweets = [
        {
            "tweet_type": "service",
            "text": "Wenn dein Flug am Gate annulliert wird, frag nach Betreuung.",
            "tags": ["boarding_gate"],
        },
        {
            "tweet_type": "service",
            "text": "Wenn dein Flug wegen Wetter annulliert wird, sichere dir Verpflegung.",
            "tags": ["wetter_irrops"],
        },
    ]

    filtered = filter_crewai_tweets(
        tweets,
        recent_texts=[],
        max_travel_hack=1,
    )

    assert len(filtered) == 1
    assert filtered[0]["text"].startswith("Wenn dein Flug am Gate")


def test_filter_crewai_tweets_travel_hack_limit():
    tweets = [
        {
            "tweet_type": "travel_hack",
            "text": "Wenn dein Flug am Gate verspätet ist, check die Boardinggruppe.",
            "tags": ["boarding_gate"],
        },
        {
            "tweet_type": "travel_hack",
            "text": "Wenn dein Flug wegen Wetter verspätet ist, such die Betreuung am Gate.",
            "tags": ["wetter_irrops"],
        },
    ]

    filtered = filter_crewai_tweets(
        tweets,
        recent_texts=[],
        max_travel_hack=1,
    )

    assert len(filtered) == 1
    assert filtered[0]["tweet_type"] == "travel_hack"


def test_filter_crewai_tweets_doc_tip_blocked_by_recent():
    tweets = [
        {
            "tweet_type": "service",
            "text": "Wenn dein Flug am Gate ist, check den Boardingpass.",
            "tags": ["boarding_gate"],
        },
        {
            "tweet_type": "service",
            "text": "Wenn dein Flug wegen Streik ausfällt, frag nach Ersatz.",
            "tags": ["streik"],
        },
    ]

    filtered = filter_crewai_tweets(
        tweets,
        recent_texts=["Bitte Boardingpass bereithalten"],
        max_travel_hack=1,
    )

    assert len(filtered) == 1
    assert filtered[0]["text"].startswith("Wenn dein Flug wegen Streik")
