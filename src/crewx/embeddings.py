from __future__ import annotations

import math
from typing import Iterable

from litellm import embedding as litellm_embedding


def is_embedding_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "invalid_api_key" in message or "incorrect api key" in message or "401" in message


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_texts(texts: list[str], settings) -> list[list[float]]:
    if not texts or not settings.embedding_model_name:
        raise ValueError("Embedding disabled or empty input")
    response = litellm_embedding(
        model=settings.embedding_model_name,
        input=texts,
        api_key=settings.embedding_api_key or settings.openai_api_key,
        base_url=settings.embedding_api_base or settings.openai_api_base,
    )

    if isinstance(response, dict) and response.get("error"):
        raise ValueError(f"Embedding error: {response.get('error')}")

    data = None
    if isinstance(response, dict) and isinstance(response.get("data"), list):
        data = response.get("data")
    elif isinstance(response, list):
        data = response
    elif isinstance(response, dict) and isinstance(response.get("embedding"), list):
        return [response.get("embedding")]
    else:
        response_data = getattr(response, "data", None)
        if isinstance(response_data, list):
            data = response_data
        elif hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict) and isinstance(dumped.get("data"), list):
                data = dumped.get("data")

    if not data:
        keys = list(response.keys()) if isinstance(response, dict) else type(response).__name__
        raise ValueError(f"No embedding data returned (response={keys})")

    embeddings: list[list[float]] = []
    for item in data:
        emb = item.get("embedding") if isinstance(item, dict) else None
        if isinstance(emb, list):
            embeddings.append(emb)
    if not embeddings:
        raise ValueError("No embeddings parsed")
    return embeddings


def build_embedding_map(texts: list[str], settings) -> dict[str, list[float]]:
    embeddings = embed_texts(texts, settings)
    mapping: dict[str, list[float]] = {}
    for text, emb in zip(texts, embeddings):
        if text and text not in mapping:
            mapping[text] = emb
    if not mapping:
        raise ValueError("Empty embedding map")
    return mapping
