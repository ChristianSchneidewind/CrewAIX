from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from litellm import embedding as litellm_embedding

from crewx.io import ensure_dir


def is_embedding_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "invalid_api_key" in message or "incorrect api key" in message or "401" in message


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list, strict=True))
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
        embedding = response.get("embedding")
        if isinstance(embedding, list):
            if all(isinstance(value, int | float) for value in embedding):
                return [[float(value) for value in embedding]]
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
        if isinstance(emb, list) and all(isinstance(value, int | float) for value in emb):
            embeddings.append([float(value) for value in emb])
    if not embeddings:
        raise ValueError("No embeddings parsed")
    return embeddings


def _ensure_cache(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (text TEXT PRIMARY KEY, embedding TEXT NOT NULL)"
    )
    conn.commit()


def _load_cached_embeddings(conn: sqlite3.Connection, texts: list[str]) -> dict[str, list[float]]:
    if not texts:
        return {}
    cached: dict[str, list[float]] = {}
    chunk_size = 900
    for idx in range(0, len(texts), chunk_size):
        chunk = texts[idx : idx + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        query = f"SELECT text, embedding FROM embeddings WHERE text IN ({placeholders})"
        for text, emb_json in conn.execute(query, chunk):
            try:
                emb = json.loads(emb_json)
            except json.JSONDecodeError:
                continue
            if isinstance(emb, list) and all(isinstance(v, int | float) for v in emb):
                cached[str(text)] = [float(v) for v in emb]
    return cached


def _store_embeddings(conn: sqlite3.Connection, embeddings: dict[str, list[float]]) -> None:
    if not embeddings:
        return
    rows = [(text, json.dumps(emb)) for text, emb in embeddings.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (text, embedding) VALUES (?, ?)",
        rows,
    )
    conn.commit()


def _cache_path(settings) -> Path | None:
    raw = getattr(settings, "embedding_cache_path", None)
    if not raw:
        return None
    path = Path(raw)
    ensure_dir(path.parent)
    return path


def embed_texts_cached(texts: list[str], settings) -> list[list[float]]:
    if not texts or not settings.embedding_model_name:
        raise ValueError("Embedding disabled or empty input")
    cache_path = _cache_path(settings)
    if cache_path is None:
        return embed_texts(texts, settings)

    unique_texts = [t for t in dict.fromkeys(texts) if t]
    with sqlite3.connect(cache_path) as conn:
        _ensure_cache(conn)
        cached = _load_cached_embeddings(conn, unique_texts)
        missing = [t for t in unique_texts if t not in cached]
        if missing:
            fetched = embed_texts(missing, settings)
            if len(fetched) != len(missing):
                raise ValueError("Embedding count mismatch")
            new_entries = {text: emb for text, emb in zip(missing, fetched, strict=True)}
            _store_embeddings(conn, new_entries)
            cached.update(new_entries)

    return [cached[t] for t in texts]


def build_embedding_map(texts: list[str], settings) -> dict[str, list[float]]:
    embeddings = embed_texts_cached(texts, settings)
    if len(embeddings) != len(texts):
        raise ValueError("Embedding count mismatch")
    mapping: dict[str, list[float]] = {}
    for text, emb in zip(texts, embeddings, strict=True):
        if text and text not in mapping:
            mapping[text] = emb
    if not mapping:
        raise ValueError("Empty embedding map")
    return mapping
