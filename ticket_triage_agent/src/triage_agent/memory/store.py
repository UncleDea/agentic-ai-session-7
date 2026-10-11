"""LangGraph SqliteStore. One file for dev, `:memory:` for tests."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.store.sqlite import SqliteStore

from triage_agent.config import Settings

SEMANTIC = ("semantic", "kb")
EPISODIC = ("episodic", "resolved_tickets")
PROCEDURAL = ("procedural", "response_rules")
META = ("meta", "store")

FAKE_MODEL = "fake"
FAKE_DIMS = 8192
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_TITLE_WEIGHT = 3.0
_BODY_WEIGHT = 1.0


def _bucket(token: str, dims: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dims


def fake_embeddings(dims: int = FAKE_DIMS):
    """Deterministic token vectors. No network. Title tokens outweigh body tokens."""

    def embed(texts: list[str]) -> list[list[float]]:
        from triage_agent.tools.knowledge_base import content_tokens

        vectors: list[list[float]] = []
        for text in texts:
            title, _, body = text.partition("\n")
            counts: dict[int, float] = {}
            for token in content_tokens(title):
                bucket = _bucket(token, dims)
                counts[bucket] = counts.get(bucket, 0.0) + _TITLE_WEIGHT
            for token in content_tokens(body):
                bucket = _bucket(token, dims)
                counts[bucket] = counts.get(bucket, 0.0) + _BODY_WEIGHT
            vector = [0.0] * dims
            for bucket, weight in counts.items():
                vector[bucket] += weight
            vectors.append(vector)
        return vectors

    return embed


def open_store(path: str, embeddings, *, dims: int, model: str) -> SqliteStore:
    """Open a store and stamp the embedding model on first use.

    `isolation_level=None` is required. SqliteStore issues its own BEGIN, and a
    connection in the default isolation mode raises
    `cannot start a transaction within a transaction`.
    The index key is `text_fields`. This SqliteStore build ignores `fields`.
    """
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    store = SqliteStore(
        conn,
        index={
            "dims": dims,
            "embed": embeddings,
            "text_fields": ["text"],
        },
    )
    store.setup()
    existing = store.get(META, "embeddings")
    if existing is None:
        store.put(META, "embeddings", {"model": model, "dims": dims}, index=False)
        return store
    stamped_model = existing.value.get("model")
    stamped_dims = existing.value.get("dims")
    if stamped_model != model or stamped_dims != dims:
        conn.close()
        raise ValueError(
            f"store at {path} was built with {stamped_model} ({stamped_dims} dims); "
            f"this process is using {model} ({dims} dims)"
        )
    return store


def bedrock_embeddings(model_id: str, *, dimensions: int = 1024):
    """Titan Text Embeddings v2 (or another Bedrock model) in AWS_REGION."""
    from langchain_aws import BedrockEmbeddings

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    kwargs: dict[str, object] = {"model_id": model_id, "dimensions": dimensions}
    if region:
        kwargs["region_name"] = region
    return BedrockEmbeddings(**kwargs)


def store_for_settings(settings: Settings) -> SqliteStore:
    """Fake embeddings and `:memory:` unless settings name a model and a file."""
    path = settings.memory_db_path or ":memory:"
    if settings.embedding_model:
        embeddings = bedrock_embeddings(
            settings.embedding_model,
            dimensions=settings.embedding_dims,
        )
        return open_store(
            path,
            embeddings,
            dims=settings.embedding_dims,
            model=settings.embedding_model,
        )
    return open_store(
        path,
        fake_embeddings(FAKE_DIMS),
        dims=FAKE_DIMS,
        model=FAKE_MODEL,
    )


def ranked_hits(
    store: SqliteStore,
    namespace: tuple[str, ...],
    query: str,
    *,
    limit: int,
    min_score: float,
    relative_margin: float | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    """Best score per key, highest first.

    Hits under min_score are dropped. When relative_margin is set, a hit must
    also reach top_score * (1 - relative_margin).
    """
    raw = store.search(namespace, query=query, limit=max(limit * 5, 20))
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for item in raw:
        score = float(item.score) if item.score is not None else 0.0
        value = dict(item.value)
        previous = best.get(item.key)
        if previous is None or score > previous[0]:
            best[item.key] = (score, value)
    ordered = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
    return select_hits(
        ordered,
        min_score=min_score,
        relative_margin=relative_margin,
        limit=limit,
    )


def select_hits(
    scored: list[tuple[float, dict[str, Any]]],
    *,
    min_score: float,
    relative_margin: float | None,
    limit: int,
) -> list[tuple[float, dict[str, Any]]]:
    """Keep scores at or above min_score and within the relative margin of the top."""
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
    eligible = [pair for pair in ranked if pair[0] >= min_score]
    if not eligible:
        return []
    top = eligible[0][0]
    if relative_margin is not None:
        floor = top * (1.0 - relative_margin)
        eligible = [pair for pair in eligible if pair[0] >= floor]
    return eligible[:limit]
