"""Semantic memory: knowledge-base facts and policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.store.sqlite import SqliteStore

from triage_agent.memory.store import DATA_DIR, SEMANTIC, ranked_hits


def article_value(row: dict[str, Any]) -> dict[str, str]:
    article_id = str(row["id"])
    title = str(row.get("title", ""))
    body = str(row.get("body", ""))
    source = str(row.get("source") or f"kb:{article_id}")
    return {
        "id": article_id,
        "kind": "kb",
        "title": title,
        "body": body,
        "source": source,
        "text": f"{title}\n{body}",
    }


def public_article(value: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(value["id"]),
        "source": str(value["source"]),
        "title": str(value["title"]),
        "body": str(value["body"]),
    }


def upsert_article(store: SqliteStore, row: dict[str, Any]) -> None:
    value = article_value(row)
    store.put(SEMANTIC, value["id"], value)


def get_article(store: SqliteStore, article_id: str) -> dict[str, Any] | None:
    item = store.get(SEMANTIC, article_id)
    if item is None:
        return None
    return dict(item.value)


def search_articles(
    store: SqliteStore,
    query: str,
    *,
    limit: int = 3,
    min_score: float = 0.2,
    relative_margin: float | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    return ranked_hits(
        store,
        SEMANTIC,
        query,
        limit=limit,
        min_score=min_score,
        relative_margin=relative_margin,
    )


def seed_articles(store: SqliteStore, path: Path | None = None) -> int:
    """Load kb articles. The article id is the key, so a second run overwrites."""
    kb_path = path or DATA_DIR / "kb_articles.json"
    rows = json.loads(kb_path.read_text(encoding="utf-8"))
    for row in rows:
        upsert_article(store, row)
    return len(rows)
