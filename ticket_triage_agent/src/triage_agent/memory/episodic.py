"""Episodic memory: resolved tickets and the replies that were approved."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.store.sqlite import SqliteStore

from triage_agent.memory.store import DATA_DIR, EPISODIC, ranked_hits


def episode_value(row: dict[str, Any]) -> dict[str, str]:
    episode_id = str(row["id"])
    subject = str(row.get("subject", ""))
    body = str(row.get("body", ""))
    return {
        "id": episode_id,
        "ticket_id": str(row.get("ticket_id", "")),
        "category": str(row.get("category", "")),
        "subject": subject,
        "body": body,
        "approved_reply": str(row.get("approved_reply", "")),
        "text": f"{subject}\n{body}",
    }


def upsert_episode(store: SqliteStore, row: dict[str, Any]) -> None:
    value = episode_value(row)
    store.put(EPISODIC, value["id"], value)


def get_episode(store: SqliteStore, episode_id: str) -> dict[str, Any] | None:
    item = store.get(EPISODIC, episode_id)
    if item is None:
        return None
    return dict(item.value)


def search_episodes(
    store: SqliteStore,
    query: str,
    *,
    limit: int = 3,
    min_score: float = 0.2,
) -> list[tuple[float, dict[str, Any]]]:
    return ranked_hits(store, EPISODIC, query, limit=limit, min_score=min_score)


def seed_episodes(store: SqliteStore, path: Path | None = None) -> int:
    """Load resolved examples. The episode id is the key, so a second run overwrites."""
    episode_path = path or DATA_DIR / "seed_episodes.json"
    rows = json.loads(episode_path.read_text(encoding="utf-8"))
    for row in rows:
        upsert_episode(store, row)
    return len(rows)
