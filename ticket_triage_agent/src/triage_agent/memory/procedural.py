"""Procedural memory: the response rules, kept as numbered versions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.store.sqlite import SqliteStore

from triage_agent.memory.store import DATA_DIR, PROCEDURAL

_CURRENT = "current"


def _version_key(version: int) -> str:
    return f"v{version}"


def list_versions(store: SqliteStore) -> list[dict[str, Any]]:
    """Every saved version, newest first. The current pointer is not a version."""
    items = store.search(PROCEDURAL, limit=100)
    versions: list[dict[str, Any]] = []
    for item in items:
        if item.key == _CURRENT:
            continue
        value = dict(item.value)
        if "version" not in value:
            continue
        versions.append(value)
    versions.sort(key=lambda row: int(row["version"]), reverse=True)
    return versions


def get_current(store: SqliteStore) -> dict[str, Any] | None:
    pointer = store.get(PROCEDURAL, _CURRENT)
    if pointer is None:
        return None
    version = int(pointer.value["version"])
    item = store.get(PROCEDURAL, _version_key(version))
    if item is None:
        raise LookupError(f"procedure version {version} does not exist")
    return dict(item.value)


def put_version(store: SqliteStore, text: str, note: str) -> dict[str, Any]:
    """Append the next version and make it current. Older versions stay."""
    existing = list_versions(store)
    nxt = 1 if not existing else max(int(row["version"]) for row in existing) + 1
    value: dict[str, Any] = {"version": nxt, "text": text, "note": note}
    store.put(PROCEDURAL, _version_key(nxt), value)
    store.put(PROCEDURAL, _CURRENT, {"version": nxt}, index=False)
    return value


def rollback(store: SqliteStore, version: int) -> dict[str, Any]:
    """Point current at an older version. Later versions remain stored."""
    item = store.get(PROCEDURAL, _version_key(version))
    if item is None:
        raise LookupError(f"procedure version {version} does not exist")
    store.put(PROCEDURAL, _CURRENT, {"version": version}, index=False)
    return dict(item.value)


def seed_procedures(store: SqliteStore, path: Path | None = None) -> None:
    """Load the style guide.

    A missing store gets v1. While current is still v1, the seed text is
    overwritten in place. A later current version is left alone, so a second
    run cannot add a version or undo a rollback.
    """
    rules_path = path or DATA_DIR / "seed_procedures.md"
    text = rules_path.read_text(encoding="utf-8").strip()
    pointer = store.get(PROCEDURAL, _CURRENT)
    if pointer is None:
        store.put(
            PROCEDURAL,
            _version_key(1),
            {"version": 1, "text": text, "note": "seed"},
        )
        store.put(PROCEDURAL, _CURRENT, {"version": 1}, index=False)
        return
    if int(pointer.value["version"]) == 1:
        store.put(
            PROCEDURAL,
            _version_key(1),
            {"version": 1, "text": text, "note": "seed"},
        )
