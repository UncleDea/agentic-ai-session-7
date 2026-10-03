"""Startup check, review layout, and the batch summary. No interactive prompts."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict

from langgraph.store.sqlite import SqliteStore

from triage_agent.config import Settings, resolve_dir
from triage_agent.memory.procedural import get_current
from triage_agent.memory.store import (
    DATA_DIR,
    EPISODIC,
    FAKE_DIMS,
    FAKE_MODEL,
    META,
    SEMANTIC,
    store_for_settings,
)
from triage_agent.nodes.triage import load_sample_tickets
from triage_agent.state import Ticket

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _ROOT / ".env.example"
_STAMP_PREFIX = ".".join(META)
_STAMP_KEY = "embeddings"

FAKE_STORE_MESSAGE = (
    "delete data/memory.sqlite, set EMBEDDING_MODEL in .env, "
    "and rerun scripts/seed_memory.py"
)


class SummaryRow(TypedDict):
    ticket: str
    category: str
    route: str
    hitl_reason: str
    went_to_hitl: bool
    escalated: bool
    sent: bool


def example_setting_names(path: Path | None = None) -> list[str]:
    """Uncommented names in `.env.example`. Values are discarded."""
    example = path or _EXAMPLE
    names: list[str] = []
    for raw in example.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        name, sep, _value = line.partition("=")
        if not sep:
            continue
        cleaned = name.strip()
        if cleaned:
            names.append(cleaned)
    return names


def missing_setting_names(names: list[str], environ: Mapping[str, str]) -> list[str]:
    """Names that are absent or blank. The return value never includes values."""
    missing: list[str] = []
    for name in names:
        raw = environ.get(name)
        if raw is None or str(raw).strip() == "":
            missing.append(name)
    return missing


def missing_settings_line(names: list[str]) -> str:
    if not names:
        return "Missing settings: none"
    return "Missing settings: " + ", ".join(names)


def read_embedding_stamp(path: Path) -> dict[str, Any] | None:
    """The embeddings stamp, or None when the file has no stamp.

    This reads the row directly so a check cannot stamp an empty store.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        try:
            row = conn.execute(
                "SELECT value FROM store WHERE prefix = ? AND key = ?",
                (_STAMP_PREFIX, _STAMP_KEY),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    finally:
        conn.close()
    if row is None:
        return None
    payload = json.loads(row[0])
    if not isinstance(payload, dict):
        return None
    return payload


def is_seeded(store: SqliteStore) -> bool:
    """True when procedures, articles, and episodes are all present."""
    if get_current(store) is None:
        return False
    if not store.search(SEMANTIC, limit=1):
        return False
    if not store.search(EPISODIC, limit=1):
        return False
    return True


def absolute_settings(settings: Settings) -> Settings:
    """Resolve store and directory paths from the package root."""
    update: dict[str, str] = {
        "outbox_dir": str(resolve_dir(settings.outbox_dir)),
        "escalations_dir": str(resolve_dir(settings.escalations_dir)),
    }
    if settings.memory_db_path:
        update["memory_db_path"] = str(resolve_dir(settings.memory_db_path))
    if settings.checkpoint_db_path:
        update["checkpoint_db_path"] = str(resolve_dir(settings.checkpoint_db_path))
    return settings.model_copy(update=update)


def check_startup(
    settings: Settings,
    *,
    example_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    opener: Callable[[Settings], SqliteStore] | None = None,
) -> list[str]:
    """Lines to print. Raises SystemExit with the same text when the store cannot be used.

    Setting values are never included. A store stamped with the fake embedder stops
    before any ticket runs.
    """
    example = example_path or _EXAMPLE
    env = os.environ if environ is None else environ
    names = example_setting_names(example)
    missing = missing_setting_names(names, env)
    lines = [missing_settings_line(missing)]
    if settings.memory_db_path:
        path = resolve_dir(settings.memory_db_path)
    else:
        path = DATA_DIR / "memory.sqlite"
    if not path.is_file():
        lines.append(
            f"No memory store at {path}. Run: python scripts/seed_memory.py"
        )
        raise SystemExit("\n".join(lines))
    stamp = read_embedding_stamp(path)
    model = None if stamp is None else stamp.get("model")
    dims = None if stamp is None else stamp.get("dims")
    lines.append(f"Embedding model: {model if model else '(not stamped)'}")
    if model == FAKE_MODEL:
        lines.append(FAKE_STORE_MESSAGE)
        raise SystemExit("\n".join(lines))
    if not model:
        lines.append(
            f"Memory store at {path} has no embedding stamp. "
            "Run: python scripts/seed_memory.py"
        )
        raise SystemExit("\n".join(lines))
    using_model = settings.embedding_model or FAKE_MODEL
    using_dims = settings.embedding_dims if settings.embedding_model else FAKE_DIMS
    if model != using_model or dims != using_dims:
        lines.append(
            f"store at {path} was built with {model} ({dims} dims); "
            f"this process is using {using_model} ({using_dims} dims)"
        )
        raise SystemExit("\n".join(lines))
    opened = settings.model_copy(update={"memory_db_path": str(path)})
    open_fn = opener or store_for_settings
    try:
        store = open_fn(opened)
    except SystemExit:
        raise
    except Exception as exc:
        lines.append(f"Could not open the memory store: {type(exc).__name__}")
        raise SystemExit("\n".join(lines)) from None
    try:
        if not is_seeded(store):
            lines.append(
                f"Memory store at {path} is not seeded. "
                "Run: python scripts/seed_memory.py"
            )
            raise SystemExit("\n".join(lines))
    finally:
        store.conn.close()
    lines.append("Memory store is seeded.")
    if missing:
        raise SystemExit("\n".join(lines))
    return lines


def load_tickets() -> list[dict]:
    return load_sample_tickets()


def ticket_from(record: dict) -> Ticket:
    payload = {key: value for key, value in record.items() if key != "expected"}
    return Ticket.model_validate(payload)


def find_ticket(ticket_id: str) -> Ticket:
    for record in load_tickets():
        if record["id"] == ticket_id:
            return ticket_from(record)
    raise SystemExit(f"Unknown ticket id: {ticket_id}")


def format_review(payload: dict[str, Any]) -> str:
    """Readable layout for one HITL interrupt payload."""
    ticket = payload.get("ticket") or {}
    triage = payload.get("triage") or {}
    findings = payload.get("findings") or []
    lines = [
        f"Ticket {ticket.get('id', '')}",
        f"Subject: {ticket.get('subject', '')}",
        f"Body: {ticket.get('body', '')}",
        (
            f"Category: {triage.get('category', '')}  "
            f"Priority: {triage.get('priority', '')}"
        ),
        "Why review: " + "; ".join(payload.get("reasons") or []),
        "Findings:",
    ]
    if not findings:
        lines.append("  (none)")
    for item in findings:
        fact = item.get("fact", "") if isinstance(item, dict) else ""
        source = item.get("source", "") if isinstance(item, dict) else ""
        lines.append(f"  - {fact} ({source})")
    lines.append("Draft:")
    lines.append(str(payload.get("draft") or ""))
    flagged = list(payload.get("flagged_claims") or [])
    lines.append(
        "Flagged claims: " + (", ".join(str(item) for item in flagged) if flagged else "(none)")
    )
    lines.append("Escalates: " + ("yes" if payload.get("escalates") else "no"))
    return "\n".join(lines)


def _field(entry: Any, name: str) -> str:
    if isinstance(entry, dict):
        return str(entry.get(name, ""))
    return str(getattr(entry, name, ""))


def row_from_values(values: dict[str, Any], ticket_id: str = "") -> SummaryRow:
    """One summary row from a checkpoint's values."""
    ticket = values.get("ticket")
    if ticket is not None and getattr(ticket, "id", None):
        ticket_id = str(ticket.id)
    elif isinstance(ticket, dict) and ticket.get("id"):
        ticket_id = str(ticket["id"])
    triage = values.get("triage")
    if triage is None:
        category = "-"
    elif isinstance(triage, dict):
        category = str(triage.get("category", "-"))
    else:
        category = triage.category.value
    history = list(values.get("route_history") or [])
    route = " > ".join(_field(entry, "node") for entry in history) or "-"
    reasons = [_field(entry, "reason") for entry in history if _field(entry, "node") == "hitl"]
    return {
        "ticket": ticket_id,
        "category": category,
        "route": route,
        "hitl_reason": " | ".join(reasons) if reasons else "-",
        "went_to_hitl": bool(reasons),
        "escalated": bool(values.get("escalated")),
        "sent": bool(values.get("sent")),
    }


def summary_percents(rows: list[SummaryRow]) -> dict[str, float]:
    """Shares of the batch. A ticket with no HITL counts as handled without a human."""
    total = len(rows)
    if total == 0:
        return {"without_human": 0.0, "hitl": 0.0, "escalated": 0.0}
    without = sum(1 for row in rows if not row["went_to_hitl"])
    hitl = sum(1 for row in rows if row["went_to_hitl"])
    escalated = sum(1 for row in rows if row["escalated"])
    return {
        "without_human": 100.0 * without / total,
        "hitl": 100.0 * hitl / total,
        "escalated": 100.0 * escalated / total,
    }


def format_summary(rows: list[SummaryRow], *, totals: bool) -> str:
    headers = ["ticket", "category", "route", "hitl", "escalated", "sent"]
    display = [
        [
            row["ticket"],
            row["category"],
            row["route"],
            row["hitl_reason"],
            "yes" if row["escalated"] else "no",
            "yes" if row["sent"] else "no",
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for cells in display:
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    lines = [line(headers), line(["-" * width for width in widths])]
    lines.extend(line(cells) for cells in display)
    if totals:
        stats = summary_percents(rows)
        lines.append(f"handled without a human: {stats['without_human']:.1f}%")
        lines.append(f"sent to HITL: {stats['hitl']:.1f}%")
        lines.append(f"escalated: {stats['escalated']:.1f}%")
    return "\n".join(lines)


def format_versions(store: SqliteStore) -> str:
    """Procedure versions, newest first. The current one is marked."""
    from triage_agent.memory.procedural import list_versions

    current = get_current(store)
    current_version = None if current is None else int(current["version"])
    rows = list_versions(store)
    if not rows:
        return "No procedure versions."
    lines: list[str] = []
    for row in rows:
        version = int(row["version"])
        mark = " current" if version == current_version else ""
        note = str(row.get("note", ""))
        lines.append(f"v{version}{mark}  {note}")
    return "\n".join(lines)
