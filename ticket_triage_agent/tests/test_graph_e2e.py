"""Offline end-to-end runs. The model and embeddings are fakes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from triage_agent.cli import (
    FAKE_STORE_MESSAGE,
    SummaryRow,
    check_startup,
    example_setting_names,
    format_summary,
    missing_setting_names,
    missing_settings_line,
    summary_percents,
)
from triage_agent.config import Settings
from triage_agent.graph import build_graph, open_checkpointer, thread_config
from triage_agent.memory.episodic import seed_episodes
from triage_agent.memory.procedural import seed_procedures
from triage_agent.memory.semantic import seed_articles
from triage_agent.memory.store import (
    FAKE_DIMS,
    FAKE_MODEL,
    fake_embeddings,
    open_store,
)
from triage_agent.nodes.triage import example_label, load_sample_tickets
from triage_agent.state import (
    Category,
    Draft,
    Investigation,
    Priority,
    Sentiment,
    Ticket,
    TriageResult,
    initial_state,
)
from triage_agent.tools.sender import EscalationRecord, OutboxSender, list_open

_BLAND = "Thanks for writing. We are looking into this."
_NO_HITL = {"T-002", "T-003", "T-004", "T-006", "T-008", "T-011"}
_HITL = {
    "T-001": "priority P1",
    "T-005": "category refund",
    "T-009": "triage confidence below threshold; needs_human",
    "T-010": "triage confidence below threshold; needs_human",
    "T-012": "category refund",
    "T-013": "priority P1; category security; needs_human",
    "T-014": "priority P1; sentiment negative",
    "T-015": "priority P1; sentiment negative; category legal; needs_human",
}
_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


class OfflineModel:
    """Triage from the sample labels. No tools, no invented facts, no escalation."""

    def __init__(self, labels: dict[str, TriageResult]) -> None:
        self.labels = labels
        self.schema = None

    def bind_tools(self, tools, **kwargs):
        self.schema = None
        return self

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    def invoke(self, messages):
        if self.schema is None:
            return AIMessage(content="done")
        if self.schema is TriageResult:
            return self.labels[_ticket_id(messages)]
        if self.schema is Investigation:
            return Investigation()
        if self.schema is Draft:
            return Draft(reply=_BLAND, confidence=0.95)
        raise AssertionError(self.schema)


def _ticket_id(messages) -> str:
    found = ""
    for message in messages:
        content = message.content if isinstance(message.content, str) else ""
        for line in content.splitlines():
            if line.startswith("id: "):
                found = line.removeprefix("id: ").strip()
    if not found:
        raise AssertionError("no ticket id")
    return found


def _label(record: dict) -> TriageResult:
    expected = record.get("expected") or {}
    if "category" not in expected:
        return TriageResult(
            category=Category.other,
            priority=Priority.P4,
            sentiment=Sentiment.neutral,
            language="en",
            confidence=0.2,
            needs_human=True,
            needs_human_reason="empty or unintelligible",
        )
    return TriageResult.model_validate(example_label(record))


def _ticket(record: dict) -> Ticket:
    payload = {key: value for key, value in record.items() if key != "expected"}
    return Ticket.model_validate(payload)


def _nodes(values: dict) -> list[str]:
    nodes = []
    for entry in values.get("route_history") or []:
        nodes.append(entry.node if not isinstance(entry, dict) else entry["node"])
    return nodes


def _hitl_reason(values: dict) -> str:
    reasons = []
    for entry in values.get("route_history") or []:
        node = entry.node if not isinstance(entry, dict) else entry["node"]
        if node == "hitl":
            reason = entry.reason if not isinstance(entry, dict) else entry["reason"]
            reasons.append(reason)
    return " | ".join(reasons)


def _row(**overrides) -> SummaryRow:
    base: SummaryRow = {
        "ticket": "T",
        "category": "billing",
        "route": "triage",
        "hitl_reason": "-",
        "went_to_hitl": False,
        "escalated": False,
        "sent": True,
    }
    base.update(overrides)
    return base


def test_sample_tickets_finish(tmp_path: Path) -> None:
    samples = load_sample_tickets()
    store = open_store(
        str(tmp_path / "memory.sqlite"),
        fake_embeddings(),
        dims=FAKE_DIMS,
        model=FAKE_MODEL,
    )
    seed_articles(store)
    seed_episodes(store)
    seed_procedures(store)
    settings = Settings(
        outbox_dir=str(tmp_path / "outbox"),
        escalations_dir=str(tmp_path / "escalations"),
    )
    saver = open_checkpointer(str(tmp_path / "checkpoints.sqlite"))
    graph = build_graph(
        model=OfflineModel({record["id"]: _label(record) for record in samples}),
        settings=settings,
        store=store,
        checkpointer=saver,
    )
    outbox = OutboxSender(tmp_path / "outbox")
    failures: list[str] = []
    try:
        for record in samples:
            ticket_id = record["id"]
            config = thread_config(ticket_id, "e2e")
            state = initial_state(_ticket(record))
            graph.invoke(state, config, durability="sync")
            snapshot = graph.get_state(config)
            if snapshot.next == ("hitl",):
                graph.invoke(
                    Command(resume={"action": "approve"}),
                    config,
                    durability="sync",
                )
                snapshot = graph.get_state(config)
            values = dict(snapshot.values)
            nodes = _nodes(values)
            problems: list[str] = []
            if snapshot.next:
                problems.append(f"still at {snapshot.next}")
            if values.get("errors"):
                problems.append(f"errors={values['errors']}")
            if values.get("escalated"):
                problems.append("escalated")
            if ticket_id == "T-007":
                ended = any(
                    (entry.reason if not isinstance(entry, dict) else entry["reason"])
                    == "spam; no reply"
                    for entry in values.get("route_history") or []
                )
                if values.get("sent"):
                    problems.append("spam was sent")
                if "end" not in nodes or not ended:
                    problems.append(f"route={nodes}")
                if "investigator" in nodes or "send" in nodes:
                    problems.append(f"spam route={nodes}")
                if outbox.get(f"{ticket_id}-e2e") is not None:
                    problems.append("spam wrote an outbox file")
            else:
                if not values.get("sent"):
                    problems.append("not sent")
                if outbox.get(f"{ticket_id}-e2e") is None:
                    problems.append("missing outbox file")
                if ticket_id in _HITL:
                    if _hitl_reason(values) != _HITL[ticket_id]:
                        problems.append(f"hitl={_hitl_reason(values)!r}")
                elif ticket_id in _NO_HITL:
                    if "hitl" in nodes:
                        problems.append(f"unexpected hitl={_hitl_reason(values)!r}")
                else:
                    problems.append("missing expected outcome")
                if ticket_id == "T-006":
                    if "investigator" in nodes:
                        problems.append("feedback went to the investigator")
                elif "investigator" not in nodes:
                    problems.append(f"no investigator in {nodes}")
            if problems:
                failures.append(f"{ticket_id}: {'; '.join(problems)}")
    finally:
        saver.conn.close()
        store.conn.close()
    assert failures == []


def test_summary_percentages() -> None:
    rows = [
        _row(ticket="T-002"),
        _row(ticket="T-003"),
        _row(ticket="T-007", category="spam", sent=False),
        _row(ticket="T-001", went_to_hitl=True, hitl_reason="priority P1"),
        _row(
            ticket="T-015",
            went_to_hitl=True,
            hitl_reason="revision cap",
            escalated=True,
            sent=False,
        ),
    ]
    assert summary_percents(rows) == {
        "without_human": 60.0,
        "hitl": 40.0,
        "escalated": 20.0,
    }
    text = format_summary(rows, totals=True)
    assert "handled without a human: 60.0%" in text
    assert "sent to HITL: 40.0%" in text
    assert "escalated: 20.0%" in text


def test_example_file_skips_commented_secrets() -> None:
    names = example_setting_names(_EXAMPLE)
    assert "EMBEDDING_MODEL" in names
    assert "OPENAI_API_KEY" not in names
    assert "ANTHROPIC_API_KEY" not in names
    assert "LANGSMITH_API_KEY" not in names


def test_missing_setting_names_are_names_only(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text(
        "VISIBLE=hello\nHIDDEN=super-secret\n# COMMENTED=nope\nBLANK=\n",
        encoding="utf-8",
    )
    names = example_setting_names(example)
    assert names == ["VISIBLE", "HIDDEN", "BLANK"]
    missing = missing_setting_names(names, {"VISIBLE": "hello", "BLANK": "  "})
    assert missing == ["HIDDEN", "BLANK"]
    line = missing_settings_line(missing)
    assert line == "Missing settings: HIDDEN, BLANK"
    assert "super-secret" not in line
    assert "hello" not in line
    assert "COMMENTED" not in line


def test_missing_store_does_not_create_a_file(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text("MEMORY_DB_PATH=data/memory.sqlite\n", encoding="utf-8")
    missing = tmp_path / "missing.sqlite"
    settings = Settings(
        memory_db_path=str(missing),
        embedding_model="text-embedding-3-small",
        embedding_dims=1536,
    )
    with pytest.raises(SystemExit) as exc:
        check_startup(
            settings,
            example_path=example,
            environ={"MEMORY_DB_PATH": "data/memory.sqlite"},
        )
    assert "seed_memory.py" in str(exc.value)
    assert not missing.exists()


def test_fake_stamp_stops(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    store = open_store(str(path), fake_embeddings(), dims=FAKE_DIMS, model=FAKE_MODEL)
    store.conn.close()
    example = tmp_path / ".env.example"
    example.write_text(
        "EMBEDDING_MODEL=text-embedding-3-small\nVISIBLE=do-not-print-this\n",
        encoding="utf-8",
    )
    settings = Settings(
        memory_db_path=str(path),
        embedding_model="text-embedding-3-small",
        embedding_dims=1536,
    )
    with pytest.raises(SystemExit) as exc:
        check_startup(
            settings,
            example_path=example,
            environ={
                "EMBEDDING_MODEL": "text-embedding-3-small",
                "VISIBLE": "do-not-print-this",
            },
        )
    text = str(exc.value)
    assert "Embedding model: fake" in text
    assert FAKE_STORE_MESSAGE in text
    assert "do-not-print-this" not in text


def test_stamp_mismatch_names_the_built_model(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    store = open_store(str(path), fake_embeddings(8), dims=8, model="other-model")
    store.conn.close()
    example = tmp_path / ".env.example"
    example.write_text("EMBEDDING_MODEL=text-embedding-3-small\n", encoding="utf-8")
    settings = Settings(
        memory_db_path=str(path),
        embedding_model="text-embedding-3-small",
        embedding_dims=1536,
    )
    with pytest.raises(SystemExit) as exc:
        check_startup(
            settings,
            example_path=example,
            environ={"EMBEDDING_MODEL": "text-embedding-3-small"},
        )
    text = str(exc.value)
    assert "Embedding model: other-model" in text
    assert "was built with other-model" in text


def test_unseeded_store_names_the_seed_script(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    store = open_store(
        str(path),
        fake_embeddings(1536),
        dims=1536,
        model="text-embedding-3-small",
    )
    store.conn.close()
    example = tmp_path / ".env.example"
    example.write_text("EMBEDDING_MODEL=text-embedding-3-small\n", encoding="utf-8")
    settings = Settings(
        memory_db_path=str(path),
        embedding_model="text-embedding-3-small",
        embedding_dims=1536,
    )

    def opener(active: Settings):
        return open_store(
            active.memory_db_path or "",
            fake_embeddings(1536),
            dims=1536,
            model="text-embedding-3-small",
        )

    with pytest.raises(SystemExit) as exc:
        check_startup(
            settings,
            example_path=example,
            environ={"EMBEDDING_MODEL": "text-embedding-3-small"},
            opener=opener,
        )
    text = str(exc.value)
    assert "is not seeded" in text
    assert "seed_memory.py" in text


def test_list_open_skips_unreadable(tmp_path: Path) -> None:
    record = EscalationRecord(
        ticket_id="T-001",
        thread_id="T-001-e2e",
        reasons=["escalation promised"],
        reply="A specialist will follow up.",
        created_at=datetime(2026, 3, 12, tzinfo=timezone.utc),
    )
    (tmp_path / "open.json").write_text(record.model_dump_json(), encoding="utf-8")
    closed = record.model_dump_json().replace('"open"', '"closed"', 1)
    (tmp_path / "closed.json").write_text(closed, encoding="utf-8")
    records, skipped = list_open(tmp_path)
    assert [row.thread_id for row in records] == ["T-001-e2e"]
    assert skipped == ["closed.json"]
