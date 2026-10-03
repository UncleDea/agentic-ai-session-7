"""HITL pause, resume, and memory learning. The model is stubbed unless marked live."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from triage_agent.config import Settings, get_settings
from triage_agent.graph import (
    build_graph,
    list_paused_runs,
    open_checkpointer,
    thread_config,
)
from triage_agent.memory.episodic import get_episode
from triage_agent.memory.procedural import get_current, list_versions, rollback, seed_procedures
from triage_agent.memory.semantic import seed_articles
from triage_agent.memory.store import FAKE_DIMS, FAKE_MODEL, fake_embeddings, open_store
from triage_agent.nodes.hitl import apply_decision, hitl
from triage_agent.nodes.responder import responder
from triage_agent.state import (
    Category,
    Channel,
    Draft,
    Finding,
    Investigation,
    Priority,
    Review,
    ReviewStatus,
    RouteEntry,
    Sentiment,
    Ticket,
    TriageResult,
    initial_state,
)

_DATA = Path(__file__).resolve().parents[1] / "data"
_RULE = "Don't repeat the customer's email address in the reply."
_PII_REPLY = (
    "Hola Jules, confirma jules.vega@example.com o llama al 415-555-0134. "
    "El pedido 88421 de $49.00 sigue activo."
)


class _DraftModel:
    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return Draft(reply="A clearer update with no new promise.", confidence=0.95)


@pytest.fixture
def store():
    opened = open_store(
        ":memory:",
        fake_embeddings(),
        dims=FAKE_DIMS,
        model=FAKE_MODEL,
    )
    seed_procedures(opened)
    try:
        yield opened
    finally:
        opened.conn.close()


def _ticket(ticket_id: str = "T-001", customer_id: str = "C-1001") -> Ticket:
    return Ticket(
        id=ticket_id,
        customer_id=customer_id,
        channel=Channel.email,
        subject="Charged twice",
        body="Please reverse the extra charge.",
        created_at=datetime(2026, 3, 3, 15, 4, tzinfo=timezone.utc),
    )


def _ready(
    reply: str = "Thanks for writing. We are looking into this.",
    ticket: Ticket | None = None,
    *,
    escalates: bool = False,
    flagged_claims: list[str] | None = None,
) -> dict:
    state = initial_state(ticket or _ticket())
    state["triage"] = TriageResult(
        category=Category.billing,
        priority=Priority.P1,
        sentiment=Sentiment.neutral,
        language="en",
        confidence=0.9,
        needs_human=False,
    )
    state["investigation"] = Investigation(
        findings=[Finding(fact="invoice 42 is open", source="orders")],
    )
    state["draft"] = Draft(
        reply=reply,
        confidence=0.95,
        escalates=escalates,
        flagged_claims=list(flagged_claims or []),
    )
    return state


def _compile(path, store, model=None, settings: Settings | None = None):
    root = Path(path).parent
    active = (settings or Settings()).model_copy(
        update={
            "outbox_dir": str(root / "outbox"),
            "escalations_dir": str(root / "escalations"),
        }
    )
    saver = open_checkpointer(str(path))
    graph = build_graph(
        model=model,
        settings=active,
        store=store,
        checkpointer=saver,
    )
    return saver, graph


def _pause(graph, state, config):
    graph.invoke(state, config, durability="sync")
    return graph.get_state(config)


def test_missing_ticket_is_an_error() -> None:
    assert hitl({})["errors"] == ["hitl: state has no ticket"]


def test_pause_shows_the_review_payload(store, tmp_path) -> None:
    saver, graph = _compile(tmp_path / "cp.sqlite", store)
    state = _ready(
        reply="The extra charge was $99.00.",
        escalates=True,
        flagged_claims=["$99.00"],
    )
    config = thread_config("T-001", "payload")
    try:
        paused = _pause(graph, state, config)
    finally:
        saver.conn.close()
    assert paused.next == ("hitl",)
    payload = paused.interrupts[0].value
    assert payload["ticket"]["id"] == "T-001"
    assert payload["triage"]["priority"] == "P1"
    assert payload["findings"] == [{"fact": "invoice 42 is open", "source": "orders"}]
    assert payload["draft"] == "The extra charge was $99.00."
    assert payload["flagged_claims"] == ["$99.00"]
    assert payload["escalates"] is True
    assert "priority P1" in payload["reasons"]
    assert "escalation promised" in payload["reasons"]


def test_approve_saves_a_redacted_episode(store, tmp_path) -> None:
    saver, graph = _compile(tmp_path / "cp.sqlite", store)
    config = thread_config("T-001", "approve")
    try:
        _pause(graph, _ready(), config)
        graph.invoke(Command(resume={"action": "approve"}), config, durability="sync")
        finished = graph.get_state(config)
    finally:
        saver.conn.close()
    assert finished.next == ()
    assert finished.values["review"].status is ReviewStatus.approved
    assert finished.values["sent"] is True
    episode = get_episode(store, "EP-T-001")
    assert episode is not None
    assert episode["approved_reply"] == "Thanks for writing. We are looking into this."
    assert get_current(store)["version"] == 1


def test_edit_saves_the_new_text(store, tmp_path) -> None:
    saver, graph = _compile(tmp_path / "cp.sqlite", store)
    config = thread_config("T-001", "edit")
    edited = "Here is the reply the reviewer wrote."
    try:
        _pause(graph, _ready(), config)
        graph.invoke(
            Command(resume={"action": "edit", "text": edited}),
            config,
            durability="sync",
        )
        finished = graph.get_state(config)
    finally:
        saver.conn.close()
    review = finished.values["review"]
    assert review.status is ReviewStatus.edited
    assert review.edited_text == edited
    episode = get_episode(store, "EP-T-001")
    assert episode is not None
    assert episode["approved_reply"] == edited
    assert "Thanks for writing" not in episode["approved_reply"]


def test_reject_does_not_write_memory(store, tmp_path) -> None:
    saver, graph = _compile(tmp_path / "cp.sqlite", store, model=_DraftModel())
    config = thread_config("T-001", "reject")
    try:
        _pause(graph, _ready(), config)
        graph.invoke(
            Command(resume={"action": "reject", "feedback": "be specific"}),
            config,
            durability="sync",
        )
        paused = graph.get_state(config)
    finally:
        saver.conn.close()
    reasons = [entry.reason for entry in paused.values["route_history"]]
    assert "review rejected; revision" in reasons
    assert get_episode(store, "EP-T-001") is None
    assert get_current(store)["version"] == 1


def test_rule_shaped_feedback_is_not_inferred(store) -> None:
    state = _ready()
    apply_decision(
        state,
        {"action": "reject", "feedback": "never promise a delivery date"},
        store,
    )
    assert get_episode(store, "EP-T-001") is None
    assert get_current(store)["version"] == 1
    assert [row["version"] for row in list_versions(store)] == [1]


def test_save_as_rule_appends_a_version_and_rollback_restores_it(
    store, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="triage_agent.hitl")
    saver, graph = _compile(tmp_path / "cp.sqlite", store, model=_DraftModel())
    config = thread_config("T-001", "rule")
    try:
        _pause(graph, _ready(), config)
        graph.invoke(
            Command(
                resume={
                    "action": "reject",
                    "feedback": _RULE,
                    "save_as_rule": True,
                }
            ),
            config,
            durability="sync",
        )
    finally:
        saver.conn.close()
    current = get_current(store)
    assert current is not None
    assert current["version"] == 2
    assert current["text"].endswith(_RULE)
    assert current["note"] == f"hitl ticket_id=T-001 action=reject rule={_RULE}"
    assert "decision=procedure v2" in caplog.text
    assert get_episode(store, "EP-T-001") is None
    restored = rollback(store, 1)
    assert restored["version"] == 1
    assert get_current(store)["version"] == 1
    assert _RULE not in get_current(store)["text"]
    assert any(row["version"] == 2 and _RULE in row["text"] for row in list_versions(store))


def test_learning_is_safe_to_run_twice(store) -> None:
    state = _ready()
    decision = {
        "action": "edit",
        "text": "A short approved reply.",
        "feedback": _RULE,
        "save_as_rule": True,
    }
    apply_decision(state, decision, store)
    apply_decision(state, decision, store)
    episode = get_episode(store, "EP-T-001")
    assert episode is not None
    assert episode["approved_reply"] == "A short approved reply."
    current = get_current(store)
    assert current is not None
    assert current["version"] == 2
    assert current["text"].count(_RULE) == 1
    assert [row["version"] for row in list_versions(store)] == [2, 1]


def test_approving_t011_redacts_the_episode(store, tmp_path) -> None:
    ticket = Ticket(
        id="T-011",
        customer_id="C-1011",
        channel=Channel.email,
        subject="No puedo iniciar sesión",
        body=(
            "Hola, desde ayer no puedo entrar a mi cuenta. "
            "Me dice que la contraseña es incorrecta y ya la restablecí dos veces."
        ),
        created_at=datetime(2026, 3, 9, 10, 30, tzinfo=timezone.utc),
    )
    state = initial_state(ticket)
    state["triage"] = TriageResult(
        category=Category.account,
        priority=Priority.P1,
        sentiment=Sentiment.neutral,
        language="es",
        confidence=0.9,
        needs_human=False,
    )
    state["investigation"] = Investigation(
        findings=[
            Finding(
                fact="Jules Vega email jules.vega@example.com plan plus status active.",
                source="customer:C-1011",
            ),
            Finding(
                fact="Order 88421 status charged amount 49.00.",
                source="order:88421",
            ),
        ],
    )
    state["draft"] = Draft(reply=_PII_REPLY, confidence=0.95)
    saver, graph = _compile(tmp_path / "cp.sqlite", store)
    config = thread_config("T-011", "redact")
    try:
        _pause(graph, state, config)
        graph.invoke(Command(resume={"action": "approve"}), config, durability="sync")
    finally:
        saver.conn.close()
    episode = get_episode(store, "EP-T-011")
    assert episode is not None
    stored = json.dumps(episode)
    assert "jules.vega@example.com" not in stored.casefold()
    assert "jules" not in stored.casefold()
    reply = episode["approved_reply"]
    assert "[EMAIL]" in reply
    assert "[CUSTOMER_NAME]" in reply
    assert "[PHONE]" in reply
    assert "[ORDER_ID]" in reply
    assert "[AMOUNT]" in reply
    assert "88421" not in reply
    assert "49.00" not in reply
    assert "415-555-0134" not in reply


def test_reloading_a_paused_run_emits_no_unregistered_type_warning(
    store, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "cp.sqlite"
    saver, graph = _compile(path, store)
    config = thread_config("T-001", "serde")
    state = _ready()
    state["review"] = Review(status=ReviewStatus.pending)
    try:
        _pause(graph, state, config)
    finally:
        saver.conn.close()

    from langgraph.checkpoint.serde.jsonplus import _warned_unregistered_types

    _warned_unregistered_types.clear()
    caplog.clear()
    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    reopened = open_checkpointer(str(path))
    try:
        graph2 = build_graph(settings=Settings(), store=store, checkpointer=reopened)
        snapshot = graph2.get_state(config)
    finally:
        reopened.conn.close()

    assert "Deserializing unregistered type" not in caplog.text
    values = snapshot.values
    assert snapshot.next == ("hitl",)
    assert isinstance(values["ticket"], Ticket)
    assert values["ticket"].channel is Channel.email
    assert isinstance(values["triage"], TriageResult)
    assert values["triage"].category is Category.billing
    assert values["triage"].priority is Priority.P1
    assert values["triage"].sentiment is Sentiment.neutral
    assert isinstance(values["investigation"], Investigation)
    assert isinstance(values["draft"], Draft)
    assert isinstance(values["review"], Review)
    assert values["review"].status is ReviewStatus.pending
    assert values["route_history"]
    assert all(isinstance(entry, RouteEntry) for entry in values["route_history"])


def test_resume_after_reopening_the_checkpointer(store, tmp_path) -> None:
    path = tmp_path / "cp.sqlite"
    saver, graph = _compile(path, store)
    config = thread_config("T-001", "reopen")
    _pause(graph, _ready(), config)
    payload = graph.get_state(config).interrupts[0].value
    saver.conn.close()

    reopened = open_checkpointer(str(path))
    try:
        graph2 = build_graph(settings=Settings(), store=store, checkpointer=reopened)
        paused = list_paused_runs(graph2)
        assert paused == [
            {
                "thread_id": "T-001-reopen",
                "ticket_id": "T-001",
                "reasons": ["priority P1"],
            }
        ]
        snapshot = graph2.get_state(config)
        assert snapshot.next == ("hitl",)
        assert snapshot.interrupts[0].value == payload
        graph2.invoke(Command(resume={"action": "approve"}), config, durability="sync")
        finished = graph2.get_state(config)
    finally:
        reopened.conn.close()
    assert finished.next == ()
    assert finished.values["sent"] is True
    assert get_episode(store, "EP-T-001") is not None


def test_same_ticket_twice_gives_independent_runs(store, tmp_path) -> None:
    saver, graph = _compile(tmp_path / "cp.sqlite", store)
    state = _ready()
    first = thread_config("T-001")
    second = thread_config("T-001")
    first_id = first["configurable"]["thread_id"]
    second_id = second["configurable"]["thread_id"]
    try:
        assert first_id != second_id
        assert first_id.startswith("T-001-")
        assert second_id.startswith("T-001-")
        _pause(graph, state, first)
        graph.invoke(Command(resume={"action": "approve"}), first, durability="sync")
        assert graph.get_state(first).values["sent"] is True
        assert graph.get_state(first).next == ()
        _pause(graph, state, second)
        paused = graph.get_state(second)
        assert paused.next == ("hitl",)
        assert paused.values["sent"] is False
        assert paused.values.get("review") is None
        listed = list_paused_runs(graph)
    finally:
        saver.conn.close()
    assert [row["thread_id"] for row in listed] == [second_id]
    assert listed[0]["ticket_id"] == "T-001"
    assert listed[0]["reasons"] == ["priority P1"]


def test_edit_without_text_is_rejected(store, tmp_path) -> None:
    saver, graph = _compile(tmp_path / "cp.sqlite", store)
    config = thread_config("T-001", "bad")
    try:
        _pause(graph, _ready(), config)
        with pytest.raises(ValidationError):
            graph.invoke(
                Command(resume={"action": "edit"}),
                config,
                durability="sync",
            )
    finally:
        saver.conn.close()


def _t011_state(ticket_id: str) -> dict:
    samples = json.loads((_DATA / "sample_tickets.json").read_text(encoding="utf-8"))
    record = next(row for row in samples if row["id"] == "T-011")
    ticket = Ticket.model_validate(
        {key: value for key, value in record.items() if key != "expected"}
    )
    ticket = ticket.model_copy(update={"id": ticket_id})
    customers = json.loads((_DATA / "customers.json").read_text(encoding="utf-8"))
    customer = next(row for row in customers if row["customer_id"] == ticket.customer_id)
    state = initial_state(ticket)
    state["triage"] = TriageResult(
        category=Category.account,
        priority=Priority.P3,
        sentiment=Sentiment.neutral,
        language="es",
        confidence=0.9,
        needs_human=False,
    )
    state["investigation"] = Investigation(
        findings=[
            Finding(
                fact=(
                    f"{customer['name']} email {customer['email']} "
                    f"plan {customer['plan']} status {customer['status']}."
                ),
                source=customer["source"],
            )
        ],
        kb_article_ids=["KB-004"],
    )
    return state


@pytest.mark.live
def test_live_rule_drops_the_email_from_a_similar_ticket(store, tmp_path) -> None:
    settings = get_settings()
    key_name = "OPENAI_API_KEY" if settings.model_provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(key_name):
        pytest.skip(f"{key_name} is not set")
    seed_articles(store)
    first = responder(_t011_state("T-011"), settings=settings, store=store)
    draft = first.get("draft")
    assert isinstance(draft, Draft), first.get("errors")
    print(f"T-011 reply: {draft.reply}")
    state = _t011_state("T-011")
    state["draft"] = draft
    review_settings = settings.model_copy(
        update={
            "hitl_categories": [*settings.hitl_categories, Category.account],
            "max_revisions": 0,
        }
    )
    saver, graph = _compile(tmp_path / "cp.sqlite", store, settings=review_settings)
    config = thread_config("T-011", "live")
    try:
        paused = _pause(graph, state, config)
        assert paused.next == ("hitl",)
        graph.invoke(
            Command(
                resume={
                    "action": "reject",
                    "feedback": _RULE,
                    "save_as_rule": True,
                }
            ),
            config,
            durability="sync",
        )
    finally:
        saver.conn.close()
    current = get_current(store)
    assert current is not None
    assert _RULE in current["text"]
    print(f"procedure v{current['version']}: {current['text']}")
    second = responder(_t011_state("T-011B"), settings=settings, store=store)
    revised = second.get("draft")
    assert isinstance(revised, Draft), second.get("errors")
    print(f"T-011B reply: {revised.reply}")
    assert "jules.vega@example.com" not in revised.reply.casefold()
