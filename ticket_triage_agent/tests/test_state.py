"""Offline validation for TicketState and the sample tickets."""

from __future__ import annotations

import json
import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from triage_agent.state import (
    Category,
    Channel,
    Draft,
    Priority,
    Review,
    ReviewStatus,
    RouteEntry,
    Sentiment,
    Ticket,
    TicketState,
    TriageResult,
    initial_state,
)

_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_tickets.json"


def _sample_records() -> list[dict]:
    records = json.loads(_SAMPLE_PATH.read_text(encoding="utf-8"))
    assert len(records) == 15
    return records


def _ticket_payload(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "expected"}


def _triage(**overrides) -> TriageResult:
    data = {
        "category": Category.billing,
        "priority": Priority.P3,
        "sentiment": Sentiment.neutral,
        "language": "en",
        "confidence": 0.8,
        "needs_human": False,
    }
    data.update(overrides)
    return TriageResult(**data)


def _ticket() -> Ticket:
    return Ticket(
        id="T-001",
        customer_id="C-1001",
        channel=Channel.email,
        subject="Charged twice",
        body="Please reverse the extra charge.",
        created_at=datetime(2026, 3, 3, 15, 4, tzinfo=timezone.utc),
    )


def test_ticket_round_trip() -> None:
    ticket = _ticket()
    assert Ticket.model_validate(ticket.model_dump()) == ticket


def test_unknown_category_rejected() -> None:
    with pytest.raises(ValidationError):
        _triage(category="nope")


def test_security_and_legal_are_categories() -> None:
    security = _triage(
        category=Category.security,
        needs_human=True,
        needs_human_reason="account takeover",
    )
    legal = _triage(
        category=Category.legal,
        needs_human=True,
        needs_human_reason="chargeback threat",
    )
    assert security.category is Category.security
    assert legal.category is Category.legal


def test_priority_must_be_p1_through_p4() -> None:
    with pytest.raises(ValidationError):
        _triage(priority="P5")
    assert _triage(priority=Priority.P1).priority is Priority.P1


def test_confidence_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        _triage(confidence=1.1)
    with pytest.raises(ValidationError):
        Draft(reply="Thanks for writing.", confidence=-0.01)


def test_review_status_enum() -> None:
    assert Review(status=ReviewStatus.approved).status is ReviewStatus.approved
    with pytest.raises(ValidationError):
        Review(status="maybe")


def test_missing_ticket_fields_rejected() -> None:
    payload = _ticket().model_dump()
    del payload["id"]
    with pytest.raises(ValidationError):
        Ticket.model_validate(payload)
    del payload["customer_id"]
    payload["id"] = "T-001"
    with pytest.raises(ValidationError):
        Ticket.model_validate(payload)


def test_empty_and_short_bodies_are_valid() -> None:
    created_at = datetime(2026, 3, 8, tzinfo=timezone.utc)
    empty = Ticket(
        id="T-009",
        customer_id="C-1009",
        channel=Channel.web,
        subject="Question",
        body="",
        created_at=created_at,
    )
    short = Ticket(
        id="T-010",
        customer_id="C-1010",
        channel=Channel.chat,
        subject="help",
        body="help",
        created_at=created_at,
    )
    assert empty.body == ""
    assert short.body == "help"


def test_needs_human_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        _triage(needs_human=True, needs_human_reason="  ")
    result = _triage(needs_human=True, needs_human_reason="refund over the limit")
    assert result.needs_human_reason == "refund over the limit"


def test_initial_state_is_unset() -> None:
    state = initial_state(_ticket())
    assert state["triage"] is None
    assert state["investigation"] is None
    assert state["draft"] is None
    assert state["review"] is None
    assert state["sent"] is False
    assert state["route_history"] == []
    assert state["errors"] == []
    assert state["iteration_count"] == 0
    assert state["escalated"] is False


def test_list_fields_accumulate_and_counts_overwrite() -> None:
    hints = get_type_hints(TicketState, include_extras=True)
    history_hint = hints["route_history"]
    errors_hint = hints["errors"]
    assert get_origin(history_hint) is Annotated
    assert get_args(history_hint)[1] is operator.add
    assert get_origin(errors_hint) is Annotated
    assert get_args(errors_hint)[1] is operator.add
    assert hints["iteration_count"] is int

    def first(state: TicketState) -> dict:
        return {
            "route_history": [RouteEntry(node="supervisor", reason="no triage yet")],
            "errors": ["lookup failed"],
            "iteration_count": 1,
        }

    def second(state: TicketState) -> dict:
        return {
            "route_history": [RouteEntry(node="triage", reason="classified")],
            "errors": ["draft failed"],
            "iteration_count": 2,
        }

    graph = StateGraph(TicketState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    result = graph.compile().invoke(initial_state(_ticket()))

    assert [entry.node for entry in result["route_history"]] == ["supervisor", "triage"]
    assert result["errors"] == ["lookup failed", "draft failed"]
    assert result["iteration_count"] == 2


def test_sample_tickets_parse() -> None:
    records = _sample_records()
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
    tickets = [Ticket.model_validate(_ticket_payload(record)) for record in records]
    assert len(tickets) == 15
    by_id = {record["id"]: record for record in records}
    assert by_id["T-013"]["expected"]["category"] == "security"
    assert by_id["T-013"]["expected"]["needs_human"] is True
    assert by_id["T-015"]["expected"]["category"] == "legal"
    assert by_id["T-015"]["expected"]["needs_human"] is True
    assert tickets[12].id == "T-013"
    assert tickets[14].id == "T-015"
