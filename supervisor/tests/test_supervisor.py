"""Supervisor tests. Triage is stubbed, so these do not call an API."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from triage_agent import TicketInput, TicketState, run_triage
from triage_agent.schemas import TriageClassification
from triage_agent.tests.test_agent import FakeModel
from hitl.schemas import HumanDecision
from responder.schemas import ResponseDraft
from send.schemas import Delivery
from supervisor import SupervisorResult, resume_review, run_supervisor
from supervisor.routing import choose_next_node

RECEIVED_AT = datetime(2026, 5, 10, 9, 14, 22, tzinfo=timezone.utc)


def _state(
    *,
    severity: str = "low",
    hint: str = "investigator",
    category: str = "billing",
    confidence: float = 0.9,
) -> TicketState:
    return TicketState(
        ticket_id="TKT-1",
        raw_text="I was charged twice for Pro this month.",
        customer_id="cust-1",
        channel="email",
        received_at=RECEIVED_AT,
        category=category,
        severity=severity,
        confidence=confidence,
        triage_reasoning="test",
        status="triaged",
        next_node_hint=hint,
    )


def _ticket() -> TicketInput:
    return TicketInput(
        ticket_id="TKT-1",
        raw_text="I was charged twice for Pro this month.",
        customer_id="cust-1",
        channel="email",
        received_at=RECEIVED_AT,
    )


def test_high_severity_goes_to_hitl_even_if_hint_says_investigator():
    assert choose_next_node(_state(severity="high", hint="investigator")) == "hitl"


def test_critical_severity_goes_to_hitl():
    assert choose_next_node(_state(severity="critical", hint="investigator")) == "hitl"


def test_fallback_hint_goes_to_hitl():
    assert choose_next_node(_state(severity="medium", hint="hitl", category="other", confidence=0.3)) == "hitl"


def test_ordinary_ticket_goes_to_investigator():
    assert choose_next_node(_state(severity="low", hint="investigator")) == "investigator"


def test_graph_routes_high_severity_to_hitl():
    seen = {}

    def triage(ticket: TicketInput) -> TicketState:
        seen["ticket_id"] = ticket.ticket_id
        return _state(severity="high", hint="hitl")

    result = run_supervisor(_ticket(), triage_fn=triage)
    assert seen["ticket_id"] == "TKT-1"
    assert isinstance(result, SupervisorResult)
    assert result.routed_to == "hitl"
    assert result.investigation is None
    assert result.draft is None
    assert result.review is not None
    assert result.review.status == "pending"
    assert "Severity is high" in result.review.reason
    assert result.ticket_state.category == "billing"
    assert result.ticket_state.status == "triaged"


def test_graph_routes_low_severity_through_investigator_to_responder():
    def triage(ticket: TicketInput) -> TicketState:
        return _state(severity="low", hint="investigator")

    def respond(ticket: TicketState, investigation) -> ResponseDraft:
        return ResponseDraft(
            status="drafted",
            subject="Re: TKT-1",
            body="We found the billing records and a specialist can explain the charge.",
            domain="support",
            memory_used=["semantic"],
            needs_review=False,
        )

    result = run_supervisor(_ticket(), triage_fn=triage, respond_fn=respond, send_fn=_sent)
    assert result.routed_to == "send"
    assert result.delivery is not None
    assert result.delivery.status == "sent"
    assert result.delivery.recipient == "cust-1"
    assert result.investigation is not None
    assert result.investigation.status == "investigated"
    assert result.investigation.service == "billing-service"
    assert result.draft is not None
    assert result.draft.status == "drafted"
    assert "billing records" in result.draft.body


def test_supervisor_log_is_json(capsys):
    def triage(ticket: TicketInput) -> TicketState:
        return _state(severity="high", hint="hitl")

    run_supervisor(_ticket(), triage_fn=triage)
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload == {
        "ticket_id": "TKT-1",
        "severity": "high",
        "next_node_hint": "hitl",
        "routed_to": "hitl",
    }


def test_real_triage_double_charge_is_routed_to_hitl():
    model = FakeModel(
        TriageClassification(
            category="billing",
            severity="high",
            confidence=0.95,
            reasoning="Duplicate charge.",
        )
    )

    def triage(ticket: TicketInput) -> TicketState:
        return run_triage(ticket, model=model)

    result = run_supervisor(_ticket(), triage_fn=triage)
    assert result.ticket_state.category == "billing"
    assert result.ticket_state.severity == "high"
    assert result.routed_to == "hitl"
    assert result.investigation is None
    assert result.draft is None


def _sent(ticket: TicketState, draft: ResponseDraft) -> Delivery:
    return Delivery(
        status="sent",
        outbound_id="OUT-TEST0001",
        recipient=ticket.customer_id,
        subject=draft.subject,
        body=draft.body,
        sent_at="2026-05-10T12:00:00+00:00",
    )


def _draft() -> ResponseDraft:
    return ResponseDraft(
        status="drafted",
        subject="Re: TKT-1",
        body="Approved after review.",
        domain="support",
        memory_used=["semantic"],
        needs_review=False,
    )


def test_approved_review_continues_to_a_draft():
    from investigator.schemas import Investigation

    calls = {"investigate": 0}

    def triage(ticket: TicketInput) -> TicketState:
        return _state(severity="critical", hint="hitl")

    def investigate_ok(ticket: TicketState):
        calls["investigate"] += 1
        return Investigation(
            status="investigated",
            summary="Matched billing-service.",
            service="billing-service",
            evidence=[],
        )

    paused = run_supervisor(_ticket(), triage_fn=triage, investigate_fn=investigate_ok)
    assert paused.review is not None
    assert paused.review.status == "pending"
    assert calls["investigate"] == 0

    finished = resume_review(
        paused.thread_id,
        HumanDecision(action="approve", note="Charge looks duplicated."),
        triage_fn=triage,
        investigate_fn=investigate_ok,
        respond_fn=lambda ticket, investigation: _draft(),
        send_fn=_sent,
    )
    assert calls["investigate"] == 1
    assert finished.review.status == "approved"
    assert finished.review.note == "Charge looks duplicated."
    assert finished.routed_to == "send"
    assert finished.draft.body == "Approved after review."
    assert finished.delivery.status == "sent"


def test_rejected_review_stops():
    def triage(ticket: TicketInput) -> TicketState:
        return _state(severity="high", hint="hitl")

    def explode(*args, **kwargs):
        raise AssertionError("should not run after a rejection")

    paused = run_supervisor(_ticket(), triage_fn=triage, investigate_fn=explode, respond_fn=explode)
    finished = resume_review(
        paused.thread_id,
        {"action": "reject", "note": "Not our customer."},
        triage_fn=triage,
        investigate_fn=explode,
        respond_fn=explode,
    )
    assert finished.review.status == "rejected"
    assert finished.review.note == "Not our customer."
    assert finished.routed_to == "hitl"
    assert finished.investigation is None
    assert finished.draft is None


def test_review_hold_does_not_send():
    def triage(ticket: TicketInput) -> TicketState:
        return _state(severity="low", hint="investigator")

    def respond(ticket: TicketState, investigation) -> ResponseDraft:
        draft = _draft()
        return draft.model_copy(update={"needs_review": True, "body": "A specialist will review this."})

    def explode(*args, **kwargs):
        raise AssertionError("send should not run")

    result = run_supervisor(_ticket(), triage_fn=triage, respond_fn=respond, send_fn=explode)
    assert result.routed_to == "responder"
    assert result.delivery is not None
    assert result.delivery.status == "not_sent"
    assert "review" in result.delivery.reason


def test_invalid_ticket_raises():
    with pytest.raises(ValidationError):
        run_supervisor(
            {"raw_text": "hello", "channel": "email", "received_at": RECEIVED_AT},
            triage_fn=lambda ticket: (_ for _ in ()).throw(AssertionError("triage should not run")),
        )
