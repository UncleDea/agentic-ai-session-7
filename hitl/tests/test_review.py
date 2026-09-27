"""Reason text for the human review payload."""

from datetime import datetime, timezone

from triage_agent.schemas import TicketState
from hitl.review import review_reason

RECEIVED_AT = datetime(2026, 5, 10, 9, 14, 22, tzinfo=timezone.utc)


def _state(severity: str, hint: str, reasoning: str) -> TicketState:
    return TicketState(
        ticket_id="TKT-1",
        raw_text="I was charged twice for Pro this month.",
        customer_id="cust-1",
        channel="email",
        received_at=RECEIVED_AT,
        category="billing",
        severity=severity,
        confidence=0.4 if hint == "hitl" and severity == "medium" else 0.9,
        triage_reasoning=reasoning,
        status="triaged",
        next_node_hint=hint,
    )


def test_high_severity_names_the_severity():
    reason = review_reason(_state("high", "hitl", "Duplicate charge."))
    assert reason == "Severity is high. Duplicate charge."


def test_low_confidence_names_the_triage_request():
    reason = review_reason(_state("medium", "hitl", "Low-confidence fallback."))
    assert reason == "Triage asked for a human review. Low-confidence fallback."
