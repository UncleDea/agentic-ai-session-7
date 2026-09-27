"""Send tests write only to a temporary outbox."""

import json
from datetime import datetime, timezone
from pathlib import Path

from responder.schemas import ResponseDraft
from triage_agent.schemas import TicketState
from send.agent import run_send

RECEIVED_AT = datetime(2026, 5, 10, 9, 14, 22, tzinfo=timezone.utc)


def _ticket(customer_id: str | None = "priya.s@example.com") -> TicketState:
    return TicketState(
        ticket_id="TKT-NEW",
        raw_text="Authenticator codes loop back to the MFA page.",
        customer_id=customer_id,
        channel="email",
        received_at=RECEIVED_AT,
        category="account",
        severity="medium",
        confidence=0.9,
        triage_reasoning="test",
        status="triaged",
        next_node_hint="investigator",
    )


def _draft(*, needs_review: bool = False, body: str = "Please reopen your authenticator app.") -> ResponseDraft:
    return ResponseDraft(
        status="drafted",
        subject="Re: MFA loop",
        body=body,
        domain="support",
        memory_used=["procedural"],
        needs_review=needs_review,
    )


def test_send_appends_an_outbox_record(tmp_path: Path):
    outbox = tmp_path / "sent.jsonl"
    delivery = run_send(_ticket(), _draft(), outbox=outbox)
    assert delivery.status == "sent"
    assert delivery.outbound_id.startswith("OUT-")
    assert delivery.recipient == "priya.s@example.com"
    assert delivery.sent_at
    lines = outbox.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticket_id"] == "TKT-NEW"
    assert record["subject"] == "Re: MFA loop"
    assert record["body"] == "Please reopen your authenticator app."
    assert record["outbound_id"] == delivery.outbound_id


def test_second_send_appends(tmp_path: Path):
    outbox = tmp_path / "nested" / "sent.jsonl"
    run_send(_ticket(), _draft(), outbox=outbox)
    run_send(_ticket(), _draft(body="Second note."), outbox=outbox)
    assert len(outbox.read_text(encoding="utf-8").splitlines()) == 2


def test_needs_review_is_not_written(tmp_path: Path):
    outbox = tmp_path / "sent.jsonl"
    delivery = run_send(_ticket(), _draft(needs_review=True), outbox=outbox)
    assert delivery.status == "not_sent"
    assert delivery.outbound_id is None
    assert "review" in delivery.reason
    assert not outbox.exists()


def test_missing_recipient_is_not_written(tmp_path: Path):
    outbox = tmp_path / "sent.jsonl"
    delivery = run_send(_ticket(customer_id=None), _draft(), outbox=outbox)
    assert delivery.status == "not_sent"
    assert delivery.reason == "Ticket has no recipient."
    assert not outbox.exists()
