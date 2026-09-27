"""Send a drafted reply to the local outbox. No network calls."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from responder.schemas import ResponseDraft
from triage_agent.schemas import TicketState

from send.schemas import Delivery

__all__ = ["run_send"]


def default_outbox() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "outbox" / "sent.jsonl"


def hold_reason(ticket: TicketState, draft: ResponseDraft) -> str | None:
    if draft.needs_review:
        return "Draft needs a human review before it can be sent."
    if not (ticket.customer_id or "").strip():
        return "Ticket has no recipient."
    return None


def run_send(
    ticket: TicketState | dict[str, Any],
    draft: ResponseDraft | dict[str, Any],
    *,
    outbox: Path | None = None,
) -> Delivery:
    """Append one outbound record, or refuse the draft without writing."""
    validated = TicketState.model_validate(
        ticket.model_dump() if isinstance(ticket, TicketState) else ticket
    )
    reply = ResponseDraft.model_validate(
        draft.model_dump() if isinstance(draft, ResponseDraft) else draft
    )
    reason = hold_reason(validated, reply)
    if reason:
        delivery = Delivery(
            status="not_sent",
            recipient=(validated.customer_id or "").strip() or None,
            subject=reply.subject,
            body=reply.body,
            reason=reason,
        )
        _log(validated.ticket_id, delivery)
        return delivery

    sent_at = datetime.now(timezone.utc).isoformat()
    outbound_id = f"OUT-{uuid.uuid4().hex[:8].upper()}"
    recipient = validated.customer_id.strip()
    record = {
        "outbound_id": outbound_id,
        "ticket_id": validated.ticket_id,
        "recipient": recipient,
        "subject": reply.subject,
        "body": reply.body,
        "sent_at": sent_at,
    }
    path = outbox or default_outbox()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    delivery = Delivery(
        status="sent",
        outbound_id=outbound_id,
        recipient=recipient,
        subject=reply.subject,
        body=reply.body,
        sent_at=sent_at,
    )
    _log(validated.ticket_id, delivery)
    return delivery


def _log(ticket_id: str, delivery: Delivery) -> None:
    payload = {
        "ticket_id": ticket_id,
        "status": delivery.status,
        "outbound_id": delivery.outbound_id,
        "recipient": delivery.recipient,
    }
    print(json.dumps(payload), flush=True)
