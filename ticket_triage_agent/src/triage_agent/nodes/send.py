"""Send stub. Phase 7 writes the outbox and enforces idempotency."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from triage_agent.state import Ticket, TicketState

logger = logging.getLogger("triage_agent.send")


def send(state: TicketState) -> dict:
    started = time.perf_counter()
    ticket = state.get("ticket")
    ticket_id = ticket.id if isinstance(ticket, Ticket) else None
    logger.info(
        "node=send ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        "stub",
        int((time.perf_counter() - started) * 1000),
    )
    return {
        "final_reply": "stub reply",
        "sent": True,
        "sent_at": datetime.now(timezone.utc),
    }
