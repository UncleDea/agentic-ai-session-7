"""Responder stub. Phase 5 drafts from memory and applies revisions."""

from __future__ import annotations

import logging
import time

from triage_agent.state import Draft, Ticket, TicketState

logger = logging.getLogger("triage_agent.responder")


def responder(state: TicketState) -> dict:
    # Phase 5: when review is rejected, return review=None and revision_count + 1.
    started = time.perf_counter()
    ticket = state.get("ticket")
    ticket_id = ticket.id if isinstance(ticket, Ticket) else None
    logger.info(
        "node=responder ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        "stub",
        int((time.perf_counter() - started) * 1000),
    )
    return {"draft": Draft(reply="stub reply", memory_ids=[], confidence=1.0)}
