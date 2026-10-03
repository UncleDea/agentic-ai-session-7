"""HITL stub. Phase 6 pauses with interrupt and resumes on a human decision."""

from __future__ import annotations

import logging
import time

from triage_agent.state import Review, ReviewStatus, Ticket, TicketState

logger = logging.getLogger("triage_agent.hitl")


def hitl(state: TicketState) -> dict:
    started = time.perf_counter()
    ticket = state.get("ticket")
    ticket_id = ticket.id if isinstance(ticket, Ticket) else None
    logger.info(
        "node=hitl ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        "stub approval",
        int((time.perf_counter() - started) * 1000),
    )
    return {
        "review": Review(
            status=ReviewStatus.approved,
            reviewer_feedback="stub approval",
        )
    }
