"""Investigator stub. Phase 3 replaces this with the tool-calling agent."""

from __future__ import annotations

import logging
import time

from triage_agent.state import Finding, Investigation, Ticket, TicketState

logger = logging.getLogger("triage_agent.investigator")


def investigator(state: TicketState) -> dict:
    started = time.perf_counter()
    ticket = state.get("ticket")
    ticket_id = ticket.id if isinstance(ticket, Ticket) else None
    logger.info(
        "node=investigator ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        "stub",
        int((time.perf_counter() - started) * 1000),
    )
    return {
        "investigation": Investigation(
            findings=[Finding(fact="stub finding", source="stub")],
            missing_info=[],
        )
    }
