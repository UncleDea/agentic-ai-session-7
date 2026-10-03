"""Supervisor. Deterministic routing over TicketState."""

from __future__ import annotations

import logging
import time

from triage_agent.config import Settings, get_settings
from triage_agent.state import (
    Category,
    ReviewStatus,
    RouteEntry,
    Ticket,
    TicketState,
)

logger = logging.getLogger("triage_agent.supervisor")


def needs_hitl(state: TicketState, settings: Settings) -> tuple[bool, str]:
    """Return whether the draft must be reviewed, and the matching criteria."""
    triage = state.get("triage")
    if triage is None:
        return False, ""
    reasons: list[str] = []
    if triage.priority in settings.hitl_priorities:
        reasons.append(f"priority {triage.priority.value}")
    if triage.confidence < settings.confidence_threshold:
        reasons.append("triage confidence below threshold")
    draft = state.get("draft")
    if draft is not None and draft.confidence < settings.confidence_threshold:
        reasons.append("draft confidence below threshold")
    if draft is not None and draft.escalates:
        reasons.append("escalation promised")
    if triage.sentiment in settings.hitl_sentiments:
        reasons.append(f"sentiment {triage.sentiment.value}")
    if triage.category in settings.hitl_categories:
        reasons.append(f"category {triage.category.value}")
    if triage.needs_human:
        reasons.append("needs_human")
    investigation = state.get("investigation")
    if investigation is not None and investigation.missing_info:
        reasons.append("missing_info")
    if not reasons:
        return False, ""
    return True, "; ".join(reasons)


def _status(state: TicketState) -> ReviewStatus | None:
    review = state.get("review")
    if review is None:
        return None
    return review.status


def _repeated(errors: list[str]) -> bool:
    seen: set[str] = set()
    for error in errors:
        if error in seen:
            return True
        seen.add(error)
    return False


def _decide(state: TicketState, settings: Settings) -> tuple[str, str, bool, bool]:
    """Return next node, reason, whether to mark escalated, and force sent false.

    Approved or edited reviews are checked before the iteration cap and
    repeated-error rules so a reviewer can still release a capped run.
    Cap, repeated errors, and a rejected review at the revision cap mark
    the run escalated. The next pass then sends an approval or stops.
    A revised draft that nobody has reviewed yet goes to HITL before the
    general criteria and the default send. An approval still sends.
    """
    if state.get("sent"):
        return "end", "already sent", False, False

    status = _status(state)
    if status is ReviewStatus.approved:
        return "send", "review approved", False, False
    if status is ReviewStatus.edited:
        return "send", "review edited", False, False

    if state.get("escalated"):
        return "end", "escalated; manual handling", True, True

    if state.get("iteration_count", 0) >= settings.max_iterations:
        return "hitl", "iteration cap", True, False

    if _repeated(list(state.get("errors") or [])):
        return "hitl", "repeated error", True, False

    triage = state.get("triage")
    if triage is None:
        return "triage", "no triage yet", False, False
    if triage.category is Category.spam:
        return "end", "spam; no reply", False, False

    investigation = state.get("investigation")
    needs_facts = (
        investigation is None and triage.category in settings.investigation_categories
    )
    if needs_facts:
        return "investigator", f"{triage.category.value} needs facts", False, False

    if state.get("draft") is None:
        if investigation is None:
            return "responder", "no facts required", False, False
        return "responder", "investigation done, no draft", False, False

    if status is ReviewStatus.rejected:
        if state.get("revision_count", 0) >= settings.max_revisions:
            return "hitl", "revision cap", True, False
        return "responder", "review rejected; revision", False, False

    if state.get("revision_count", 0) > 0 and status in (None, ReviewStatus.pending):
        return "hitl", "revised draft needs review", False, False

    required, reason = needs_hitl(state, settings)
    if required:
        return "hitl", reason, False, False
    return "send", "hitl criteria not met", False, False


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log(ticket_id: str | None, decision: str, started: float) -> None:
    logger.info(
        "node=supervisor ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        decision,
        _latency_ms(started),
    )


def supervisor(state: TicketState, settings: Settings | None = None) -> dict:
    """Choose the next node. Returns next_node, one route entry, and the new count."""
    started = time.perf_counter()
    active = settings if settings is not None else get_settings()
    ticket = state.get("ticket")
    ticket_id = ticket.id if isinstance(ticket, Ticket) else None
    decision, reason, escalate, unsent = _decide(state, active)
    update: dict = {
        "next_node": decision,
        "route_history": [RouteEntry(node=decision, reason=reason)],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
    if escalate:
        update["escalated"] = True
    if unsent:
        update["sent"] = False
    _log(ticket_id, decision, started)
    return update
