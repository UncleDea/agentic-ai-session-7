"""Send the final reply. A follow-up case is recorded before the outbox write."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from triage_agent.config import Settings, get_settings, resolve_dir
from triage_agent.nodes.responder import promises_escalation
from triage_agent.state import (
    Review,
    ReviewStatus,
    RouteEntry,
    Ticket,
    TicketState,
)
from triage_agent.tools.sender import (
    EscalationLog,
    EscalationRecord,
    JsonEscalationLog,
    OutboundMessage,
    OutboxSender,
    Sender,
)

logger = logging.getLogger("triage_agent.send")

_REFUSAL = "unapproved flagged or escalating draft"


def final_reply_text(state: TicketState) -> str:
    """Edited review text when the reviewer changed it, otherwise the draft."""
    review = state.get("review")
    if isinstance(review, Review) and review.status is ReviewStatus.edited:
        edited = review.edited_text or ""
        if edited.strip():
            return edited
    draft = state.get("draft")
    if draft is None:
        return ""
    return draft.reply


def send(
    state: TicketState,
    config: RunnableConfig | None = None,
    settings: Settings | None = None,
    sender: Sender | None = None,
    escalations: EscalationLog | None = None,
) -> dict:
    """Deliver the reply once per run. Refuse an unapproved risky draft."""
    started = time.perf_counter()
    ticket = state.get("ticket")
    ticket_id = ticket.id if isinstance(ticket, Ticket) else None
    try:
        return _deliver(
            state,
            config,
            settings,
            sender,
            escalations,
            started,
        )
    except Exception as exc:
        _log("send", ticket_id, "error", started)
        return {"errors": [f"send: {type(exc).__name__}: {exc}"]}


def close_run(
    state: TicketState,
    config: RunnableConfig | None = None,
    settings: Settings | None = None,
    escalations: EscalationLog | None = None,
) -> dict:
    """Record a follow-up when the run ends escalated and nothing was sent."""
    started = time.perf_counter()
    if state.get("sent") or not state.get("escalated"):
        return {}
    ticket = state.get("ticket")
    ticket_id = ticket.id if isinstance(ticket, Ticket) else None
    try:
        if not isinstance(ticket, Ticket):
            _log("close", None, "error", started)
            return {"errors": ["close: state has no ticket"]}
        thread_id = _thread_id(config)
        if not thread_id:
            _log("close", ticket.id, "error", started)
            return {"errors": ["close: missing thread_id"]}
        cases = escalations if escalations is not None else _cases(settings)
        draft = state.get("draft")
        reply = "" if draft is None else draft.reply
        reasons = _unsent_reasons(state)
        if draft is not None and draft.escalates and "escalation promised" not in reasons:
            reasons.append("escalation promised")
        created = _open_case(
            cases,
            ticket.id,
            thread_id,
            reasons,
            reply,
            datetime.now(timezone.utc),
        )
        _log("close", ticket.id, "escalated" if created else "already_sent", started)
        return {}
    except Exception as exc:
        _log("close", ticket_id, "error", started)
        return {"errors": [f"close: {type(exc).__name__}: {exc}"]}


def _deliver(
    state: TicketState,
    config: RunnableConfig | None,
    settings: Settings | None,
    sender: Sender | None,
    escalations: EscalationLog | None,
    started: float,
) -> dict:
    ticket = state.get("ticket")
    if not isinstance(ticket, Ticket):
        _log("send", None, "error", started)
        return {"errors": ["send: state has no ticket"]}
    thread_id = _thread_id(config)
    if not thread_id:
        _log("send", ticket.id, "error", started)
        return {"errors": ["send: missing thread_id"]}

    outbound = sender if sender is not None else _outbox(settings)
    cases = escalations if escalations is not None else _cases(settings)
    existing = outbound.get(thread_id)
    if existing is not None:
        _backfill_escalation(cases, existing)
        _log("send", ticket.id, "already_sent", started)
        return {
            "final_reply": existing.final_reply,
            "sent": True,
            "sent_at": existing.sent_at,
        }
    if state.get("sent"):
        _log("send", ticket.id, "already_sent", started)
        return {}

    reply = final_reply_text(state)
    if not reply.strip():
        _log("send", ticket.id, "error", started)
        return {"errors": ["send: no reply"]}
    if _unapproved_risk(state):
        _log("send", ticket.id, "refused", started)
        return {
            "next_node": "hitl",
            "route_history": [RouteEntry(node="hitl", reason=_REFUSAL)],
        }

    sent_at = datetime.now(timezone.utc)
    if promises_escalation(reply):
        _open_case(
            cases,
            ticket.id,
            thread_id,
            ["escalation promised"],
            reply,
            sent_at,
        )
    message = OutboundMessage(
        ticket_id=ticket.id,
        thread_id=thread_id,
        final_reply=reply,
        sent_at=sent_at,
    )
    stored = outbound.send(message)
    _log("send", ticket.id, "sent", started)
    return {
        "final_reply": stored.final_reply,
        "sent": True,
        "sent_at": stored.sent_at,
    }


def _backfill_escalation(cases: EscalationLog, existing: OutboundMessage) -> None:
    """The outbox is already there. Open the follow-up only when it is missing."""
    if not promises_escalation(existing.final_reply):
        return
    if cases.get(existing.thread_id) is not None:
        return
    _open_case(
        cases,
        existing.ticket_id,
        existing.thread_id,
        ["escalation promised"],
        existing.final_reply,
        existing.sent_at,
    )


def _open_case(
    cases: EscalationLog,
    ticket_id: str,
    thread_id: str,
    reasons: list[str],
    reply: str,
    created_at: datetime,
) -> bool:
    """True when this call created the case. An existing case is left as it is."""
    if cases.get(thread_id) is not None:
        return False
    cases.open_case(
        EscalationRecord(
            ticket_id=ticket_id,
            thread_id=thread_id,
            reasons=reasons,
            reply=reply,
            created_at=created_at,
            status="open",
        )
    )
    return cases.get(thread_id) is not None


def _unapproved_risk(state: TicketState) -> bool:
    draft = state.get("draft")
    if draft is None:
        return False
    if not draft.flagged_claims and not draft.escalates:
        return False
    review = state.get("review")
    if not isinstance(review, Review):
        return True
    return review.status not in (ReviewStatus.approved, ReviewStatus.edited)


def _unsent_reasons(state: TicketState) -> list[str]:
    reason = ""
    for entry in state.get("route_history") or []:
        if entry.node == "hitl":
            reason = entry.reason
    parts = [part.strip() for part in reason.split(";") if part.strip()]
    if parts:
        return parts
    return ["escalated; manual handling"]


def _thread_id(config: RunnableConfig | None) -> str:
    if not config:
        return ""
    configurable = config.get("configurable") or {}
    return str(configurable.get("thread_id") or "")


def _outbox(settings: Settings | None) -> OutboxSender:
    active = settings if settings is not None else get_settings()
    return OutboxSender(resolve_dir(active.outbox_dir))


def _cases(settings: Settings | None) -> JsonEscalationLog:
    active = settings if settings is not None else get_settings()
    return JsonEscalationLog(resolve_dir(active.escalations_dir))


def _log(node: str, ticket_id: str | None, decision: str, started: float) -> None:
    logger.info(
        "node=%s ticket_id=%s decision=%s latency_ms=%d",
        node,
        ticket_id,
        decision,
        int((time.perf_counter() - started) * 1000),
    )
