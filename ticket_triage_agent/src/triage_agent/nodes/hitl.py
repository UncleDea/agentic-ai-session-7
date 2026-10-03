"""HITL pause. The node restarts from the top on resume, so memory writes come after interrupt."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

from langgraph.store.sqlite import SqliteStore
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, model_validator

from triage_agent.memory.episodic import upsert_episode
from triage_agent.memory.procedural import get_current, list_versions, put_version
from triage_agent.state import (
    Review,
    ReviewStatus,
    Ticket,
    TicketState,
)
from triage_agent.tools.ticket_db import JsonCustomerStore

logger = logging.getLogger("triage_agent.hitl")

_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\w)"
)
_AMOUNT = re.compile(r"\$\s?\d+(?:\.\d{1,2})?|\b\d+\.\d{2}\b")
_ORDER_PHRASE = re.compile(
    r"\b(?:order|pedido|orden)\s+#?(\d{3,})\b",
    re.IGNORECASE,
)
_ORDER_SOURCE = re.compile(r"\border:(\d{3,})\b", re.IGNORECASE)
_CUSTOMER_FACT = re.compile(r"^(.+?)\s+email\s+", re.IGNORECASE)


class HitlResume(BaseModel):
    """What the reviewer sends back. A rule is saved only when save_as_rule is set."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "edit", "reject"]
    text: str | None = None
    feedback: str = ""
    save_as_rule: bool = False

    @model_validator(mode="after")
    def _complete(self) -> HitlResume:
        if self.action == "edit" and not (self.text or "").strip():
            raise ValueError("edit requires text")
        if self.action == "reject" and not self.feedback.strip():
            raise ValueError("reject requires feedback")
        if self.save_as_rule and not self.feedback.strip():
            raise ValueError("save_as_rule requires feedback")
        return self


def review_payload(state: TicketState) -> dict[str, Any]:
    """Plain dict shown to the reviewer. Safe to store in a checkpoint."""
    ticket = state.get("ticket")
    triage = state.get("triage")
    investigation = state.get("investigation")
    draft = state.get("draft")
    findings = []
    if investigation is not None:
        findings = [item.model_dump(mode="json") for item in investigation.findings]
    return {
        "ticket": ticket.model_dump(mode="json") if isinstance(ticket, Ticket) else None,
        "triage": triage.model_dump(mode="json") if triage is not None else None,
        "findings": findings,
        "draft": draft.reply if draft is not None else "",
        "flagged_claims": list(draft.flagged_claims) if draft is not None else [],
        "escalates": bool(draft.escalates) if draft is not None else False,
        "reasons": _reasons(state),
    }


def apply_decision(
    state: TicketState,
    decision: HitlResume | dict[str, Any],
    store: SqliteStore | None = None,
) -> dict:
    """Write the review, then the episode and optional rule. Safe to call twice."""
    choice = decision if isinstance(decision, HitlResume) else HitlResume.model_validate(decision)
    review = _review(choice)
    if store is not None:
        _learn(state, choice, store)
    return {"review": review}


def redact_customer_details(text: str, state: TicketState) -> str:
    """Replace emails, phones, the customer name, order ids, and amounts."""
    blobs = _blobs(state, text)
    redacted = _EMAIL.sub("[EMAIL]", text)
    redacted = _PHONE.sub("[PHONE]", redacted)
    redacted = _AMOUNT.sub("[AMOUNT]", redacted)
    for order_id in _order_ids(blobs):
        redacted = re.sub(rf"\b{re.escape(order_id)}\b", "[ORDER_ID]", redacted)
    for name in _names(state):
        redacted = re.sub(
            rf"\b{re.escape(name)}\b",
            "[CUSTOMER_NAME]",
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted


def hitl(state: TicketState, store: SqliteStore | None = None) -> dict:
    """Pause for a decision. Memory writes run only after interrupt returns."""
    started = time.perf_counter()
    ticket = state.get("ticket")
    ticket_id = ticket.id if isinstance(ticket, Ticket) else None
    if not isinstance(ticket, Ticket):
        _log(None, "error", started)
        return {"errors": ["hitl: state has no ticket"]}
    decision = interrupt(review_payload(state), response_schema=HitlResume)
    try:
        update = apply_decision(state, decision, store)
    except Exception as exc:
        _log(ticket_id, "error", started)
        return {"errors": [f"hitl: {type(exc).__name__}: {exc}"]}
    choice = decision if isinstance(decision, HitlResume) else HitlResume.model_validate(decision)
    _log(ticket_id, choice.action, started)
    return update


def _review(choice: HitlResume) -> Review:
    if choice.action == "approve":
        return Review(
            status=ReviewStatus.approved,
            reviewer_feedback=choice.feedback,
        )
    if choice.action == "edit":
        return Review(
            status=ReviewStatus.edited,
            reviewer_feedback=choice.feedback,
            edited_text=choice.text,
        )
    return Review(
        status=ReviewStatus.rejected,
        reviewer_feedback=choice.feedback,
    )


def _learn(state: TicketState, choice: HitlResume, store: SqliteStore) -> None:
    ticket = state.get("ticket")
    if not isinstance(ticket, Ticket):
        return
    reply = _approved_reply(state, choice)
    if reply is not None:
        _save_episode(state, ticket, reply, store)
    if choice.save_as_rule:
        _save_rule(ticket.id, choice, store)


def _approved_reply(state: TicketState, choice: HitlResume) -> str | None:
    if choice.action == "edit":
        text = (choice.text or "").strip()
        return text or None
    if choice.action != "approve":
        return None
    draft = state.get("draft")
    if draft is None or not draft.reply.strip():
        return None
    return draft.reply


def _save_episode(state: TicketState, ticket: Ticket, reply: str, store: SqliteStore) -> None:
    triage = state.get("triage")
    category = triage.category.value if triage is not None else ""
    row = {
        "id": f"EP-{ticket.id}",
        "ticket_id": ticket.id,
        "category": category,
        "subject": redact_customer_details(ticket.subject, state),
        "body": redact_customer_details(ticket.body, state),
        "approved_reply": redact_customer_details(reply, state),
    }
    upsert_episode(store, row)


def _save_rule(ticket_id: str, choice: HitlResume, store: SqliteStore) -> None:
    rule = choice.feedback.strip()
    note = f"hitl ticket_id={ticket_id} action={choice.action} rule={rule}"
    if _rule_already_saved(store, ticket_id, rule):
        return
    current = get_current(store)
    base = "" if current is None else str(current["text"]).rstrip()
    text = f"{base}\n\n{rule}" if base else rule
    saved = put_version(store, text, note)
    logger.info(
        "node=hitl ticket_id=%s decision=procedure v%s latency_ms=%d",
        ticket_id,
        saved["version"],
        0,
    )


def _rule_already_saved(store: SqliteStore, ticket_id: str, rule: str) -> bool:
    marker = f"ticket_id={ticket_id}"
    for row in list_versions(store):
        note = str(row.get("note", ""))
        if marker in note and rule in note:
            return True
    return False


def _reasons(state: TicketState) -> list[str]:
    history = list(state.get("route_history") or [])
    reason = ""
    for entry in history:
        if entry.node == "hitl":
            reason = entry.reason
    if not reason:
        return []
    return [part.strip() for part in reason.split(";") if part.strip()]


def _blobs(state: TicketState, reply: str) -> list[str]:
    blobs = [reply]
    ticket = state.get("ticket")
    if isinstance(ticket, Ticket):
        blobs.extend([ticket.subject, ticket.body])
    investigation = state.get("investigation")
    if investigation is not None:
        for finding in investigation.findings:
            blobs.extend([finding.fact, finding.source])
    return blobs


def _order_ids(blobs: list[str]) -> list[str]:
    found: list[str] = []
    for blob in blobs:
        for match in _ORDER_PHRASE.finditer(blob):
            found.append(match.group(1))
        for match in _ORDER_SOURCE.finditer(blob):
            found.append(match.group(1))
    unique = sorted(set(found), key=len, reverse=True)
    return unique


def _names(state: TicketState) -> list[str]:
    found: list[str] = []
    ticket = state.get("ticket")
    if isinstance(ticket, Ticket):
        customer = JsonCustomerStore().get_customer(ticket.customer_id)
        if customer.found:
            name = str(customer.records[0].get("name", "")).strip()
            if name:
                found.append(name)
    investigation = state.get("investigation")
    if investigation is not None:
        for finding in investigation.findings:
            if not finding.source.startswith("customer:"):
                continue
            match = _CUSTOMER_FACT.match(finding.fact.strip())
            if match:
                found.append(match.group(1).strip())
    tokens: list[str] = []
    for name in found:
        tokens.append(name)
        for part in name.split():
            if len(part) >= 3:
                tokens.append(part)
    unique = sorted(set(tokens), key=len, reverse=True)
    return unique


def _log(ticket_id: str | None, decision: str, started: float) -> None:
    logger.info(
        "node=hitl ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        decision,
        int((time.perf_counter() - started) * 1000),
    )
