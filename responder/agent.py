"""Responder. Drafts a reply. It does not send it."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from investigator.schemas import Investigation
from triage_agent.schemas import TicketState

from responder.config import MODEL_NAME, TEMPERATURE
from responder.memory import default_data_dir, domain_for, gather_memory
from responder.prompts import draft_messages
from responder.schemas import ReplyText, ResponseDraft

__all__ = ["run_response"]

_PROMISE = re.compile(r"\b(refund|credits?)\b", re.IGNORECASE)


def run_response(
    ticket: TicketState | dict[str, Any],
    investigation: Investigation | dict[str, Any],
    *,
    model: Any | None = None,
    data_dir: Path | None = None,
) -> ResponseDraft:
    """Draft a reply from the triaged ticket, the investigation, and local memory."""
    validated = TicketState.model_validate(
        ticket.model_dump() if isinstance(ticket, TicketState) else ticket
    )
    findings = Investigation.model_validate(
        investigation.model_dump() if isinstance(investigation, Investigation) else investigation
    )
    root = data_dir or default_data_dir()
    domain = domain_for(findings)
    memory = gather_memory(findings, validated.raw_text, validated.customer_id, root)
    needs_review = not findings.evidence

    try:
        reply = _draft(validated, findings, memory, model)
        subject = reply.subject.strip()
        body = reply.body.strip()
        if not subject or not body:
            raise ValueError("empty draft")
    except Exception:
        subject, body = _fallback(validated.ticket_id)
        needs_review = True

    if domain == "support" and _PROMISE.search(body):
        subject, body = _fallback(validated.ticket_id)
        needs_review = True

    draft = ResponseDraft(
        status="drafted",
        subject=subject,
        body=body,
        domain=domain,
        memory_used=[kind for kind in ("semantic", "episodic", "procedural") if kind in memory],
        needs_review=needs_review,
    )
    _log(validated.ticket_id, draft)
    return draft


def _draft(
    ticket: TicketState,
    investigation: Investigation,
    memory: dict[str, str],
    model: Any | None,
) -> ReplyText:
    chat = model if model is not None else _default_model()
    structured = chat.with_structured_output(ReplyText)
    result = structured.invoke(draft_messages(ticket, investigation, memory))
    if isinstance(result, ReplyText):
        return result
    return ReplyText.model_validate(result)


def _default_model() -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


def _fallback(ticket_id: str) -> tuple[str, str]:
    return (
        f"Re: {ticket_id}",
        "Thank you for contacting us. A specialist needs to review this ticket "
        "before we send a resolution. We have not changed your account.",
    )


def _log(ticket_id: str, draft: ResponseDraft) -> None:
    payload = {
        "ticket_id": ticket_id,
        "domain": draft.domain,
        "needs_review": draft.needs_review,
        "memory_used": draft.memory_used,
    }
    print(json.dumps(payload), flush=True)
