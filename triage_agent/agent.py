"""Triage Agent. The only public entry point is run_triage."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from triage_agent.config import CONFIDENCE_THRESHOLD, MODEL_NAME, TEMPERATURE
from triage_agent.prompts import classification_messages
from triage_agent.schemas import TicketInput, TicketState, TriageClassification

__all__ = ["run_triage"]


def run_triage(
    ticket: TicketInput | dict[str, Any],
    *,
    model: Any | None = None,
) -> TicketState:
    """Classify one ticket and return the updated ticket state.

    The Supervisor calls this function. It does not share globals or hidden state.
    Pass ``model`` in tests. When omitted, a ChatOpenAI client is built from config.
    """
    validated = TicketInput.model_validate(
        ticket.model_dump() if isinstance(ticket, TicketInput) else ticket
    )
    if not validated.raw_text.strip():
        return _assemble(
            validated,
            category="other",
            severity="medium",
            confidence=0.0,
            reasoning=(
                "raw_text was empty or whitespace. A human should review this ticket."
            ),
            fallback=True,
        )

    try:
        classification = _classify(validated.raw_text, model)
    except Exception as exc:
        return _assemble(
            validated,
            category="other",
            severity="medium",
            confidence=0.0,
            reasoning=(
                "Classification failed schema validation or the model call failed. "
                "A human should review this ticket. "
                f"Error: {_public_error(exc)}"
            ),
            fallback=True,
        )

    if classification.confidence < CONFIDENCE_THRESHOLD:
        return _assemble(
            validated,
            category="other",
            severity="medium",
            confidence=classification.confidence,
            reasoning=(
                "Low-confidence fallback: "
                f"model confidence {classification.confidence:.2f} is below "
                f"threshold {CONFIDENCE_THRESHOLD:.2f}. "
                "A human should review this ticket. "
                f"Model reasoning: {classification.reasoning}"
            ),
            fallback=True,
        )

    return _assemble(
        validated,
        category=classification.category,
        severity=classification.severity,
        confidence=classification.confidence,
        reasoning=classification.reasoning,
        fallback=False,
    )


def _classify(raw_text: str, model: Any | None) -> TriageClassification:
    chat = model if model is not None else _default_model()
    structured = chat.with_structured_output(TriageClassification)
    result = structured.invoke(classification_messages(raw_text))
    if isinstance(result, TriageClassification):
        return result
    return TriageClassification.model_validate(result)


def _default_model() -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


def _next_node_hint(severity: str, fallback: bool) -> str:
    if fallback or severity in {"high", "critical"}:
        return "hitl"
    return "investigator"


def _assemble(
    ticket: TicketInput,
    *,
    category: str,
    severity: str,
    confidence: float,
    reasoning: str,
    fallback: bool,
) -> TicketState:
    state = TicketState(
        ticket_id=ticket.ticket_id,
        raw_text=ticket.raw_text,
        customer_id=ticket.customer_id,
        channel=ticket.channel,
        received_at=ticket.received_at,
        category=category,
        severity=severity,
        confidence=confidence,
        triage_reasoning=reasoning,
        status="triaged",
        next_node_hint=_next_node_hint(severity, fallback),
    )
    _log_decision(state)
    return state


def _public_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    secret = os.environ.get("OPENAI_API_KEY", "")
    if secret:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"sk-[A-Za-z0-9_\-]+", "[redacted]", text)
    return " ".join(text.split())[:300]


def _log_decision(state: TicketState) -> None:
    payload = {
        "ticket_id": state.ticket_id,
        "category": state.category,
        "severity": state.severity,
        "confidence": state.confidence,
    }
    print(json.dumps(payload), flush=True)
