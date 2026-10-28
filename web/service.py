"""Classify one ticket. This app does not investigate, draft, or send mail."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import ValidationError

from supervisor.cli import _check_token, _require_model
from triage_agent import TicketInput, run_triage
from triage_agent.schemas import TicketState


class DeskError(Exception):
    """A triage request the page can show to the operator."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class Desk:
    """The ticket triage agent, with a stand-in classifier available for tests."""

    def __init__(self, triage_fn=None) -> None:
        self.triage_fn = triage_fn

    def classify(
        self,
        text: str,
        *,
        customer_id: str | None = None,
        channel: str = "email",
        ticket_id: str | None = None,
    ) -> dict:
        if not text or not text.strip():
            raise DeskError("Enter the customer's message.")
        try:
            if self.triage_fn is None:
                _require_model(True)
            chosen_id = ticket_id.strip() if ticket_id else f"TKT-{uuid.uuid4().hex[:8].upper()}"
            _check_token(chosen_id, "ticket id")
            ticket = TicketInput(
                ticket_id=chosen_id,
                raw_text=text,
                customer_id=(customer_id or "").strip() or None,
                channel=(channel or "email").strip() or "email",
                received_at=datetime.now(timezone.utc),
            )
            state = self.triage_fn(ticket) if self.triage_fn is not None else run_triage(ticket)
            if not isinstance(state, TicketState):
                state = TicketState.model_validate(state)
        except DeskError:
            raise
        except (SystemExit, ValidationError) as exc:
            raise DeskError(_error_text(exc)) from exc
        payload = state.model_dump(mode="json")
        payload["classification_failed"] = _classification_failed(state.confidence, state.triage_reasoning)
        return payload


def _classification_failed(confidence: float, reasoning: str) -> bool:
    text = reasoning.casefold()
    if confidence != 0.0:
        return False
    return (
        "model call failed" in text
        or "401" in text
        or "empty or whitespace" in text
        or "schema validation" in text
    )


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, SystemExit) and isinstance(exc.code, str) and exc.code.strip():
        return exc.code.strip()
    if isinstance(exc, ValidationError):
        return "The ticket could not be read. Check the ticket id and try again."
    return "The ticket could not be classified."
