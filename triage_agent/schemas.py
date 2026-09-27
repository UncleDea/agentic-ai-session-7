"""Typed contract between the Supervisor and the Triage Agent."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal[
    "billing",
    "technical",
    "account",
    "feature_request",
    "spam",
    "other",
]
Severity = Literal["low", "medium", "high", "critical"]
NextNodeHint = Literal["investigator", "hitl"]


class TicketInput(BaseModel):
    """Payload the Supervisor passes into Triage."""

    ticket_id: str = Field(min_length=1)
    raw_text: str
    customer_id: str | None = None
    channel: str = Field(min_length=1)
    received_at: datetime

    @field_validator("ticket_id", "channel")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TriageClassification(BaseModel):
    """Private structured result from the classifier. Not part of the Supervisor contract."""

    category: Category
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class TicketState(BaseModel):
    """Ticket state after Triage has run."""

    ticket_id: str
    raw_text: str
    customer_id: str | None
    channel: str
    received_at: datetime
    category: Category
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    triage_reasoning: str
    status: Literal["triaged"]
    next_node_hint: NextNodeHint | None
