"""Shared ticket state. Nodes read and write only this object."""

from __future__ import annotations

import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    shipping = "shipping"
    refund = "refund"
    feedback = "feedback"
    spam = "spam"
    security = "security"
    legal = "legal"
    other = "other"


class Priority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class ReviewStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    edited = "edited"
    rejected = "rejected"


class Channel(str, Enum):
    email = "email"
    chat = "chat"
    phone = "phone"
    web = "web"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ticket(_Model):
    id: str
    customer_id: str
    channel: Channel
    subject: str
    body: str
    created_at: datetime


class TriageResult(_Model):
    """Classification of one support ticket."""

    category: Category = Field(
        description="Primary category: the action the customer wants."
    )
    priority: Priority = Field(
        description=(
            "P1 critical, P2 high, P3 normal, P4 low. "
            "Start from the category default and raise by at most one level "
            "for a clear severity signal. Never set it below the default."
        )
    )
    sentiment: Sentiment = Field(
        description="Customer tone: positive, neutral, or negative."
    )
    language: str = Field(
        description="ISO 639-1 code of the ticket body, such as en or es."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "Confidence from 0 to 1. Use a value below 0.7 when the body "
            "is empty or only a few words."
        ),
    )
    needs_human: bool = Field(
        description=(
            "True for security, legal or chargeback threats, or an empty "
            "or unintelligible body."
        )
    )
    needs_human_reason: str = Field(
        default="",
        description="Short reason when needs_human is true; empty string otherwise.",
    )
    secondary_categories: list[Category] = Field(
        default_factory=list,
        description="Other real issues, excluding the primary category.",
    )

    @model_validator(mode="after")
    def _reason_when_human(self) -> TriageResult:
        if self.needs_human and not self.needs_human_reason.strip():
            raise ValueError("needs_human requires a reason")
        return self


class Finding(_Model):
    fact: str
    source: str


class Investigation(_Model):
    findings: list[Finding] = Field(default_factory=list)
    related_ticket_ids: list[str] = Field(default_factory=list)
    kb_article_ids: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


class Draft(_Model):
    reply: str
    memory_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class Review(_Model):
    status: ReviewStatus
    reviewer_feedback: str = ""
    edited_text: str | None = None


class RouteEntry(_Model):
    node: str
    reason: str


class TicketState(TypedDict, total=False):
    ticket: Ticket
    triage: TriageResult | None
    investigation: Investigation | None
    draft: Draft | None
    review: Review | None
    final_reply: str | None
    sent: bool
    sent_at: datetime | None
    next_node: str | None
    route_history: Annotated[list[RouteEntry], operator.add]
    iteration_count: int
    revision_count: int
    errors: Annotated[list[str], operator.add]


def initial_state(ticket: Ticket) -> TicketState:
    """A new run: nothing classified, drafted, reviewed, or sent."""
    return {
        "ticket": ticket,
        "triage": None,
        "investigation": None,
        "draft": None,
        "review": None,
        "final_reply": None,
        "sent": False,
        "sent_at": None,
        "next_node": None,
        "route_history": [],
        "iteration_count": 0,
        "revision_count": 0,
        "errors": [],
    }
