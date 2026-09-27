"""The human review contract."""

from typing import Literal

from pydantic import BaseModel, Field


class HumanDecision(BaseModel):
    """What a reviewer sends back to resume the graph."""

    action: Literal["approve", "reject"]
    note: str = ""


class ReviewRequest(BaseModel):
    """Why a ticket is waiting, and what the reviewer decided."""

    status: Literal["pending", "approved", "rejected"]
    ticket_id: str
    severity: str
    category: str
    reason: str
    note: str = Field(default="")
