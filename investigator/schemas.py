"""What Investigator adds to the ticket state."""

from typing import Literal

from pydantic import BaseModel, Field

EvidenceSource = Literal["ticket", "log", "metric", "runbook"]


class Evidence(BaseModel):
    source: EvidenceSource
    ref: str
    excerpt: str


class Investigation(BaseModel):
    """Local lookup result. Responder is the suggested next node; the Supervisor decides."""

    status: Literal["investigated"]
    summary: str
    service: str | None = None
    runbook_path: str | None = None
    related_ticket_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    next_node_hint: Literal["responder"] = "responder"
