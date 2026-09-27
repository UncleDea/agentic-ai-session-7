"""What the Responder adds to the ticket state."""

from typing import Literal

from pydantic import BaseModel, Field

DomainName = Literal["support", "it-helpdesk", "oncall"]
MemoryKind = Literal["semantic", "episodic", "procedural"]


class ReplyText(BaseModel):
    """Private structured result from the draft model."""

    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class ResponseDraft(BaseModel):
    """A reply that has not been sent. Send is the suggested next node."""

    status: Literal["drafted"]
    subject: str
    body: str
    domain: DomainName
    memory_used: list[MemoryKind]
    needs_review: bool
    next_node_hint: Literal["send"] = "send"
