"""Result the Supervisor returns after routing a triaged ticket."""

from typing import Literal

from pydantic import BaseModel

from hitl.schemas import ReviewRequest
from investigator.schemas import Investigation
from responder.schemas import ResponseDraft
from send.schemas import Delivery
from triage_agent.schemas import TicketState

RoutedTo = Literal["hitl", "investigator", "responder", "send"]


class SupervisorResult(BaseModel):
    """Ticket state, the last node reached, and any review, findings, or draft."""

    ticket_state: TicketState
    routed_to: RoutedTo
    investigation: Investigation | None = None
    draft: ResponseDraft | None = None
    delivery: Delivery | None = None
    review: ReviewRequest | None = None
    thread_id: str | None = None
