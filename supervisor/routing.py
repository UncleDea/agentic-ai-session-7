"""The Supervisor's routing policy. Other nodes only suggest; this module decides."""

from investigator.schemas import Investigation
from responder.schemas import ResponseDraft
from send.agent import hold_reason
from triage_agent.schemas import TicketState


def choose_next_node(ticket_state: TicketState) -> str:
    """Pick the next node from the triaged state.

    High and critical tickets go to a human. A hitl hint does too, which covers
    the low-confidence fallback. Every other triaged ticket goes to Investigator.
    """
    if ticket_state.severity in {"high", "critical"}:
        return "hitl"
    if ticket_state.next_node_hint == "hitl":
        return "hitl"
    return "investigator"


def choose_after_investigation(investigation: Investigation) -> str:
    """After a completed investigation, the next built node is the Responder."""
    if investigation.next_node_hint == "responder":
        return "responder"
    return "responder"


def choose_after_draft(ticket_state: TicketState, draft: ResponseDraft) -> str:
    """Send a finished draft. Hold it when a person still needs to review it."""
    if hold_reason(ticket_state, draft):
        return "hold"
    return "send"
