"""Build the payload a reviewer sees. The graph is what pauses."""

from triage_agent.schemas import TicketState

from hitl.schemas import HumanDecision, ReviewRequest


def review_reason(ticket_state: TicketState) -> str:
    if ticket_state.severity in {"high", "critical"}:
        why = f"Severity is {ticket_state.severity}."
    else:
        why = "Triage asked for a human review."
    detail = ticket_state.triage_reasoning.strip()
    if detail:
        return f"{why} {detail}"
    return why


def review_payload(ticket_state: TicketState) -> dict[str, str]:
    return {
        "ticket_id": ticket_state.ticket_id,
        "severity": ticket_state.severity,
        "category": ticket_state.category,
        "reason": review_reason(ticket_state),
        "raw_text": ticket_state.raw_text,
    }


def review_from_decision(ticket_state: TicketState, decision: HumanDecision) -> ReviewRequest:
    status = "approved" if decision.action == "approve" else "rejected"
    return ReviewRequest(
        status=status,
        ticket_id=ticket_state.ticket_id,
        severity=ticket_state.severity,
        category=ticket_state.category,
        reason=review_reason(ticket_state),
        note=decision.note,
    )
