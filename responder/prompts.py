"""Draft prompt. The domain policy and memory are filled in by the agent."""

from langchain_core.messages import HumanMessage, SystemMessage

from investigator.schemas import Investigation
from triage_agent.schemas import TicketState


def draft_messages(
    ticket: TicketState,
    investigation: Investigation,
    memory: dict[str, str],
) -> list[SystemMessage | HumanMessage]:
    policy = memory.get("semantic", "Be concise and professional. Do not invent facts.")
    system = (
        f"{policy}\n\n"
        "Draft a customer-facing reply from the ticket and the memory below. "
        "Use only those facts. Do not promise a refund or a credit. "
        "If the memory does not contain a fix, say a specialist will review the ticket."
    )
    evidence = "\n".join(
        f"- [{item.source}] {item.ref}: {item.excerpt}" for item in investigation.evidence
    ) or "(none)"
    human = (
        f"Ticket {ticket.ticket_id} ({ticket.channel}):\n{ticket.raw_text}\n\n"
        f"Investigation: {investigation.summary}\n"
        f"Service: {investigation.service or '(unknown)'}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Episodic memory:\n{memory.get('episodic', '(none)')}\n\n"
        f"Procedural memory:\n{memory.get('procedural', '(none)')}"
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]
