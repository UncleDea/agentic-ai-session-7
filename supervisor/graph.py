"""LangGraph supervisor. Send writes the outbox. A review hold is not delivered."""

from __future__ import annotations

import json
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hitl.review import review_from_decision, review_payload
from hitl.schemas import HumanDecision, ReviewRequest
from investigator import Investigation, run_investigation
from responder import ResponseDraft, run_response
from send import Delivery, run_send
from send.agent import hold_reason
from triage_agent import TicketInput, TicketState, run_triage
from supervisor.routing import choose_after_draft, choose_after_investigation, choose_next_node
from supervisor.schemas import SupervisorResult

CHECKPOINTER = InMemorySaver()


class SupervisorState(TypedDict, total=False):
    ticket: dict[str, Any]
    ticket_state: dict[str, Any]
    routed_to: str
    investigation: dict[str, Any]
    draft: dict[str, Any]
    delivery: dict[str, Any]
    review: dict[str, Any]


def build_graph(
    triage_fn=run_triage,
    investigate_fn=run_investigation,
    respond_fn=run_response,
    send_fn=run_send,
    checkpointer=None,
):
    """Compile the graph. Pass replacements for later nodes in tests."""

    def triage_node(state: SupervisorState) -> dict[str, Any]:
        ticket = TicketInput.model_validate(state["ticket"])
        result = triage_fn(ticket)
        if not isinstance(result, TicketState):
            result = TicketState.model_validate(result)
        return {"ticket_state": result.model_dump(mode="json")}

    def supervisor_node(state: SupervisorState) -> dict[str, str]:
        ticket_state = TicketState.model_validate(state["ticket_state"])
        return {"routed_to": choose_next_node(ticket_state)}

    def hitl_node(state: SupervisorState) -> dict[str, Any]:
        ticket_state = TicketState.model_validate(state["ticket_state"])
        decision = interrupt(review_payload(ticket_state), response_schema=HumanDecision)
        if not isinstance(decision, HumanDecision):
            decision = HumanDecision.model_validate(decision)
        review = review_from_decision(ticket_state, decision)
        return {"review": review.model_dump(mode="json")}

    def route_after_hitl(state: SupervisorState) -> str:
        review = state.get("review") or {}
        if review.get("status") == "approved":
            return "investigator"
        return "end"

    def investigator_node(state: SupervisorState) -> dict[str, Any]:
        ticket_state = TicketState.model_validate(state["ticket_state"])
        result = investigate_fn(ticket_state)
        if not isinstance(result, Investigation):
            result = Investigation.model_validate(result)
        return {"investigation": result.model_dump(mode="json")}

    def supervisor_after_investigation(state: SupervisorState) -> dict[str, str]:
        investigation = Investigation.model_validate(state["investigation"])
        return {"routed_to": choose_after_investigation(investigation)}

    def responder_node(state: SupervisorState) -> dict[str, Any]:
        ticket_state = TicketState.model_validate(state["ticket_state"])
        investigation = Investigation.model_validate(state["investigation"])
        result = respond_fn(ticket_state, investigation)
        if not isinstance(result, ResponseDraft):
            result = ResponseDraft.model_validate(result)
        return {"draft": result.model_dump(mode="json")}

    def supervisor_after_draft(state: SupervisorState) -> dict[str, str]:
        ticket_state = TicketState.model_validate(state["ticket_state"])
        draft = ResponseDraft.model_validate(state["draft"])
        return {"routed_to": choose_after_draft(ticket_state, draft)}

    def send_node(state: SupervisorState) -> dict[str, Any]:
        ticket_state = TicketState.model_validate(state["ticket_state"])
        draft = ResponseDraft.model_validate(state["draft"])
        result = send_fn(ticket_state, draft)
        if not isinstance(result, Delivery):
            result = Delivery.model_validate(result)
        return {"delivery": result.model_dump(mode="json"), "routed_to": "send"}

    def hold_node(state: SupervisorState) -> dict[str, Any]:
        ticket_state = TicketState.model_validate(state["ticket_state"])
        draft = ResponseDraft.model_validate(state["draft"])
        delivery = Delivery(
            status="not_sent",
            recipient=(ticket_state.customer_id or "").strip() or None,
            subject=draft.subject,
            body=draft.body,
            reason=hold_reason(ticket_state, draft),
        )
        return {"delivery": delivery.model_dump(mode="json"), "routed_to": "responder"}

    def route(state: SupervisorState) -> str:
        return state["routed_to"]

    graph = StateGraph(SupervisorState)
    graph.add_node("triage", triage_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("investigator", investigator_node)
    graph.add_node("supervisor_after_investigation", supervisor_after_investigation)
    graph.add_node("responder", responder_node)
    graph.add_node("supervisor_after_draft", supervisor_after_draft)
    graph.add_node("send", send_node)
    graph.add_node("hold", hold_node)
    graph.add_edge(START, "triage")
    graph.add_edge("triage", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route,
        {"hitl": "hitl", "investigator": "investigator"},
    )
    graph.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {"investigator": "investigator", "end": END},
    )
    graph.add_edge("investigator", "supervisor_after_investigation")
    graph.add_conditional_edges(
        "supervisor_after_investigation",
        route,
        {"responder": "responder"},
    )
    graph.add_edge("responder", "supervisor_after_draft")
    graph.add_conditional_edges(
        "supervisor_after_draft",
        route,
        {"send": "send", "hold": "hold"},
    )
    graph.add_edge("send", END)
    graph.add_edge("hold", END)
    return graph.compile(checkpointer=checkpointer or CHECKPOINTER)


def run_supervisor(
    ticket: TicketInput | dict[str, Any],
    *,
    triage_fn=None,
    investigate_fn=None,
    respond_fn=None,
    send_fn=None,
) -> SupervisorResult:
    """Run Triage and route. High-severity tickets pause until a person replies.

    The function arguments replace Triage, Investigator, Responder, or Send in tests.
    """
    validated = TicketInput.model_validate(
        ticket.model_dump() if isinstance(ticket, TicketInput) else ticket
    )
    graph = build_graph(
        triage_fn or run_triage,
        investigate_fn or run_investigation,
        respond_fn or run_response,
        send_fn or run_send,
    )
    thread_id = f"{validated.ticket_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"ticket": validated.model_dump(mode="json")}, config)
    return _result_from_graph(graph, config, thread_id)


def resume_review(
    thread_id: str,
    decision: HumanDecision | dict[str, Any],
    *,
    triage_fn=None,
    investigate_fn=None,
    respond_fn=None,
    send_fn=None,
) -> SupervisorResult:
    """Continue a paused review. Approve continues through Send. Reject stops."""
    validated = HumanDecision.model_validate(
        decision.model_dump() if isinstance(decision, HumanDecision) else decision
    )
    graph = build_graph(
        triage_fn or run_triage,
        investigate_fn or run_investigation,
        respond_fn or run_response,
        send_fn or run_send,
    )
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(Command(resume=validated.model_dump()), config)
    return _result_from_graph(graph, config, thread_id)


def _result_from_graph(graph, config: dict[str, Any], thread_id: str) -> SupervisorResult:
    snapshot = graph.get_state(config)
    values = snapshot.values
    ticket_state = TicketState.model_validate(values["ticket_state"])
    investigation = values.get("investigation")
    draft = values.get("draft")
    delivery = values.get("delivery")
    if snapshot.interrupts:
        payload = snapshot.interrupts[0].value
        review = ReviewRequest(
            status="pending",
            ticket_id=payload["ticket_id"],
            severity=payload["severity"],
            category=payload["category"],
            reason=payload["reason"],
            note="",
        )
        routed_to = "hitl"
    else:
        review = ReviewRequest.model_validate(values["review"]) if values.get("review") else None
        routed_to = values.get("routed_to", "hitl")

    result = SupervisorResult(
        ticket_state=ticket_state,
        routed_to=routed_to,
        investigation=Investigation.model_validate(investigation) if investigation else None,
        draft=ResponseDraft.model_validate(draft) if draft else None,
        delivery=Delivery.model_validate(delivery) if delivery else None,
        review=review,
        thread_id=thread_id,
    )
    _log_route(result)
    return result


def _log_route(result: SupervisorResult) -> None:
    payload = {
        "ticket_id": result.ticket_state.ticket_id,
        "severity": result.ticket_state.severity,
        "next_node_hint": result.ticket_state.next_node_hint,
        "routed_to": result.routed_to,
    }
    print(json.dumps(payload), flush=True)
