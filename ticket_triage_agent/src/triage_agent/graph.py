"""Hub-and-spoke graph. Workers return to the supervisor; send and end stop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from langgraph.graph import END, START, StateGraph

from triage_agent.config import Settings, get_settings
from triage_agent.nodes.hitl import hitl
from triage_agent.nodes.investigator import investigator
from triage_agent.nodes.responder import responder
from triage_agent.nodes.send import send
from triage_agent.nodes.supervisor import supervisor
from triage_agent.nodes.triage import triage
from triage_agent.state import TicketState

_Next = Literal["triage", "investigator", "responder", "hitl", "send", "end"]


def build_graph(
    model=None,
    settings: Settings | None = None,
    hitl_node: Callable[[TicketState], dict] | None = None,
):
    """Compile the skeleton. Pass a model to avoid constructing a live client."""
    active = settings if settings is not None else get_settings()
    review = hitl if hitl_node is None else hitl_node

    def supervisor_node(state: TicketState) -> dict:
        return supervisor(state, settings=active)

    def triage_node(state: TicketState) -> dict:
        return triage(state, model=model)

    def investigator_node(state: TicketState) -> dict:
        return investigator(state, model=model, settings=active)

    def route(state: TicketState) -> _Next:
        return cast(_Next, state["next_node"])

    graph = StateGraph(TicketState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("triage", triage_node)
    graph.add_node("investigator", investigator_node)
    graph.add_node("responder", responder)
    graph.add_node("hitl", review)
    graph.add_node("send", send)
    graph.add_edge(START, "supervisor")
    graph.add_edge("triage", "supervisor")
    graph.add_edge("investigator", "supervisor")
    graph.add_edge("responder", "supervisor")
    graph.add_edge("hitl", "supervisor")
    graph.add_edge("send", END)
    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "triage": "triage",
            "investigator": "investigator",
            "responder": "responder",
            "hitl": "hitl",
            "send": "send",
            "end": END,
        },
    )
    return graph.compile()
