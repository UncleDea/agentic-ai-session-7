"""Hub-and-spoke graph. Workers return to the supervisor; send and close stop."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from enum import Enum
from typing import Literal, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel

import triage_agent.state as state_types
from triage_agent.config import Settings, get_settings
from triage_agent.memory.store import store_for_settings
from triage_agent.nodes.hitl import hitl
from triage_agent.nodes.investigator import investigator
from triage_agent.nodes.responder import responder
from triage_agent.nodes.send import close_run, send
from triage_agent.nodes.supervisor import supervisor
from triage_agent.nodes.triage import triage
from triage_agent.state import TicketState

_Next = Literal["triage", "investigator", "responder", "hitl", "send", "end"]
_AfterSend = Literal["hitl", "end"]


class PausedRun(TypedDict):
    thread_id: str
    ticket_id: str
    reasons: list[str]


def _state_serializer() -> JsonPlusSerializer:
    """Allow every model and enum defined on the ticket state.

    An explicit ``allowed_msgpack_modules`` list is what keeps those types
    loadable once unregistered checkpoint types are blocked.
    """
    allowed: list[type] = []
    for value in vars(state_types).values():
        if not isinstance(value, type) or value.__module__ != state_types.__name__:
            continue
        try:
            registered = issubclass(value, Enum) or issubclass(value, BaseModel)
        except TypeError:
            continue
        if registered:
            allowed.append(value)
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def open_checkpointer(path: str) -> SqliteSaver:
    """Persistent saver. The caller keeps the connection open."""
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=_state_serializer())


def thread_config(ticket_id: str, run_id: str | None = None) -> dict:
    """One run of a ticket. A second call gets a new thread id.

    The ticket id stays on the ticket in state. The thread id is
    ``{ticket_id}-{short run id}`` so a finished run does not block the next one.
    """
    short = run_id if run_id else uuid.uuid4().hex[:8]
    return {"configurable": {"thread_id": f"{ticket_id}-{short}"}}


def list_paused_runs(graph) -> list[PausedRun]:
    """Runs whose latest checkpoint is waiting at the HITL node."""
    checkpointer = getattr(graph, "checkpointer", None)
    if not checkpointer or checkpointer is True or not hasattr(checkpointer, "list"):
        return []
    thread_ids: list[str] = []
    for saved in list(checkpointer.list(None)):
        configurable = saved.config.get("configurable", {})
        if configurable.get("checkpoint_ns"):
            continue
        thread_id = str(configurable.get("thread_id", ""))
        if thread_id and thread_id not in thread_ids:
            thread_ids.append(thread_id)
    paused: list[PausedRun] = []
    for thread_id in thread_ids:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        if "hitl" not in snapshot.next:
            continue
        paused.append(
            {
                "thread_id": thread_id,
                "ticket_id": _ticket_id(snapshot.values),
                "reasons": _interrupt_reasons(snapshot.interrupts),
            }
        )
    paused.sort(key=lambda row: row["thread_id"])
    return paused


def build_graph(
    model=None,
    settings: Settings | None = None,
    hitl_node: Callable[[TicketState], dict] | None = None,
    store: SqliteStore | None = None,
    checkpointer: SqliteSaver | None = None,
):
    """Compile the graph. Pass a model to avoid constructing a live client.

    A checkpointer is attached when one is passed or when settings name
    ``checkpoint_db_path``. Without one, ``interrupt`` cannot pause.
    """
    active = settings if settings is not None else get_settings()
    memory = store if store is not None else store_for_settings(active)
    saver = checkpointer
    if saver is None and active.checkpoint_db_path:
        saver = open_checkpointer(active.checkpoint_db_path)

    def supervisor_node(state: TicketState) -> dict:
        return supervisor(state, settings=active)

    def triage_node(state: TicketState) -> dict:
        return triage(state, model=model)

    def investigator_node(state: TicketState) -> dict:
        return investigator(state, model=model, settings=active)

    def responder_node(state: TicketState) -> dict:
        return responder(state, model=model, settings=active, store=memory)

    def review_node(state: TicketState) -> dict:
        return hitl(state, store=memory)

    def send_node(state: TicketState, config: RunnableConfig) -> dict:
        return send(state, config=config, settings=active)

    def close_node(state: TicketState, config: RunnableConfig) -> dict:
        return close_run(state, config=config, settings=active)

    def route(state: TicketState) -> _Next:
        return cast(_Next, state["next_node"])

    def route_after_send(state: TicketState) -> _AfterSend:
        if state.get("next_node") == "hitl":
            return "hitl"
        return "end"

    review = review_node if hitl_node is None else hitl_node
    graph = StateGraph(TicketState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("triage", triage_node)
    graph.add_node("investigator", investigator_node)
    graph.add_node("responder", responder_node)
    graph.add_node("hitl", review)
    graph.add_node("send", send_node)
    graph.add_node("close", close_node)
    graph.add_edge(START, "supervisor")
    graph.add_edge("triage", "supervisor")
    graph.add_edge("investigator", "supervisor")
    graph.add_edge("responder", "supervisor")
    graph.add_edge("hitl", "supervisor")
    graph.add_edge("close", END)
    graph.add_conditional_edges(
        "send",
        route_after_send,
        {"hitl": "hitl", "end": END},
    )
    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "triage": "triage",
            "investigator": "investigator",
            "responder": "responder",
            "hitl": "hitl",
            "send": "send",
            "end": "close",
        },
    )
    if saver is None:
        return graph.compile(store=memory)
    return graph.compile(store=memory, checkpointer=saver)


def _ticket_id(values: dict) -> str:
    ticket = values.get("ticket") if isinstance(values, dict) else None
    if isinstance(ticket, dict):
        return str(ticket.get("id", ""))
    ticket_id = getattr(ticket, "id", "")
    return str(ticket_id or "")


def _interrupt_reasons(interrupts) -> list[str]:
    for item in interrupts:
        value = getattr(item, "value", None)
        if isinstance(value, dict) and isinstance(value.get("reasons"), list):
            return [str(reason) for reason in value["reasons"]]
    return []

