"""Review decisions for the local page. No UI toolkit."""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command
from pydantic import ValidationError

from triage_agent.nodes.hitl import HitlResume

Outcome = Literal["sent", "paused", "ended"]


def decision_from_action(
    action: str,
    text: str,
    feedback: str,
    save_as_rule: bool,
) -> HitlResume | str:
    """Build a resume value, or a short error when the action is incomplete."""
    cleaned = feedback.strip()
    try:
        return HitlResume(
            action=action,  # type: ignore[arg-type]
            text=(text or "").strip() or None,
            feedback=cleaned,
            save_as_rule=bool(save_as_rule and cleaned),
        )
    except ValidationError as exc:
        for err in exc.errors():
            inner = (err.get("ctx") or {}).get("error")
            if isinstance(inner, Exception) and inner.args:
                return str(inner.args[0])
        return "Choose approve, edit, or reject."


def resume_thread(graph, thread_id: str, decision: HitlResume) -> Outcome:
    """Resume one paused run. Returns sent, paused, or ended."""
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(Command(resume=decision.model_dump()), config, durability="sync")
    finished = graph.get_state(config)
    if finished.next == ("hitl",):
        return "paused"
    if finished.values.get("sent"):
        return "sent"
    return "ended"
