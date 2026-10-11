"""Local review page for paused ticket runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

from triage_agent.config import get_settings
from triage_agent.graph import build_graph, list_paused_runs
from triage_agent.review_ui import decision_from_action, resume_thread
from triage_agent.runtime_client import invoke_runtime


@st.cache_resource
def _graph():
    return build_graph(settings=get_settings())


def _label(row: dict) -> str:
    reasons = "; ".join(row["reasons"]) or "-"
    return f"{row['ticket_id']}  {row['thread_id']}  {reasons}"


def _plain(text: str) -> str:
    """Keep dollar amounts visible. Streamlit treats `$...$` as math."""
    return text.replace("$", r"\$")


def _show(payload: dict) -> None:
    ticket = payload.get("ticket") or {}
    triage = payload.get("triage") or {}
    st.subheader(f"Ticket {ticket.get('id', '')}")
    st.write(f"**Subject:** {_plain(str(ticket.get('subject', '')))}")
    st.write(_plain(str(ticket.get("body") or "")))
    st.write(
        f"Category: {triage.get('category', '')}    "
        f"Priority: {triage.get('priority', '')}"
    )
    reasons = payload.get("reasons") or []
    st.write("Why review: " + ("; ".join(reasons) if reasons else "-"))
    st.write("**Findings**")
    findings = payload.get("findings") or []
    if not findings:
        st.write("(none)")
    for item in findings:
        if not isinstance(item, dict):
            continue
        fact = _plain(str(item.get("fact", "")))
        source = _plain(str(item.get("source", "")))
        st.write(f"- {fact} ({source})")
    flagged = [str(item) for item in (payload.get("flagged_claims") or [])]
    st.write("Flagged claims: " + (", ".join(flagged) if flagged else "(none)"))
    st.write("Escalates: " + ("yes" if payload.get("escalates") else "no"))


def _notice(outcome: str, ticket_id: str) -> str:
    if outcome == "sent":
        return f"Sent {ticket_id}."
    if outcome == "paused":
        return "Paused again for review."
    return f"Ended {ticket_id}."


def _remote() -> bool:
    return bool(os.environ.get("AGENT_RUNTIME_ARN", "").strip())


def _allowed() -> bool:
    """The public page asks for the review token before it shows a ticket."""
    if not _remote():
        return True
    expected = os.environ.get("REVIEW_TOKEN", "").strip()
    if not expected:
        st.error("REVIEW_TOKEN is not set.")
        return False
    entered = st.text_input("Review token", type="password", key="review_token")
    if entered != expected:
        st.info("Enter the review token to see paused tickets.")
        return False
    return True


def _paused_rows(graph):
    if _remote():
        body = invoke_runtime({"action": "list"})
        if body.get("error"):
            st.error(str(body["error"]))
            return None
        return list(body.get("paused") or [])
    return list_paused_runs(graph)


def _review_payload(graph, thread_id: str) -> dict | None:
    if _remote():
        body = invoke_runtime({"action": "get", "thread_id": thread_id})
        if body.get("error"):
            st.warning(str(body["error"]))
            return None
        payload = body.get("payload")
        return payload if isinstance(payload, dict) else None
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.interrupts:
        st.warning("That run is not waiting for review.")
        return None
    value = snapshot.interrupts[0].value
    return value if isinstance(value, dict) else None


def _finish(graph, thread_id: str, decision) -> str:
    if _remote():
        body = invoke_runtime(
            {
                "action": "resume",
                "thread_id": thread_id,
                "decision_action": decision.action,
                "text": decision.text or "",
                "feedback": decision.feedback,
                "save_as_rule": decision.save_as_rule,
            }
        )
        if body.get("error"):
            raise RuntimeError(str(body["error"]))
        return str(body.get("outcome") or "ended")
    return resume_thread(graph, thread_id, decision)


def main() -> None:
    st.set_page_config(page_title="Ticket review", layout="centered")
    st.title("Ticket review")
    notice = st.session_state.pop("notice", "")
    if notice:
        st.success(notice)
    if not _allowed():
        return

    graph = None if _remote() else _graph()
    paused = _paused_rows(graph)
    if paused is None:
        return
    if not paused:
        st.info("No paused runs.")
        return

    labels = [_label(row) for row in paused]
    choice = st.selectbox("Paused run", labels, key="paused_run")
    chosen = paused[labels.index(choice)]
    thread_id = chosen["thread_id"]
    payload = _review_payload(graph, thread_id)
    if payload is None:
        return
    if st.session_state.get("loaded_thread") != thread_id:
        st.session_state["loaded_thread"] = thread_id
        st.session_state["reply_text"] = str(payload.get("draft") or "")
        st.session_state["feedback_text"] = ""
        st.session_state["save_rule"] = False

    _show(payload)
    st.text_area("Reply", key="reply_text", height=220)
    st.text_area("Feedback", key="feedback_text")
    st.checkbox("Save feedback as a rule", key="save_rule")

    approve, edit, reject = st.columns(3)
    action = None
    if approve.button("Approve", type="primary"):
        action = "approve"
    elif edit.button("Edit"):
        action = "edit"
    elif reject.button("Reject"):
        action = "reject"
    if action is None:
        return

    feedback = str(st.session_state.get("feedback_text") or "")
    save_rule = bool(st.session_state.get("save_rule")) and bool(feedback.strip())
    decision = decision_from_action(
        action,
        str(st.session_state.get("reply_text") or ""),
        feedback,
        save_rule,
    )
    if isinstance(decision, str):
        st.error(decision)
        return

    try:
        outcome = _finish(graph, thread_id, decision)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    st.session_state["notice"] = _notice(outcome, chosen["ticket_id"])
    st.session_state.pop("loaded_thread", None)
    st.rerun()


if __name__ == "__main__":
    main()
