"""Offline send and escalation tests. No API key is required."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig

from triage_agent.config import Settings
from triage_agent.graph import build_graph, open_checkpointer, thread_config
from triage_agent.nodes.send import close_run, final_reply_text, send
from triage_agent.state import (
    Category,
    Channel,
    Draft,
    Finding,
    Investigation,
    Priority,
    Review,
    ReviewStatus,
    RouteEntry,
    Sentiment,
    Ticket,
    TicketState,
    TriageResult,
    initial_state,
)
from triage_agent.tools.sender import (
    EscalationRecord,
    JsonEscalationLog,
    OutboundMessage,
    OutboxSender,
)

_REPLY = "Thanks for writing. We are looking into this."
_ESCALATION = "I'm escalating this to a specialist, who will follow up."
_EDITED = "Here is the reply the reviewer wrote."
_SENT_AT = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


class _CrashBeforeOutbox:
    """Raises after the caller has had a chance to record the escalation."""

    def get(self, thread_id: str) -> None:
        return None

    def send(self, message: OutboundMessage) -> OutboundMessage:
        raise RuntimeError("crash before outbox")


class _CrashBeforeEscalation:
    def get(self, thread_id: str) -> None:
        return None

    def open_case(self, record: EscalationRecord) -> EscalationRecord:
        raise RuntimeError("crash before escalation")


class _CountingSender:
    def __init__(self) -> None:
        self.sends = 0

    def get(self, thread_id: str) -> None:
        return None

    def send(self, message: OutboundMessage) -> OutboundMessage:
        self.sends += 1
        return message


def _settings(root: Path) -> Settings:
    return Settings(
        outbox_dir=str(root / "outbox"),
        escalations_dir=str(root / "escalations"),
    )


def _ticket() -> Ticket:
    return Ticket(
        id="T-001",
        customer_id="C-1001",
        channel=Channel.email,
        subject="Charged twice",
        body="Please reverse the extra charge.",
        created_at=datetime(2026, 3, 3, 15, 4, tzinfo=timezone.utc),
    )


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _ready(
    *,
    reply: str = _REPLY,
    escalates: bool = False,
    flagged_claims: list[str] | None = None,
    review: Review | None = None,
    sent: bool = False,
) -> TicketState:
    state = initial_state(_ticket())
    state["triage"] = TriageResult(
        category=Category.billing,
        priority=Priority.P3,
        sentiment=Sentiment.neutral,
        language="en",
        confidence=0.9,
        needs_human=False,
    )
    state["investigation"] = Investigation(
        findings=[Finding(fact="invoice 42 is open", source="orders")],
    )
    state["draft"] = Draft(
        reply=reply,
        confidence=0.95,
        escalates=escalates,
        flagged_claims=list(flagged_claims or []),
    )
    if review is not None:
        state["review"] = review
    if sent:
        state["sent"] = True
        state["final_reply"] = reply
    return state


def _approved(reply: str = _REPLY, *, escalates: bool = False, flagged: list[str] | None = None) -> TicketState:
    return _ready(
        reply=reply,
        escalates=escalates,
        flagged_claims=flagged,
        review=Review(status=ReviewStatus.approved),
    )


def _files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_reply_uses_edited_text_when_present() -> None:
    edited = _ready(review=Review(status=ReviewStatus.edited, edited_text=_EDITED))
    assert final_reply_text(edited) == _EDITED
    assert final_reply_text(_approved()) == _REPLY


def test_clean_draft_writes_one_outbox_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    update = send(_ready(), config=_config("T-001-clean"), settings=settings)
    assert update["sent"] is True
    assert update["final_reply"] == _REPLY
    assert update["sent_at"] is not None
    files = _files(tmp_path / "outbox")
    assert len(files) == 1
    record = _record(files[0])
    assert record["ticket_id"] == "T-001"
    assert record["thread_id"] == "T-001-clean"
    assert record["final_reply"] == _REPLY
    assert record["sent_at"]
    assert _files(tmp_path / "escalations") == []


def test_edited_review_sends_the_edited_text(tmp_path: Path) -> None:
    state = _ready(
        flagged_claims=["$99.00"],
        escalates=True,
        review=Review(status=ReviewStatus.edited, edited_text=_EDITED),
    )
    update = send(state, config=_config("T-001-edit"), settings=_settings(tmp_path))
    assert update["final_reply"] == _EDITED
    assert _record(_files(tmp_path / "outbox")[0])["final_reply"] == _EDITED
    assert _files(tmp_path / "escalations") == []


def test_approved_review_sends_the_draft(tmp_path: Path) -> None:
    update = send(_approved(), config=_config("T-001-ok"), settings=_settings(tmp_path))
    assert update["final_reply"] == _REPLY
    assert _record(_files(tmp_path / "outbox")[0])["final_reply"] == _REPLY


def test_sent_flag_does_not_call_the_sender(tmp_path: Path) -> None:
    sender = _CountingSender()
    update = send(
        _ready(sent=True),
        config=_config("T-001-flag"),
        settings=_settings(tmp_path),
        sender=sender,
    )
    assert sender.sends == 0
    assert update == {}
    assert _files(tmp_path / "outbox") == []


def test_existing_outbox_file_is_not_rewritten(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    config = _config("T-001-once")
    send(_ready(), config=config, settings=settings)
    original = _files(tmp_path / "outbox")[0].read_bytes()
    again = send(_ready(), config=config, settings=settings)
    assert again["sent"] is True
    assert again["final_reply"] == _REPLY
    assert _files(tmp_path / "outbox")[0].read_bytes() == original
    assert len(_files(tmp_path / "outbox")) == 1


def test_same_ticket_two_runs_writes_two_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    send(_ready(), config=_config("T-001-a"), settings=settings)
    send(_ready(), config=_config("T-001-b"), settings=settings)
    files = _files(tmp_path / "outbox")
    assert [path.name for path in files] == ["T-001-a.json", "T-001-b.json"]
    assert { _record(path)["ticket_id"] for path in files } == {"T-001"}


@pytest.mark.parametrize(
    "review",
    [
        None,
        Review(status=ReviewStatus.pending),
        Review(status=ReviewStatus.rejected, reviewer_feedback="no"),
    ],
)
def test_unapproved_flagged_or_escalating_draft_is_refused(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    review: Review | None,
) -> None:
    caplog.set_level(logging.INFO, logger="triage_agent.send")
    settings = _settings(tmp_path)
    flagged = _ready(flagged_claims=["$99.00"], review=review)
    escalating = _ready(reply=_ESCALATION, escalates=True, review=review)
    for state in (flagged, escalating):
        thread_id = f"T-001-{id(state)}"
        update = send(state, config=_config(thread_id), settings=settings)
        assert update["next_node"] == "hitl"
        assert update["route_history"] == [
            RouteEntry(node="hitl", reason="unapproved flagged or escalating draft")
        ]
        assert "sent" not in update
    assert "decision=refused" in caplog.text
    assert _files(tmp_path / "outbox") == []
    assert _files(tmp_path / "escalations") == []


def test_approved_flagged_draft_sends_without_an_escalation(tmp_path: Path) -> None:
    state = _approved(flagged=["$99.00"])
    update = send(state, config=_config("T-001-flagged"), settings=_settings(tmp_path))
    assert update["sent"] is True
    assert len(_files(tmp_path / "outbox")) == 1
    assert _files(tmp_path / "escalations") == []


def test_escalating_reply_records_the_case_before_the_outbox(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cases = JsonEscalationLog(tmp_path / "escalations")
    seen: list[bool] = []

    class _Watch:
        def get(self, thread_id: str):
            return OutboxSender(tmp_path / "outbox").get(thread_id)

        def send(self, message: OutboundMessage) -> OutboundMessage:
            seen.append(cases.get(message.thread_id) is not None)
            return OutboxSender(tmp_path / "outbox").send(message)

    update = send(
        _approved(_ESCALATION, escalates=True),
        config=_config("T-001-esc"),
        settings=settings,
        sender=_Watch(),
        escalations=cases,
    )
    assert seen == [True]
    assert update["final_reply"] == _ESCALATION
    record = _record(_files(tmp_path / "escalations")[0])
    assert record["ticket_id"] == "T-001"
    assert record["thread_id"] == "T-001-esc"
    assert record["reasons"] == ["escalation promised"]
    assert record["reply"] == _ESCALATION
    assert record["status"] == "open"
    assert record["created_at"]
    assert len(_files(tmp_path / "outbox")) == 1


def test_edit_that_drops_the_escalation_sentence_does_not_open_a_case(tmp_path: Path) -> None:
    state = _ready(
        reply=_ESCALATION,
        escalates=True,
        review=Review(status=ReviewStatus.edited, edited_text=_EDITED),
    )
    send(state, config=_config("T-001-drop"), settings=_settings(tmp_path))
    assert len(_files(tmp_path / "outbox")) == 1
    assert _files(tmp_path / "escalations") == []


def test_rerun_after_a_crash_between_escalation_and_outbox(tmp_path: Path) -> None:
    """A crash on either side of the two writes still leaves one file of each."""
    _forward_crash(tmp_path / "forward")
    _reverse_crash(tmp_path / "reverse")


def _forward_crash(root: Path) -> None:
    settings = _settings(root)
    cases = JsonEscalationLog(root / "escalations")
    config = _config("T-001-forward")
    state = _approved(_ESCALATION, escalates=True)
    crashed = send(
        state,
        config=config,
        settings=settings,
        sender=_CrashBeforeOutbox(),
        escalations=cases,
    )
    assert crashed["errors"] == ["send: RuntimeError: crash before outbox"]
    escalation_files = _files(root / "escalations")
    assert len(escalation_files) == 1
    assert _files(root / "outbox") == []
    original = escalation_files[0].read_bytes()

    finished = send(state, config=config, settings=settings, escalations=cases)
    assert finished["sent"] is True
    assert finished["final_reply"] == _ESCALATION
    assert len(_files(root / "outbox")) == 1
    assert len(_files(root / "escalations")) == 1
    assert escalation_files[0].read_bytes() == original

    outbox_bytes = _files(root / "outbox")[0].read_bytes()
    send(state, config=config, settings=settings, escalations=cases)
    assert len(_files(root / "outbox")) == 1
    assert len(_files(root / "escalations")) == 1
    assert _files(root / "outbox")[0].read_bytes() == outbox_bytes
    assert escalation_files[0].read_bytes() == original


def _reverse_crash(root: Path) -> None:
    settings = _settings(root)
    config = _config("T-001-reverse")
    planted = OutboundMessage(
        ticket_id="T-001",
        thread_id="T-001-reverse",
        final_reply=_ESCALATION,
        sent_at=_SENT_AT,
    )
    OutboxSender(root / "outbox").send(planted)
    outbox_bytes = _files(root / "outbox")[0].read_bytes()
    assert _files(root / "escalations") == []

    crashed = send(
        _approved(_ESCALATION, escalates=True),
        config=config,
        settings=settings,
        escalations=_CrashBeforeEscalation(),
    )
    assert crashed["errors"] == ["send: RuntimeError: crash before escalation"]
    assert _files(root / "outbox")[0].read_bytes() == outbox_bytes
    assert _files(root / "escalations") == []

    finished = send(
        _approved(_ESCALATION, escalates=True),
        config=config,
        settings=settings,
    )
    assert finished["sent"] is True
    assert finished["final_reply"] == _ESCALATION
    assert finished["sent_at"] == _SENT_AT
    assert len(_files(root / "outbox")) == 1
    assert len(_files(root / "escalations")) == 1
    assert _files(root / "outbox")[0].read_bytes() == outbox_bytes
    case = EscalationRecord.model_validate(_record(_files(root / "escalations")[0]))
    assert case.reasons == ["escalation promised"]
    assert case.reply == _ESCALATION
    assert case.status == "open"
    assert case.created_at == _SENT_AT

    case_bytes = _files(root / "escalations")[0].read_bytes()
    send(_approved(_ESCALATION, escalates=True), config=config, settings=settings)
    assert len(_files(root / "outbox")) == 1
    assert len(_files(root / "escalations")) == 1
    assert _files(root / "outbox")[0].read_bytes() == outbox_bytes
    assert _files(root / "escalations")[0].read_bytes() == case_bytes


def test_close_run_records_an_unsent_escalation(tmp_path: Path) -> None:
    state = _ready(reply=_ESCALATION, escalates=True)
    state["escalated"] = True
    state["route_history"] = [RouteEntry(node="hitl", reason="iteration cap")]
    close_run(state, config=_config("T-001-cap"), settings=_settings(tmp_path))
    assert _files(tmp_path / "outbox") == []
    files = _files(tmp_path / "escalations")
    assert len(files) == 1
    record = _record(files[0])
    assert record["reasons"] == ["iteration cap", "escalation promised"]
    assert record["reply"] == _ESCALATION
    assert record["status"] == "open"
    original = files[0].read_bytes()
    close_run(state, config=_config("T-001-cap"), settings=_settings(tmp_path))
    assert files[0].read_bytes() == original
    assert len(_files(tmp_path / "escalations")) == 1


@pytest.mark.parametrize(
    "reason",
    ["iteration cap", "repeated error", "revision cap"],
)
def test_close_run_uses_the_hitl_reason(tmp_path: Path, reason: str) -> None:
    root = tmp_path / reason.replace(" ", "-")
    state = _ready()
    state["escalated"] = True
    state["route_history"] = [RouteEntry(node="hitl", reason=reason)]
    close_run(state, config=_config("T-001-why"), settings=_settings(root))
    assert _record(_files(root / "escalations")[0])["reasons"] == [reason]
    assert _files(root / "outbox") == []


def test_close_run_skips_spam_and_a_sent_reply(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    spam = _ready()
    spam["escalated"] = False
    close_run(spam, config=_config("T-001-spam"), settings=settings)
    sent = _ready(reply=_ESCALATION, escalates=True, sent=True)
    sent["escalated"] = True
    close_run(sent, config=_config("T-001-sent"), settings=settings)
    assert _files(tmp_path / "escalations") == []
    assert _files(tmp_path / "outbox") == []


def test_missing_ticket_or_thread_writes_nothing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert send({}, settings=settings)["errors"] == ["send: state has no ticket"]
    assert send(_ready(), config={}, settings=settings)["errors"] == ["send: missing thread_id"]
    empty = _ready()
    empty["draft"] = Draft(reply="   ", confidence=0.95)
    assert send(empty, config=_config("T-001-empty"), settings=settings)["errors"] == [
        "send: no reply"
    ]
    assert _files(tmp_path / "outbox") == []
    assert _files(tmp_path / "escalations") == []


def test_graph_sends_the_real_reply(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    graph = build_graph(settings=settings)
    result = graph.invoke(_ready(), thread_config("T-001", "graph"))
    assert result["final_reply"] == _REPLY
    assert result["sent"] is True
    assert [entry.node for entry in result["route_history"]] == ["send"]
    assert len(_files(tmp_path / "outbox")) == 1
    assert _files(tmp_path / "escalations") == []


def test_graph_refuses_an_unapproved_flagged_draft(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    saver = open_checkpointer(str(tmp_path / "checkpoints.sqlite"))
    try:
        graph = build_graph(settings=settings, checkpointer=saver)
        config = thread_config("T-001", "flag")
        graph.invoke(_ready(flagged_claims=["$99.00"]), config, durability="sync")
        paused = graph.get_state(config)
    finally:
        saver.conn.close()
    assert paused.next == ("hitl",)
    assert paused.values["sent"] is False
    assert paused.interrupts[0].value["reasons"] == [
        "unapproved flagged or escalating draft"
    ]
    assert _files(tmp_path / "outbox") == []


def test_graph_unsent_escalation_writes_one_case(tmp_path: Path) -> None:
    def _rejected(state: TicketState) -> dict:
        return {
            "review": Review(
                status=ReviewStatus.rejected,
                reviewer_feedback="stub rejection",
            )
        }

    settings = _settings(tmp_path)
    graph = build_graph(settings=settings, hitl_node=_rejected)
    state = _ready()
    state["iteration_count"] = 10
    result = graph.invoke(state, thread_config("T-001", "cap"))
    assert result["escalated"] is True
    assert result["sent"] is False
    assert result["next_node"] == "end"
    assert _files(tmp_path / "outbox") == []
    files = _files(tmp_path / "escalations")
    assert len(files) == 1
    assert _record(files[0])["reasons"] == ["iteration cap"]
    assert _record(files[0])["status"] == "open"


def test_logs_a_successful_send(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="triage_agent.send")
    send(_ready(), config=_config("T-001-log"), settings=_settings(tmp_path))
    assert "node=send" in caplog.text
    assert "ticket_id=T-001" in caplog.text
    assert "decision=sent" in caplog.text
