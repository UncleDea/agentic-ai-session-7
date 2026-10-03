"""Offline supervisor and graph-skeleton tests. No API key is required."""

from __future__ import annotations

import importlib
import logging
from datetime import datetime, timezone

import pytest

from triage_agent.config import Settings
from triage_agent.graph import build_graph
from triage_agent.nodes.responder import responder
from triage_agent.nodes.supervisor import supervisor
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

# A looping supervisor/HITL cycle exceeds this. A terminating run does not.
_LOW_RECURSION = 6


class FakeModel:
    def __init__(self, result: TriageResult) -> None:
        self.result = result
        self.invoke_count = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.invoke_count += 1
        return self.result


def _ticket() -> Ticket:
    return Ticket(
        id="T-001",
        customer_id="C-1001",
        channel=Channel.email,
        subject="Charged twice",
        body="Please reverse the extra charge.",
        created_at=datetime(2026, 3, 3, 15, 4, tzinfo=timezone.utc),
    )


def _triage(**overrides) -> TriageResult:
    data = {
        "category": Category.billing,
        "priority": Priority.P3,
        "sentiment": Sentiment.neutral,
        "language": "en",
        "confidence": 0.9,
        "needs_human": False,
    }
    data.update(overrides)
    return TriageResult(**data)


def _state(**overrides) -> TicketState:
    state = initial_state(_ticket())
    state.update(overrides)
    return state


def _investigated(**overrides) -> TicketState:
    state = _state(
        triage=_triage(),
        investigation=Investigation(
            findings=[Finding(fact="invoice 42", source="orders")],
            missing_info=[],
        ),
        draft=Draft(reply="We refunded invoice 42.", memory_ids=[], confidence=0.95),
    )
    state.update(overrides)
    return state


def _assert_route(
    state: TicketState,
    node: str,
    reason: str,
    settings: Settings | None = None,
    escalated: bool = False,
    sent: bool | None = None,
) -> dict:
    before = state.get("iteration_count", 0)
    update = supervisor(state, settings=settings or Settings())
    assert update["next_node"] == node
    assert update["route_history"] == [RouteEntry(node=node, reason=reason)]
    assert update["iteration_count"] == before + 1
    if escalated:
        assert update["escalated"] is True
    else:
        assert "escalated" not in update
    if sent is None:
        assert "sent" not in update
    else:
        assert update["sent"] is sent
    return update


def _rejected_hitl(state: TicketState) -> dict:
    return {
        "review": Review(
            status=ReviewStatus.rejected,
            reviewer_feedback="stub rejection",
        )
    }


@pytest.fixture
def _no_live_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("get_chat_model was called")

    module = importlib.import_module("triage_agent.nodes.triage")
    monkeypatch.setattr(module, "get_chat_model", fail)


def test_no_triage_routes_to_triage() -> None:
    _assert_route(_state(), "triage", "no triage yet")


def test_spam_ends_without_reply() -> None:
    state = _investigated(
        triage=_triage(category=Category.spam, priority=Priority.P4),
    )
    _assert_route(state, "end", "spam; no reply")


@pytest.mark.parametrize("category", [Category.billing, Category.other])
def test_fact_category_without_investigation_routes_to_investigator(
    category: Category,
) -> None:
    _assert_route(
        _state(triage=_triage(category=category)),
        "investigator",
        f"{category.value} needs facts",
    )


def test_feedback_skips_investigation() -> None:
    _assert_route(
        _state(triage=_triage(category=Category.feedback)),
        "responder",
        "no facts required",
    )


def test_investigation_without_draft_routes_to_responder() -> None:
    state = _state(
        triage=_triage(),
        investigation=Investigation(findings=[Finding(fact="invoice 42", source="orders")]),
    )
    _assert_route(state, "responder", "investigation done, no draft")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"triage": _triage(priority=Priority.P1)}, "priority P1"),
        (
            {"triage": _triage(confidence=0.69)},
            "triage confidence below threshold",
        ),
        (
            {"draft": Draft(reply="stub", confidence=0.69)},
            "draft confidence below threshold",
        ),
        (
            {"triage": _triage(sentiment=Sentiment.negative)},
            "sentiment negative",
        ),
        ({"triage": _triage(category=Category.refund)}, "category refund"),
        ({"triage": _triage(category=Category.security)}, "category security"),
        ({"triage": _triage(category=Category.legal)}, "category legal"),
        (
            {
                "triage": _triage(
                    needs_human=True,
                    needs_human_reason="unintelligible",
                )
            },
            "needs_human",
        ),
        (
            {
                "investigation": Investigation(
                    findings=[Finding(fact="invoice 42", source="orders")],
                    missing_info=["order id"],
                )
            },
            "missing_info",
        ),
    ],
)
def test_hitl_criterion_routes_to_hitl(overrides: dict, reason: str) -> None:
    _assert_route(_investigated(**overrides), "hitl", reason)


def test_confidence_at_threshold_does_not_require_hitl() -> None:
    state = _investigated(
        triage=_triage(confidence=0.7),
        draft=Draft(reply="ok", confidence=0.7),
    )
    _assert_route(state, "send", "hitl criteria not met")


@pytest.mark.parametrize("revision_count", [0, 1])
def test_rejected_review_routes_to_responder_until_revision_cap(
    revision_count: int,
) -> None:
    state = _investigated(
        triage=_triage(priority=Priority.P1),
        review=Review(status=ReviewStatus.rejected, reviewer_feedback="too vague"),
        revision_count=revision_count,
    )
    _assert_route(state, "responder", "review rejected; revision")


def test_rejected_review_at_revision_cap_escalates_to_hitl() -> None:
    state = _investigated(
        review=Review(status=ReviewStatus.rejected, reviewer_feedback="too vague"),
        revision_count=2,
    )
    _assert_route(state, "hitl", "revision cap", escalated=True)


def test_approved_review_sends_before_cap_and_repeated_errors() -> None:
    state = _investigated(
        triage=_triage(priority=Priority.P1),
        review=Review(status=ReviewStatus.approved),
        iteration_count=10,
        errors=["boom", "boom"],
    )
    _assert_route(state, "send", "review approved")


def test_edited_review_sends_even_with_missing_info() -> None:
    state = _investigated(
        review=Review(
            status=ReviewStatus.edited,
            reviewer_feedback="shorter",
            edited_text="Use this text.",
        ),
        investigation=Investigation(
            findings=[Finding(fact="invoice 42", source="orders")],
            missing_info=["order id"],
        ),
        iteration_count=10,
        errors=["boom", "boom"],
    )
    _assert_route(state, "send", "review edited")


def test_draft_without_hitl_criteria_sends() -> None:
    _assert_route(_investigated(), "send", "hitl criteria not met")


def test_pending_review_follows_hitl_criteria() -> None:
    pending = Review(status=ReviewStatus.pending)
    _assert_route(
        _investigated(review=pending, triage=_triage(priority=Priority.P1)),
        "hitl",
        "priority P1",
    )
    _assert_route(
        _investigated(review=pending),
        "send",
        "hitl criteria not met",
    )


def test_iteration_cap_escalates_before_normal_routing() -> None:
    state = _state(iteration_count=10, errors=["boom", "boom"])
    update = _assert_route(state, "hitl", "iteration cap", escalated=True)
    assert update["iteration_count"] == 11


def test_below_iteration_cap_still_uses_normal_routing() -> None:
    update = _assert_route(_state(iteration_count=9), "triage", "no triage yet")
    assert update["iteration_count"] == 10


def test_unique_error_does_not_escalate() -> None:
    _assert_route(_state(errors=["boom"]), "triage", "no triage yet")
    _assert_route(_state(errors=["boom", "other"]), "triage", "no triage yet")


def test_repeated_error_escalates_to_hitl() -> None:
    _assert_route(
        _state(errors=["boom", "boom"]),
        "hitl",
        "repeated error",
        escalated=True,
    )


def test_settings_change_the_branch() -> None:
    drafted = _investigated()
    _assert_route(drafted, "send", "hitl criteria not met")
    _assert_route(
        drafted,
        "hitl",
        "triage confidence below threshold",
        settings=Settings(confidence_threshold=0.95),
    )
    billing = _state(triage=_triage())
    _assert_route(billing, "investigator", "billing needs facts")
    _assert_route(
        billing,
        "responder",
        "no facts required",
        settings=Settings(investigation_categories=[]),
    )


def test_already_sent_ends() -> None:
    _assert_route(_state(sent=True), "end", "already sent")
    _assert_route(
        _investigated(sent=True, review=Review(status=ReviewStatus.approved)),
        "end",
        "already sent",
    )


def test_escalated_approval_sends() -> None:
    state = _investigated(
        escalated=True,
        review=Review(status=ReviewStatus.approved),
        iteration_count=10,
        errors=["boom", "boom"],
    )
    _assert_route(state, "send", "review approved")


@pytest.mark.parametrize(
    "review",
    [
        None,
        Review(status=ReviewStatus.pending),
        Review(status=ReviewStatus.rejected, reviewer_feedback="no"),
    ],
)
def test_escalated_without_approval_ends_unsent(review: Review | None) -> None:
    state = _investigated(
        escalated=True,
        review=review,
        iteration_count=10,
        errors=["boom", "boom"],
    )
    _assert_route(
        state,
        "end",
        "escalated; manual handling",
        escalated=True,
        sent=False,
    )


def test_logs_destination_and_ticket_id(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="triage_agent.supervisor")
    supervisor(_state(), settings=Settings())
    assert "node=supervisor" in caplog.text
    assert "ticket_id=T-001" in caplog.text
    assert "decision=triage" in caplog.text


def test_graph_send_path_records_route_history(_no_live_model: None) -> None:
    fake = FakeModel(_triage())
    graph = build_graph(model=fake, settings=Settings())
    result = graph.invoke(_state())
    assert [entry.node for entry in result["route_history"]] == [
        "triage",
        "investigator",
        "responder",
        "send",
    ]
    assert [entry.reason for entry in result["route_history"]] == [
        "no triage yet",
        "billing needs facts",
        "investigation done, no draft",
        "hitl criteria not met",
    ]
    assert result["sent"] is True
    assert result["final_reply"] == "stub reply"
    assert result["escalated"] is False
    assert fake.invoke_count == 1


def test_graph_spam_ends_without_sending(_no_live_model: None) -> None:
    fake = FakeModel(_triage(category=Category.spam, priority=Priority.P4))
    graph = build_graph(model=fake, settings=Settings())
    result = graph.invoke(_state())
    assert [entry.node for entry in result["route_history"]] == ["triage", "end"]
    assert result["route_history"][-1].reason == "spam; no reply"
    assert result["sent"] is False
    assert result["final_reply"] is None
    assert result["escalated"] is False


def test_graph_priority_hitl_then_sends(_no_live_model: None) -> None:
    fake = FakeModel(_triage(priority=Priority.P1))
    graph = build_graph(model=fake, settings=Settings())
    result = graph.invoke(_state())
    assert [entry.node for entry in result["route_history"]] == [
        "triage",
        "investigator",
        "responder",
        "hitl",
        "send",
    ]
    assert result["route_history"][3].reason == "priority P1"
    assert result["route_history"][-1].reason == "review approved"
    assert result["sent"] is True
    assert result["escalated"] is False


def _assert_manual_stop(result: dict, first_reason: str) -> None:
    assert [entry.node for entry in result["route_history"]] == ["hitl", "end"]
    assert result["route_history"][0].reason == first_reason
    assert result["route_history"][1].reason == "escalated; manual handling"
    assert result["escalated"] is True
    assert result["sent"] is False
    assert result["next_node"] == "end"


def test_graph_iteration_cap_terminates(_no_live_model: None) -> None:
    graph = build_graph(settings=Settings(), hitl_node=_rejected_hitl)
    result = graph.invoke(
        _state(iteration_count=10),
        config={"recursion_limit": _LOW_RECURSION},
    )
    _assert_manual_stop(result, "iteration cap")


def test_graph_repeated_error_terminates(_no_live_model: None) -> None:
    graph = build_graph(settings=Settings(), hitl_node=_rejected_hitl)
    result = graph.invoke(
        _state(errors=["boom", "boom"]),
        config={"recursion_limit": _LOW_RECURSION},
    )
    _assert_manual_stop(result, "repeated error")


def test_graph_rejected_at_max_revisions_terminates(_no_live_model: None) -> None:
    graph = build_graph(settings=Settings(), hitl_node=_rejected_hitl)
    result = graph.invoke(
        _investigated(
            review=Review(status=ReviewStatus.rejected, reviewer_feedback="too vague"),
            revision_count=2,
        ),
        config={"recursion_limit": _LOW_RECURSION},
    )
    _assert_manual_stop(result, "revision cap")


@pytest.mark.xfail(
    reason="Phase 5 responder resets review and increments revision_count",
    strict=True,
)
def test_revision_resets_review_and_increments_revision_count() -> None:
    state = _investigated(
        review=Review(status=ReviewStatus.rejected, reviewer_feedback="be specific"),
        revision_count=1,
    )
    decision = supervisor(state, settings=Settings())
    assert decision["next_node"] == "responder"
    update = responder(state)
    assert "review" in update and update["review"] is None
    assert update["revision_count"] == 2
