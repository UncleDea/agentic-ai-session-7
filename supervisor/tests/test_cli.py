"""The one-ticket command. Models are stubbed, so these tests need no API key."""

import json
from pathlib import Path

from investigator.schemas import Investigation
from responder.schemas import ResponseDraft
from send.agent import run_send
from supervisor.cli import main
from triage_agent import TicketInput, TicketState
from triage_agent.schemas import TriageClassification
from triage_agent.tests.test_agent import FakeModel
from triage_agent import run_triage


def _triage_factory(severity: str, hint: str):
    model = FakeModel(
        TriageClassification(
            category="account",
            severity=severity,
            confidence=0.4 if hint == "hitl" and severity == "medium" else 0.93,
            reasoning="MFA login loop." if severity == "medium" else "Possible account takeover.",
        )
    )

    def triage(ticket: TicketInput) -> TicketState:
        return run_triage(ticket, model=model)

    return triage


def _respond(ticket: TicketState, investigation) -> ResponseDraft:
    return ResponseDraft(
        status="drafted",
        subject="Re: MFA loop",
        body="Please reopen your authenticator app.",
        domain="support",
        memory_used=["procedural"],
        needs_review=False,
    )


def _investigate(ticket: TicketState) -> Investigation:
    return Investigation(
        status="investigated",
        summary="Matched auth-service.",
        service="auth-service",
        runbook_path="support/runbooks/mfa-loop.md",
        evidence=[],
    )


def test_run_sends_a_medium_ticket(tmp_path: Path, capsys):
    outbox = tmp_path / "sent.jsonl"
    code = main(
        [
            "run",
            "--ticket-id",
            "TKT-CLI",
            "--text",
            "Authenticator codes loop back to the MFA page.",
            "--customer-id",
            "priya.s@example.com",
        ],
        triage_fn=_triage_factory("medium", "investigator"),
        investigate_fn=_investigate,
        respond_fn=_respond,
        send_fn=lambda ticket, draft: run_send(ticket, draft, outbox=outbox),
        review_dir=tmp_path / "reviews",
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "routed_to: send" in output
    assert "category: account" in output
    assert "service: auth-service" in output
    assert "Please reopen your authenticator app." in output
    record = json.loads(outbox.read_text(encoding="utf-8").strip())
    assert record["ticket_id"] == "TKT-CLI"
    assert record["recipient"] == "priya.s@example.com"


def test_run_pauses_a_high_ticket_for_review(tmp_path: Path, capsys):
    outbox = tmp_path / "sent.jsonl"
    reviews = tmp_path / "reviews"
    code = main(
        [
            "run",
            "--ticket-id",
            "TKT-HIGH",
            "--text",
            "Someone logged in from an unknown country.",
            "--customer-id",
            "priya.s@example.com",
        ],
        triage_fn=_triage_factory("high", "hitl"),
        investigate_fn=_investigate,
        respond_fn=_respond,
        send_fn=lambda ticket, draft: run_send(ticket, draft, outbox=outbox),
        review_dir=reviews,
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "routed_to: hitl" in output
    assert "review: pending" in output
    assert "python -m supervisor resume " in output
    assert not outbox.exists()
    saved = list(reviews.glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text(encoding="utf-8"))["review"]["status"] == "pending"


def test_resume_approve_sends_the_paused_ticket(tmp_path: Path, capsys):
    reviews = tmp_path / "reviews"
    outbox = tmp_path / "sent.jsonl"
    main(
        [
            "run",
            "--ticket-id",
            "TKT-HIGH",
            "--text",
            "Someone logged in from an unknown country.",
            "--customer-id",
            "priya.s@example.com",
        ],
        triage_fn=_triage_factory("high", "hitl"),
        investigate_fn=_investigate,
        respond_fn=_respond,
        send_fn=lambda ticket, draft: run_send(ticket, draft, outbox=outbox),
        review_dir=reviews,
    )
    thread_id = json.loads(next(reviews.glob("*.json")).read_text(encoding="utf-8"))["thread_id"]
    capsys.readouterr()
    code = main(
        ["resume", thread_id, "--action", "approve", "--note", "Looks real."],
        investigate_fn=_investigate,
        respond_fn=_respond,
        send_fn=lambda ticket, draft: run_send(ticket, draft, outbox=outbox),
        review_dir=reviews,
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "review: approved" in output
    assert "note: Looks real." in output
    assert "routed_to: send" in output
    assert "delivery: sent" in output
    assert outbox.is_file()


def test_resume_reject_does_not_send(tmp_path: Path, capsys):
    reviews = tmp_path / "reviews"
    outbox = tmp_path / "sent.jsonl"

    def explode(*args, **kwargs):
        raise AssertionError("should not run after rejection")

    main(
        [
            "run",
            "--ticket-id",
            "TKT-HIGH",
            "--text",
            "Someone logged in from an unknown country.",
            "--customer-id",
            "priya.s@example.com",
        ],
        triage_fn=_triage_factory("high", "hitl"),
        investigate_fn=explode,
        respond_fn=_respond,
        send_fn=explode,
        review_dir=reviews,
    )
    thread_id = json.loads(next(reviews.glob("*.json")).read_text(encoding="utf-8"))["thread_id"]
    capsys.readouterr()
    code = main(
        ["resume", thread_id, "--action", "reject", "--note", "Spam."],
        investigate_fn=explode,
        respond_fn=explode,
        send_fn=explode,
        review_dir=reviews,
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "review: rejected" in output
    assert "note: Spam." in output
    assert "routed_to: hitl" in output
    assert not outbox.exists()


def test_run_requires_text():
    try:
        main(["run"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing --text should exit")
