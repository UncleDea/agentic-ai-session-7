"""Run one ticket through the pipeline, and resume a paused review later."""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hitl.schemas import ReviewRequest
from investigator import Investigation, run_investigation
from responder import ResponseDraft, run_response
from send import Delivery, run_send
from send.agent import hold_reason
from supervisor.graph import run_supervisor
from supervisor.routing import choose_after_draft
from supervisor.schemas import SupervisorResult
from triage_agent import TicketInput, TicketState

_THREAD_ID = r"^[A-Za-z0-9._-]+$"


def default_review_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "reviews"


def main(
    argv: list[str] | None = None,
    *,
    triage_fn=None,
    investigate_fn=None,
    respond_fn=None,
    send_fn=None,
    review_dir: Path | None = None,
    outbox: Path | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    reviews = review_dir or default_review_dir()
    deliver = send_fn or (lambda ticket, draft: run_send(ticket, draft, outbox=outbox))
    if args.command == "run":
        return _run(args, reviews, triage_fn, investigate_fn, respond_fn, deliver)
    return _resume(args, reviews, investigate_fn, respond_fn, deliver)


def format_result(result: SupervisorResult) -> str:
    lines = [
        f"ticket_id: {result.ticket_state.ticket_id}",
        f"routed_to: {result.routed_to}",
        f"category: {result.ticket_state.category}",
        f"severity: {result.ticket_state.severity}",
    ]
    if result.review is not None:
        lines.append(f"review: {result.review.status}")
        lines.append(f"reason: {result.review.reason}")
        if result.review.note:
            lines.append(f"note: {result.review.note}")
    if result.investigation is not None:
        lines.append(f"service: {result.investigation.service}")
        if result.investigation.runbook_path:
            lines.append(f"runbook: {result.investigation.runbook_path}")
    if result.draft is not None:
        lines.append(f"subject: {result.draft.subject}")
        lines.append(f"body: {result.draft.body}")
        lines.append(f"needs_review: {str(result.draft.needs_review).lower()}")
    if result.delivery is not None:
        lines.append(f"delivery: {result.delivery.status}")
        if result.delivery.outbound_id:
            lines.append(f"outbound_id: {result.delivery.outbound_id}")
        if result.delivery.reason:
            lines.append(f"hold: {result.delivery.reason}")
    if result.review is not None and result.review.status == "pending" and result.thread_id:
        lines.append(
            "resume: python -m supervisor resume "
            f"{result.thread_id} --action approve"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m supervisor")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run one ticket")
    run.add_argument("--text", required=True)
    run.add_argument("--ticket-id", default=None)
    run.add_argument("--customer-id", default=None)
    run.add_argument("--channel", default="email")

    resume = commands.add_parser("resume", help="Approve or reject a paused ticket")
    resume.add_argument("thread_id")
    resume.add_argument("--action", required=True, choices=["approve", "reject"])
    resume.add_argument("--note", default="")
    return parser


def _run(args, review_dir: Path, triage_fn, investigate_fn, respond_fn, send_fn) -> int:
    _require_model(triage_fn is None or respond_fn is None)
    ticket_id = args.ticket_id or f"TKT-{uuid.uuid4().hex[:8].upper()}"
    _check_token(ticket_id, "ticket id")
    ticket = TicketInput(
        ticket_id=ticket_id,
        raw_text=args.text,
        customer_id=args.customer_id,
        channel=args.channel,
        received_at=datetime.now(timezone.utc),
    )
    result = run_supervisor(
        ticket,
        triage_fn=triage_fn,
        investigate_fn=investigate_fn,
        respond_fn=respond_fn,
        send_fn=send_fn,
    )
    if result.review is not None and result.review.status == "pending":
        _save_review(review_dir, result)
    print(format_result(result))
    return 0


def _resume(args, review_dir: Path, investigate_fn, respond_fn, send_fn) -> int:
    _check_token(args.thread_id, "thread id")
    stored = _load_review(review_dir, args.thread_id)
    if stored["review"]["status"] != "pending":
        print(f"review: {stored['review']['status']}")
        print("This ticket was already decided.")
        return 1
    ticket_state = TicketState.model_validate(stored["ticket_state"])
    if args.action == "reject":
        result = _rejected(ticket_state, stored["review"], args.note, args.thread_id)
        _save_review(review_dir, result)
        print(format_result(result))
        return 0
    _require_model(respond_fn is None)
    result = _approved(
        ticket_state,
        stored["review"],
        args.note,
        args.thread_id,
        investigate_fn,
        respond_fn,
        send_fn,
    )
    _save_review(review_dir, result)
    print(format_result(result))
    return 0


def _approved(
    ticket_state: TicketState,
    review: dict[str, Any],
    note: str,
    thread_id: str,
    investigate_fn,
    respond_fn,
    send_fn,
) -> SupervisorResult:
    investigate = investigate_fn or run_investigation
    respond = respond_fn or run_response
    findings = investigate(ticket_state)
    if not isinstance(findings, Investigation):
        findings = Investigation.model_validate(findings)
    draft = respond(ticket_state, findings)
    if not isinstance(draft, ResponseDraft):
        draft = ResponseDraft.model_validate(draft)
    if choose_after_draft(ticket_state, draft) == "send":
        delivery = send_fn(ticket_state, draft)
        if not isinstance(delivery, Delivery):
            delivery = Delivery.model_validate(delivery)
        routed_to = "send"
    else:
        delivery = Delivery(
            status="not_sent",
            recipient=(ticket_state.customer_id or "").strip() or None,
            subject=draft.subject,
            body=draft.body,
            reason=hold_reason(ticket_state, draft),
        )
        routed_to = "responder"
    return SupervisorResult(
        ticket_state=ticket_state,
        routed_to=routed_to,
        investigation=findings,
        draft=draft,
        delivery=delivery,
        review=_decided(review, "approved", note),
        thread_id=thread_id,
    )


def _rejected(ticket_state: TicketState, review: dict[str, Any], note: str, thread_id: str) -> SupervisorResult:
    return SupervisorResult(
        ticket_state=ticket_state,
        routed_to="hitl",
        review=_decided(review, "rejected", note),
        thread_id=thread_id,
    )


def _decided(review: dict[str, Any], status: str, note: str) -> ReviewRequest:
    return ReviewRequest(
        status=status,
        ticket_id=review["ticket_id"],
        severity=review["severity"],
        category=review["category"],
        reason=review["reason"],
        note=note,
    )


def _save_review(review_dir: Path, result: SupervisorResult) -> None:
    if result.thread_id is None or result.review is None:
        return
    review_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "thread_id": result.thread_id,
        "ticket_state": result.ticket_state.model_dump(mode="json"),
        "review": result.review.model_dump(mode="json"),
    }
    path = review_dir / f"{result.thread_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_review(review_dir: Path, thread_id: str) -> dict[str, Any]:
    path = review_dir / f"{thread_id}.json"
    if not path.is_file():
        raise SystemExit(f"No paused review for {thread_id}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_model(uses_real_model: bool) -> None:
    if not uses_real_model or os.environ.get("OPENAI_API_KEY"):
        return
    raise SystemExit("Set OPENAI_API_KEY before running the ticket models.")


def _check_token(value: str, label: str) -> None:
    if re.fullmatch(_THREAD_ID, value) is None:
        raise SystemExit(f"The {label} may only use letters, numbers, dots, underscores, and hyphens.")
