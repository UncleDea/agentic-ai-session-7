"""Run a ticket, review a paused draft, or inspect rules and escalations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from langgraph.types import Command  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from triage_agent.cli import (  # noqa: E402
    absolute_settings,
    check_startup,
    find_ticket,
    format_review,
    format_summary,
    format_versions,
    load_tickets,
    row_from_values,
    ticket_from,
)
from triage_agent.config import get_settings  # noqa: E402
from triage_agent.graph import build_graph, list_paused_runs, thread_config  # noqa: E402
from triage_agent.memory.procedural import rollback  # noqa: E402
from triage_agent.memory.store import store_for_settings  # noqa: E402
from triage_agent.nodes.hitl import HitlResume  # noqa: E402
from triage_agent.state import initial_state  # noqa: E402
from triage_agent.tools.sender import list_open  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ticket triage agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one ticket or every sample ticket")
    choice = run.add_mutually_exclusive_group(required=True)
    choice.add_argument("--ticket-id")
    choice.add_argument("--all", action="store_true")

    sub.add_parser("review", help="Approve, edit, or reject a paused run")
    sub.add_parser("check", help="Check settings and the dev memory store")

    rules = sub.add_parser("rules", help="List procedure versions")
    rules_sub = rules.add_subparsers(dest="rules_command")
    rollback_cmd = rules_sub.add_parser("rollback", help="Point current at a version")
    rollback_cmd.add_argument("version", type=int)

    sub.add_parser("escalations", help="List open escalation records")
    return parser


def _startup(settings):
    try:
        lines = check_startup(settings)
    except SystemExit as exc:
        text = exc.code if isinstance(exc.code, str) else ""
        if text:
            print(text)
        return None
    print("\n".join(lines))
    return settings


def _graph(settings):
    return build_graph(settings=settings)


def _read_block() -> str:
    print("Edited reply. End with a line that is only .")
    lines: list[str] = []
    while True:
        line = input()
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _decision() -> HitlResume:
    while True:
        action = input("Action [approve/edit/reject]: ").strip().lower()
        if action not in {"approve", "edit", "reject"}:
            print("Choose approve, edit, or reject.")
            continue
        text = None
        feedback = ""
        if action == "edit":
            text = _read_block()
            if not text:
                print("edit requires text")
                continue
            feedback = input("Feedback (optional): ").strip()
        elif action == "reject":
            feedback = input("Feedback: ").strip()
            if not feedback:
                print("reject requires feedback")
                continue
        else:
            feedback = input("Feedback (optional): ").strip()
        save = False
        if feedback:
            answer = input("Save feedback as a rule? [y/N]: ").strip().lower()
            save = answer in {"y", "yes"}
        try:
            return HitlResume(
                action=action,
                text=text,
                feedback=feedback,
                save_as_rule=save,
            )
        except ValidationError as exc:
            print(exc)


def cmd_run(settings, ticket_id: str | None, all_tickets: bool) -> int:
    if all_tickets:
        tickets = [ticket_from(record) for record in load_tickets()]
    else:
        try:
            tickets = [find_ticket(ticket_id or "")]
        except SystemExit as exc:
            text = exc.code if isinstance(exc.code, str) else ""
            if text:
                print(text)
            return 1
    graph = _graph(settings)
    rows = []
    failed = False
    for ticket in tickets:
        config = thread_config(ticket.id)
        try:
            graph.invoke(initial_state(ticket), config, durability="sync")
        except Exception as exc:
            print(f"{ticket.id} failed: {type(exc).__name__}")
            failed = True
            continue
        snapshot = graph.get_state(config)
        row = row_from_values(dict(snapshot.values), ticket.id)
        rows.append(row)
        errors = list(snapshot.values.get("errors") or [])
        if errors:
            failed = True
            print(f"{ticket.id} errors: {len(errors)}")
        elif snapshot.next == ("hitl",):
            print(f"{ticket.id} paused for review. Run: python scripts/run_ticket.py review")
    print(format_summary(rows, totals=all_tickets))
    return 1 if failed else 0


def cmd_review(settings) -> int:
    graph = _graph(settings)
    paused = list_paused_runs(graph)
    if not paused:
        print("No paused runs.")
        return 0
    for index, row in enumerate(paused, start=1):
        reasons = "; ".join(row["reasons"]) or "-"
        print(f"{index}. {row['ticket_id']}  {row['thread_id']}  {reasons}")
    picked = input("Pick a run: ").strip()
    if not picked.isdigit() or not 1 <= int(picked) <= len(paused):
        print("Unknown run.")
        return 1
    chosen = paused[int(picked) - 1]
    config = {"configurable": {"thread_id": chosen["thread_id"]}}
    snapshot = graph.get_state(config)
    if not snapshot.interrupts:
        print("That run is not waiting for review.")
        return 1
    print(format_review(snapshot.interrupts[0].value))
    decision = _decision()
    graph.invoke(Command(resume=decision.model_dump()), config, durability="sync")
    finished = graph.get_state(config)
    if finished.next == ("hitl",):
        print("Paused again for review.")
    elif finished.values.get("sent"):
        print(f"Sent {chosen['ticket_id']}.")
    else:
        print(f"Ended {chosen['ticket_id']}.")
    return 0


def cmd_rules(settings, version: int | None) -> int:
    store = store_for_settings(settings)
    try:
        if version is None:
            print(format_versions(store))
            return 0
        try:
            saved = rollback(store, version)
        except LookupError as exc:
            print(str(exc))
            return 1
        print(f"Current procedure is now v{saved['version']}")
        return 0
    finally:
        store.conn.close()


def cmd_escalations(settings) -> int:
    records, skipped = list_open(Path(settings.escalations_dir))
    for name in skipped:
        print(f"Skipping {name}")
    if not records:
        print("No open escalations.")
        return 0
    for record in records:
        reasons = "; ".join(record.reasons)
        print(
            f"{record.ticket_id}  {record.thread_id}  {reasons}  "
            f"{record.created_at.isoformat()}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = absolute_settings(get_settings())
    if args.command == "escalations":
        return cmd_escalations(settings)
    if _startup(settings) is None:
        return 1
    if args.command == "check":
        return 0
    if args.command == "run":
        return cmd_run(settings, args.ticket_id, args.all)
    if args.command == "review":
        return cmd_review(settings)
    if args.command == "rules":
        version = args.version if args.rules_command == "rollback" else None
        return cmd_rules(settings, version)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
