"""Triage node. Classifies one ticket into TriageResult."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from triage_agent.llm import get_chat_model
from triage_agent.state import Ticket, TicketState, TriageResult

logger = logging.getLogger("triage_agent.triage")

_PACKAGE = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_PATH = _PACKAGE / "prompts" / "triage.md"
_SAMPLES_PATH = _ROOT / "data" / "sample_tickets.json"

# Labeled examples are clear tickets. Confidence is not an expected label.
_FEW_SHOT_CONFIDENCE = 0.9
_LABEL_FIELDS = ("category", "priority", "sentiment", "language", "needs_human")


def load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_sample_tickets() -> list[dict]:
    return json.loads(_SAMPLES_PATH.read_text(encoding="utf-8"))


def example_label(record: dict) -> dict:
    """Build a TriageResult payload from explicit expected fields.

    Sentiment, language, priority, and needs_human must already be on the
    record. This function does not fill those in.
    """
    expected = record["expected"]
    missing = [field for field in _LABEL_FIELDS if field not in expected]
    if missing:
        raise KeyError(f"{record['id']} expected is missing {', '.join(missing)}")
    needs_human = expected["needs_human"]
    reason = expected.get("needs_human_reason", "")
    if needs_human and not str(reason).strip():
        raise ValueError(f"{record['id']} needs_human requires needs_human_reason")
    label = {
        "category": expected["category"],
        "priority": expected["priority"],
        "sentiment": expected["sentiment"],
        "language": expected["language"],
        "confidence": _FEW_SHOT_CONFIDENCE,
        "needs_human": needs_human,
        "needs_human_reason": reason if needs_human else "",
        "secondary_categories": list(expected.get("secondary_categories", [])),
    }
    TriageResult.model_validate(label)
    return label


def build_few_shot(samples: list[dict], exclude_id: str | None) -> str:
    """Examples from labeled sample tickets, excluding the one being classified."""
    blocks: list[str] = []
    for record in samples:
        if "category" not in record.get("expected", {}):
            continue
        if record["id"] == exclude_id:
            continue
        label = example_label(record)
        blocks.append(
            "\n".join(
                [
                    f"Example {record['id']}",
                    f"Subject: {record['subject']}",
                    f"Body: {record['body']}",
                    json.dumps(label, ensure_ascii=False),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_messages(ticket: Ticket, samples: list[dict] | None = None) -> list:
    records = samples if samples is not None else load_sample_tickets()
    system = load_prompt() + "\n\nExamples:\n\n" + build_few_shot(records, ticket.id)
    human = (
        f"id: {ticket.id}\n"
        f"channel: {ticket.channel.value}\n"
        f"subject: {ticket.subject}\n"
        f"body: {ticket.body}"
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log(ticket_id: str | None, decision: str, started: float) -> None:
    logger.info(
        "node=triage ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        decision,
        _latency_ms(started),
    )


def triage(state: TicketState, model: Any | None = None) -> dict:
    """Classify state['ticket']. Returns {'triage': ...} or {'errors': [...]}."""
    started = time.perf_counter()
    ticket = state.get("ticket")
    if not isinstance(ticket, Ticket):
        _log(None, "error", started)
        return {"errors": ["triage: state has no ticket"]}
    try:
        chat = model if model is not None else get_chat_model()
        structured = chat.with_structured_output(TriageResult)
        raw = structured.invoke(build_messages(ticket))
        result = (
            raw if isinstance(raw, TriageResult) else TriageResult.model_validate(raw)
        )
        _log(ticket.id, result.category.value, started)
        return {"triage": result}
    except Exception as exc:
        _log(ticket.id, "error", started)
        return {"errors": [f"triage: {type(exc).__name__}: {exc}"]}
