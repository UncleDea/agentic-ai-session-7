"""Investigator. Calls mock lookups, then writes sourced findings."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from triage_agent.config import Settings, get_settings
from triage_agent.llm import get_chat_model
from triage_agent.state import Investigation, Ticket, TicketState, TriageResult
from triage_agent.tools import build_tools

logger = logging.getLogger("triage_agent.investigator")

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "investigator.md"
_ORDER_ID = re.compile(r"\border\s+#?(\d{3,})\b", re.IGNORECASE)
_AMOUNT = re.compile(r"\$\d+(?:\.\d+)?")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONTH_DATE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
_ID = re.compile(r"\b[A-Za-z]{1,3}-\d+\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)*")
_LIMIT_NOTE = "Tool call limit reached. Do not call more tools."


@dataclass
class _Observation:
    tool: str
    found: bool
    records: list[dict]
    note: str


def load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_messages(ticket: Ticket, triage: TriageResult | None) -> list:
    category = triage.category.value if triage is not None else "unknown"
    system = load_prompt()
    human = (
        f"id: {ticket.id}\n"
        f"customer_id: {ticket.customer_id}\n"
        f"channel: {ticket.channel.value}\n"
        f"category: {category}\n"
        f"subject: {ticket.subject}\n"
        f"body: {ticket.body}"
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log(ticket_id: str | None, decision: str, started: float) -> None:
    logger.info(
        "node=investigator ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        decision,
        _latency_ms(started),
    )


def _tool_calls(message: Any) -> list[dict]:
    raw = getattr(message, "tool_calls", None) or []
    calls: list[dict] = []
    for call in raw:
        if isinstance(call, dict):
            calls.append(call)
            continue
        calls.append(
            {
                "name": call["name"],
                "args": call["args"],
                "id": call["id"],
            }
        )
    return calls


def _args(call: dict) -> dict:
    raw = call.get("args", {})
    if isinstance(raw, str):
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else {}
    return dict(raw)


def _call_id(call: dict, fallback: str) -> str:
    value = call.get("id")
    return value if isinstance(value, str) and value else fallback


def _observe(name: str, payload: str) -> _Observation | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "found" not in data:
        return None
    records = data.get("records") or []
    if not isinstance(records, list):
        records = []
    return _Observation(
        tool=name,
        found=bool(data["found"]),
        records=[row for row in records if isinstance(row, dict)],
        note=str(data.get("note") or ""),
    )


def _order_ids(ticket: Ticket) -> list[str]:
    text = f"{ticket.subject}\n{ticket.body}"
    found: list[str] = []
    for match in _ORDER_ID.finditer(text):
        order_id = match.group(1)
        if order_id not in found:
            found.append(order_id)
    return found


def _claim_tokens(fact: str) -> list[str]:
    tokens: list[str] = []
    for match in _AMOUNT.finditer(fact):
        tokens.append(match.group(0).lstrip("$"))
    for pattern in (_ISO_DATE, _MONTH_DATE, _ID):
        tokens.extend(pattern.findall(fact))
    tokens.extend(_NUMBER.findall(fact))
    unique: list[str] = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return unique


def _supported(fact: str, record: dict) -> bool:
    blob = json.dumps(record, ensure_ascii=False).lower()
    return all(token.lower() in blob for token in _claim_tokens(fact))


def _ids(observations: list[_Observation], tool_name: str) -> set[str]:
    found: set[str] = set()
    for obs in observations:
        if obs.tool != tool_name or not obs.found:
            continue
        for record in obs.records:
            record_id = record.get("id")
            if record_id is not None:
                found.add(str(record_id))
    return found


def _missing_info(observations: list[_Observation], ticket: Ticket) -> list[str]:
    """Only a missing customer, or a named order the lookup did not return."""
    missing: list[str] = []
    customer_calls = [obs for obs in observations if obs.tool == "get_customer"]
    if customer_calls and not any(obs.found for obs in customer_calls):
        note = next((obs.note for obs in customer_calls if obs.note), "")
        missing.append(note or f"customer {ticket.customer_id} not found")

    order_calls = [obs for obs in observations if obs.tool == "get_orders"]
    if not order_calls:
        return missing
    found_ids: set[str] = set()
    for obs in order_calls:
        for record in obs.records:
            order_id = record.get("order_id")
            if order_id is not None:
                found_ids.add(str(order_id))
    for order_id in _order_ids(ticket):
        if order_id not in found_ids:
            missing.append(f"order {order_id} not found")
    return missing


def reconcile(
    investigation: Investigation,
    observations: list[_Observation],
    ticket: Ticket,
) -> Investigation:
    """Keep sourced findings. Record gaps only for a missing customer or order."""
    sources: dict[str, dict] = {}
    for obs in observations:
        if not obs.found:
            continue
        for record in obs.records:
            source = record.get("source")
            if isinstance(source, str) and source:
                sources[source] = record

    kept = []
    dropped: list[str] = []
    for finding in investigation.findings:
        record = sources.get(finding.source)
        if record is None or not _supported(finding.fact, record):
            dropped.append(finding.fact)
            continue
        kept.append(finding)

    past_ids = _ids(observations, "search_past_tickets")
    kb_ids = _ids(observations, "search_kb")
    return Investigation(
        findings=kept,
        related_ticket_ids=[
            item for item in investigation.related_ticket_ids if item in past_ids
        ],
        kb_article_ids=[item for item in investigation.kb_article_ids if item in kb_ids],
        missing_info=_missing_info(observations, ticket),
        dropped_claims=dropped,
    )


def investigator(
    state: TicketState,
    model: Any | None = None,
    settings: Settings | None = None,
    tools: list | None = None,
) -> dict:
    """Gather facts for state['ticket']. Returns investigation or errors."""
    started = time.perf_counter()
    ticket = state.get("ticket")
    if not isinstance(ticket, Ticket):
        _log(None, "error", started)
        return {"errors": ["investigator: state has no ticket"]}
    try:
        active = settings if settings is not None else get_settings()
        chat = model if model is not None else get_chat_model()
        tool_list = tools if tools is not None else build_tools()
        tool_map = {item.name: item for item in tool_list}
        bound = chat.bind_tools(tool_list)
        messages = build_messages(ticket, state.get("triage"))
        observations: list[_Observation] = []
        calls = 0
        stopped = False
        while calls < active.max_tool_calls and not stopped:
            ai = bound.invoke(messages)
            if not isinstance(ai, AIMessage):
                ai = AIMessage(content=str(ai))
            messages.append(ai)
            pending = _tool_calls(ai)
            if not pending:
                break
            for index, call in enumerate(pending):
                call_id = _call_id(call, f"call-{calls}-{index}")
                if calls >= active.max_tool_calls:
                    messages.append(ToolMessage(content=_LIMIT_NOTE, tool_call_id=call_id))
                    stopped = True
                    continue
                name = str(call.get("name", ""))
                if name not in tool_map:
                    payload = json.dumps(
                        {"found": False, "records": [], "note": f"unknown tool {name}"}
                    )
                else:
                    output = tool_map[name].invoke(_args(call))
                    payload = output if isinstance(output, str) else json.dumps(output)
                    seen = _observe(name, payload)
                    if seen is not None:
                        observations.append(seen)
                messages.append(ToolMessage(content=payload, tool_call_id=call_id))
                calls += 1
        messages.append(
            HumanMessage(
                content=(
                    "Using only the tool results above, fill Investigation. "
                    "Copy source strings from the records. Put a customer or "
                    "named order in missing_info only when the lookup did not "
                    "find it. Leave missing_info empty when a past-ticket or "
                    "knowledge-base search has no hits."
                )
            )
        )
        structured = chat.with_structured_output(Investigation)
        raw = structured.invoke(messages)
        parsed = (
            raw if isinstance(raw, Investigation) else Investigation.model_validate(raw)
        )
        result = reconcile(parsed, observations, ticket)
        for claim in result.dropped_claims:
            logger.info(
                "node=investigator ticket_id=%s dropped_claim=%s",
                ticket.id,
                claim,
            )
        _log(
            ticket.id,
            (
                f"findings={len(result.findings)} "
                f"missing={len(result.missing_info)} "
                f"dropped={len(result.dropped_claims)}"
            ),
            started,
        )
        return {"investigation": result}
    except Exception as exc:
        _log(ticket.id, "error", started)
        return {"errors": [f"investigator: {type(exc).__name__}: {exc}"]}
