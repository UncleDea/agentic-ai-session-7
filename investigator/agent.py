"""Investigator. Looks up local ticket details and returns an Investigation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triage_agent.schemas import TicketState

from investigator.lookup import default_data_dir, investigate, load_catalog
from investigator.schemas import Investigation

__all__ = ["run_investigation"]


def run_investigation(
    ticket: TicketState | dict[str, Any],
    *,
    data_dir: Path | None = None,
) -> Investigation:
    """Search local tickets, logs, metrics, and runbooks for one triaged ticket."""
    validated = TicketState.model_validate(
        ticket.model_dump() if isinstance(ticket, TicketState) else ticket
    )
    catalog = load_catalog(data_dir or default_data_dir())
    result = investigate(validated, catalog)
    _log(validated.ticket_id, result)
    return result


def _log(ticket_id: str, result: Investigation) -> None:
    payload = {
        "ticket_id": ticket_id,
        "service": result.service,
        "related_tickets": len(result.related_ticket_ids),
        "evidence": len(result.evidence),
    }
    print(json.dumps(payload), flush=True)
