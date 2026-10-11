"""Customer, order, and past-ticket lookups. JSON mocks sit behind protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

_DATA = Path(__file__).resolve().parents[3] / "data"


class LookupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    records: list[dict] = Field(default_factory=list)
    note: str = ""


class CustomerStore(Protocol):
    def get_customer(self, customer_id: str) -> LookupResult: ...


class OrderStore(Protocol):
    def get_orders(self, customer_id: str) -> LookupResult: ...


class PastTicketStore(Protocol):
    def search_past_tickets(self, customer_id: str, query: str) -> LookupResult: ...


class ChargeStore(Protocol):
    def list_charges(self, order_id: str) -> LookupResult: ...


class RefundStore(Protocol):
    def list_refunds(self, order_id: str) -> LookupResult: ...


class RunbookStore(Protocol):
    def search_runbook(self, query: str) -> LookupResult: ...


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


class JsonCustomerStore:
    def __init__(self, path: Path | None = None) -> None:
        self._rows = _load(path or _DATA / "customers.json")

    def get_customer(self, customer_id: str) -> LookupResult:
        matches = [row for row in self._rows if row["customer_id"] == customer_id]
        if not matches:
            return LookupResult(
                found=False,
                records=[],
                note=f"customer {customer_id} not found",
            )
        return LookupResult(found=True, records=matches)


class JsonOrderStore:
    def __init__(self, path: Path | None = None) -> None:
        self._rows = _load(path or _DATA / "orders.json")

    def get_orders(self, customer_id: str) -> LookupResult:
        matches = [row for row in self._rows if row["customer_id"] == customer_id]
        return LookupResult(found=True, records=matches)


class JsonPastTicketStore:
    def __init__(self, path: Path | None = None) -> None:
        self._rows = _load(path or _DATA / "past_tickets.json")

    def search_past_tickets(self, customer_id: str, query: str) -> LookupResult:
        from triage_agent.tools.knowledge_base import rank_keywords

        owned = [row for row in self._rows if row["customer_id"] == customer_id]
        matches = rank_keywords(
            query,
            owned,
            lambda row: str(row.get("subject", "")),
            lambda row: str(row.get("body", "")),
        )
        if not matches:
            return LookupResult(found=False, records=[], note="no matching past tickets")
        return LookupResult(found=True, records=matches)


class JsonChargeStore:
    def __init__(self, path: Path | None = None) -> None:
        self._rows = _load(path or _DATA / "charges.json")

    def list_charges(self, order_id: str) -> LookupResult:
        matches = [row for row in self._rows if row["order_id"] == order_id]
        if not matches:
            return LookupResult(
                found=False,
                records=[],
                note=f"order {order_id} has no charges",
            )
        return LookupResult(found=True, records=matches)


class JsonRefundStore:
    def __init__(self, path: Path | None = None) -> None:
        self._rows = _load(path or _DATA / "refunds.json")

    def list_refunds(self, order_id: str) -> LookupResult:
        matches = [row for row in self._rows if row["order_id"] == order_id]
        if not matches:
            return LookupResult(
                found=True,
                records=[],
                note="nothing has been reversed",
            )
        return LookupResult(found=True, records=matches)


class JsonRunbookStore:
    def __init__(self, path: Path | None = None) -> None:
        self._rows = _load(path or _DATA / "runbooks.json")

    def search_runbook(self, query: str) -> LookupResult:
        from triage_agent.tools.knowledge_base import rank_keywords

        matches = rank_keywords(
            query,
            self._rows,
            lambda row: str(row.get("title", "")),
            lambda row: str(row.get("body", "")),
        )
        if not matches:
            return LookupResult(found=False, records=[], note="no matching runbook")
        return LookupResult(found=True, records=matches)
