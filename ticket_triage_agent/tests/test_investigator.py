"""Investigator and mock lookups. The model is stubbed unless marked live."""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from triage_agent.config import Settings, get_settings
from triage_agent.graph import build_graph
from triage_agent.nodes.investigator import investigator, load_prompt
from triage_agent.nodes.supervisor import supervisor
from triage_agent.state import (
    Category,
    Draft,
    Finding,
    Investigation,
    Priority,
    Sentiment,
    Ticket,
    TriageResult,
    initial_state,
)
from triage_agent.tools import build_tools
from triage_agent.tools.knowledge_base import JsonKnowledgeBase
from triage_agent.tools.ticket_db import (
    JsonCustomerStore,
    JsonOrderStore,
    JsonPastTicketStore,
    LookupResult,
)

_DATA = Path(__file__).resolve().parents[1] / "data"
_SAMPLES = _DATA / "sample_tickets.json"
_LIVE_IDS = ("T-001", "T-004", "T-005", "T-009")


class ScriptedModel:
    def __init__(
        self,
        turns: list[AIMessage] | None = None,
        investigation: Investigation | None = None,
        error: Exception | None = None,
    ) -> None:
        self.turns = list(turns or [])
        self.investigation = investigation or Investigation()
        self.error = error
        self.schema = None
        self.tool_names: list[str] = []

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [item.name for item in tools]
        self.schema = None
        return self

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    def invoke(self, messages):
        if self.error is not None:
            raise self.error
        if self.schema is Investigation:
            return self.investigation
        if not self.turns:
            return AIMessage(content="done")
        return self.turns.pop(0)


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _records() -> list[dict]:
    return json.loads(_SAMPLES.read_text(encoding="utf-8"))


def _by_id(ticket_id: str) -> dict:
    return next(record for record in _records() if record["id"] == ticket_id)


def _state_for(ticket_id: str):
    record = _by_id(ticket_id)
    payload = {key: value for key, value in record.items() if key != "expected"}
    return initial_state(Ticket.model_validate(payload))


def _safe_triage(**overrides) -> TriageResult:
    data = {
        "category": Category.technical,
        "priority": Priority.P3,
        "sentiment": Sentiment.neutral,
        "language": "en",
        "confidence": 0.9,
        "needs_human": False,
    }
    data.update(overrides)
    return TriageResult(**data)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_prompt_forbids_a_reply() -> None:
    text = load_prompt()
    assert "Do not draft a reply." in text
    assert "missing_info" in text


def test_mock_data_uses_sample_customer_ids_except_the_missing_one() -> None:
    customers = {
        row["customer_id"]
        for row in json.loads((_DATA / "customers.json").read_text(encoding="utf-8"))
    }
    sample_ids = {record["customer_id"] for record in _records()}
    assert "C-1009" not in customers
    assert sample_ids - customers == {"C-1009"}
    orders = json.loads((_DATA / "orders.json").read_text(encoding="utf-8"))
    order = next(row for row in orders if row["order_id"] == "88421")
    assert order["customer_id"] == "C-1001"
    assert order["amount"] == "49.00"
    assert order["source"] == "order:88421"


def test_get_customer_finds_a_sample_and_reports_the_missing_one() -> None:
    store = JsonCustomerStore()
    found = store.get_customer("C-1001")
    assert found.found is True
    assert found.records[0]["source"] == "customer:C-1001"
    assert found.records[0]["name"] == "Avery Chen"
    missing = store.get_customer("C-1009")
    assert missing.found is False
    assert missing.records == []
    assert missing.note == "customer C-1009 not found"
    assert "Avery" not in missing.note


def test_get_orders_returns_the_sample_order() -> None:
    result = JsonOrderStore().get_orders("C-1001")
    assert result.found is True
    assert [row["order_id"] for row in result.records] == ["88421"]


def test_search_past_tickets_stays_on_that_customer() -> None:
    store = JsonPastTicketStore()
    found = store.search_past_tickets("C-1001", "duplicate")
    assert found.found is True
    assert {row["customer_id"] for row in found.records} == {"C-1001"}
    assert found.records[0]["id"] == "PT-2001"
    empty = store.search_past_tickets("C-1002", "duplicate")
    assert empty.found is False
    assert empty.records == []
    body_only = store.search_past_tickets("C-1001", "january")
    assert body_only.records == []


def test_search_kb_matches_keywords_and_ignores_other_articles() -> None:
    kb = JsonKnowledgeBase()
    found = kb.search_kb("charged twice")
    ids = [row["id"] for row in found.records]
    assert "KB-001" in ids
    assert "KB-007" not in ids
    empty = kb.search_kb("zzzznomatch")
    assert empty.found is False
    assert empty.records == []


def test_t004_subject_returns_the_tracking_article_first() -> None:
    ids = [row["id"] for row in JsonKnowledgeBase().search_kb(_by_id("T-004")["subject"]).records]
    assert ids[0] == "KB-002"
    assert "KB-001" not in ids


def test_t005_subject_skips_charge_and_tracking_articles() -> None:
    ids = [row["id"] for row in JsonKnowledgeBase().search_kb(_by_id("T-005")["subject"]).records]
    assert "KB-001" not in ids
    assert "KB-002" not in ids


def test_tool_selection_cites_the_customer_record() -> None:
    model = ScriptedModel(
        turns=[_call("get_customer", {"customer_id": "C-1001"}, "c1")],
        investigation=Investigation(
            findings=[
                Finding(fact="Avery Chen is active.", source="customer:C-1001")
            ]
        ),
    )
    update = investigator(_state_for("T-001"), model=model, settings=Settings())
    investigation = update["investigation"]
    assert model.tool_names == [
        "get_customer",
        "get_orders",
        "search_past_tickets",
        "search_kb",
    ]
    assert [item.fact for item in investigation.findings] == ["Avery Chen is active."]
    assert investigation.findings[0].source == "customer:C-1001"
    assert investigation.missing_info == []
    assert investigation.dropped_claims == []


def test_missing_customer_is_missing_info_and_not_invented() -> None:
    model = ScriptedModel(
        turns=[_call("get_customer", {"customer_id": "C-1009"}, "c1")],
        investigation=Investigation(
            findings=[
                Finding(fact="The customer is Named Person.", source="customer:C-1009")
            ]
        ),
    )
    update = investigator(_state_for("T-009"), model=model, settings=Settings())
    investigation = update["investigation"]
    assert investigation.missing_info == ["customer C-1009 not found"]
    assert investigation.findings == []
    assert investigation.dropped_claims == ["The customer is Named Person."]


def test_named_order_not_in_the_lookup_is_missing_info() -> None:
    class EmptyOrders:
        def get_orders(self, customer_id: str) -> LookupResult:
            return LookupResult(found=True, records=[])

    model = ScriptedModel(
        turns=[
            _call("get_customer", {"customer_id": "C-1001"}, "c1"),
            _call("get_orders", {"customer_id": "C-1001"}, "c2"),
            _call(
                "search_past_tickets",
                {"customer_id": "C-1001", "query": "zzzznomatch"},
                "c3",
            ),
            _call("search_kb", {"query": "zzzznomatch"}, "c4"),
        ],
        investigation=Investigation(
            findings=[Finding(fact="Avery Chen is active.", source="customer:C-1001")],
            missing_info=["no matching knowledge base articles"],
        ),
    )
    tools = build_tools(orders=EmptyOrders())
    update = investigator(
        _state_for("T-001"),
        model=model,
        settings=Settings(),
        tools=tools,
    )
    assert update["investigation"].missing_info == ["order 88421 not found"]
    assert update["investigation"].findings[0].source == "customer:C-1001"


def test_tool_call_limit_stops_a_model_that_keeps_calling() -> None:
    class CountingCustomers:
        def __init__(self) -> None:
            self.calls = 0

        def get_customer(self, customer_id: str) -> LookupResult:
            self.calls += 1
            return LookupResult(
                found=True,
                records=[
                    {
                        "customer_id": customer_id,
                        "source": f"customer:{customer_id}",
                        "status": "active",
                    }
                ],
            )

    class RepeatingModel(ScriptedModel):
        def __init__(self) -> None:
            super().__init__(
                investigation=Investigation(
                    findings=[
                        Finding(
                            fact="The account status is active.",
                            source="customer:C-1001",
                        )
                    ]
                )
            )
            self.rounds = 0

        def invoke(self, messages):
            if self.schema is Investigation:
                return self.investigation
            self.rounds += 1
            return _call("get_customer", {"customer_id": "C-1001"}, f"c{self.rounds}")

    customers = CountingCustomers()
    model = RepeatingModel()
    update = investigator(
        _state_for("T-001"),
        model=model,
        settings=Settings(max_tool_calls=2),
        tools=build_tools(customers=customers),
    )
    assert customers.calls == 2
    assert model.rounds == 2
    assert update["investigation"].findings[0].source == "customer:C-1001"


def test_wrong_amount_is_dropped_and_does_not_force_review(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="triage_agent.investigator")
    model = ScriptedModel(
        turns=[_call("get_orders", {"customer_id": "C-1001"}, "o1")],
        investigation=Investigation(
            findings=[
                Finding(
                    fact="Order 88421 was charged $49.00.",
                    source="order:88421",
                ),
                Finding(
                    fact="Order 88421 was charged $99.00.",
                    source="order:88421",
                ),
            ]
        ),
    )
    update = investigator(_state_for("T-001"), model=model, settings=Settings())
    investigation = update["investigation"]
    assert [item.fact for item in investigation.findings] == [
        "Order 88421 was charged $49.00."
    ]
    assert investigation.dropped_claims == ["Order 88421 was charged $99.00."]
    assert investigation.missing_info == []
    assert "dropped_claim=Order 88421 was charged $99.00." in caplog.text

    state = _state_for("T-001")
    state["triage"] = _safe_triage(category=Category.billing)
    state["investigation"] = investigation
    state["draft"] = Draft(reply="We see the charge.", memory_ids=[], confidence=0.95)
    decision = supervisor(state, settings=Settings())
    assert decision["next_node"] == "send"
    assert decision["route_history"][0].reason == "hitl criteria not met"


def test_empty_searches_leave_missing_info_empty_and_route_to_send() -> None:
    class RouteModel:
        def __init__(self) -> None:
            self.schema = None
            self.step = 0

        def bind_tools(self, tools, **kwargs):
            self.schema = None
            return self

        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        def invoke(self, messages):
            if self.schema is Investigation:
                return Investigation(
                    findings=[
                        Finding(
                            fact="The account status is active.",
                            source="customer:C-1002",
                        )
                    ],
                    missing_info=["no matching knowledge base articles"],
                    related_ticket_ids=["PT-9999"],
                    kb_article_ids=["KB-999"],
                )
            if self.schema is None:
                self.step += 1
                calls = {
                    1: _call("get_customer", {"customer_id": "C-1002"}, "1"),
                    2: _call("get_orders", {"customer_id": "C-1002"}, "2"),
                    3: _call(
                        "search_past_tickets",
                        {"customer_id": "C-1002", "query": "zzzznomatch"},
                        "3",
                    ),
                    4: _call("search_kb", {"query": "zzzznomatch"}, "4"),
                }
                return calls.get(self.step, AIMessage(content="done"))
            return _safe_triage()

    graph = build_graph(model=RouteModel(), settings=Settings())
    result = graph.invoke(_state_for("T-002"))
    assert result["investigation"].missing_info == []
    assert result["investigation"].related_ticket_ids == []
    assert result["investigation"].kb_article_ids == []
    assert [entry.node for entry in result["route_history"]] == [
        "triage",
        "investigator",
        "responder",
        "send",
    ]
    assert result["route_history"][-1].reason == "hitl criteria not met"
    assert result["sent"] is True


def test_unsourced_claim_is_dropped_not_missing() -> None:
    model = ScriptedModel(
        turns=[_call("get_customer", {"customer_id": "C-1001"}, "c1")],
        investigation=Investigation(
            findings=[Finding(fact="A secret refund of $10.00.", source="imagination")]
        ),
    )
    update = investigator(_state_for("T-001"), model=model, settings=Settings())
    investigation = update["investigation"]
    assert investigation.findings == []
    assert investigation.dropped_claims == ["A secret refund of $10.00."]
    assert investigation.missing_info == []


def test_empty_source_is_an_error() -> None:
    class BadSchema(ScriptedModel):
        def invoke(self, messages):
            if self.schema is Investigation:
                return {
                    "findings": [{"fact": "Avery Chen is active.", "source": ""}],
                    "missing_info": [],
                    "related_ticket_ids": [],
                    "kb_article_ids": [],
                    "dropped_claims": [],
                }
            return super().invoke(messages)

    update = investigator(
        _state_for("T-001"),
        model=BadSchema(turns=[_call("get_customer", {"customer_id": "C-1001"}, "c1")]),
        settings=Settings(),
    )
    assert "investigation" not in update
    assert update["errors"][0].startswith("investigator: ")


def test_model_exception_is_an_error() -> None:
    update = investigator(
        _state_for("T-001"),
        model=ScriptedModel(error=RuntimeError("boom")),
        settings=Settings(),
    )
    assert "investigation" not in update
    assert update["errors"] == ["investigator: RuntimeError: boom"]


def test_missing_ticket_is_an_error() -> None:
    update = investigator({}, model=ScriptedModel(), settings=Settings())
    assert update == {"errors": ["investigator: state has no ticket"]}


def test_passed_model_does_not_construct_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("get_chat_model was called")

    module = importlib.import_module("triage_agent.nodes.investigator")
    monkeypatch.setattr(module, "get_chat_model", fail)
    model = ScriptedModel(
        turns=[_call("get_customer", {"customer_id": "C-1001"}, "c1")],
        investigation=Investigation(
            findings=[Finding(fact="Avery Chen is active.", source="customer:C-1001")]
        ),
    )
    update = investigator(_state_for("T-001"), model=model, settings=Settings())
    assert update["investigation"].findings[0].source == "customer:C-1001"


def test_logs_ticket_id(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="triage_agent.investigator")
    model = ScriptedModel(
        turns=[_call("get_customer", {"customer_id": "C-1001"}, "c1")],
        investigation=Investigation(
            findings=[Finding(fact="Avery Chen is active.", source="customer:C-1001")]
        ),
    )
    investigator(_state_for("T-001"), model=model, settings=Settings())
    assert "node=investigator" in caplog.text
    assert "ticket_id=T-001" in caplog.text
    assert "findings=1" in caplog.text


@pytest.mark.live
def test_live_investigation_prints_findings() -> None:
    settings = get_settings()
    key_name = "OPENAI_API_KEY" if settings.model_provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(key_name):
        pytest.skip(f"{key_name} is not set")

    lines: list[str] = []
    failures: list[str] = []
    for ticket_id in _LIVE_IDS:
        update = investigator(_state_for(ticket_id), settings=settings)
        investigation = update.get("investigation")
        if not isinstance(investigation, Investigation):
            failures.append(f"{ticket_id} produced no investigation: {update.get('errors')}")
            lines.append(f"{ticket_id} error={update.get('errors')}")
            continue
        for finding in investigation.findings:
            if not finding.source.strip():
                failures.append(f"{ticket_id} finding missing source: {finding.fact}")
        rendered = [
            f"  - {finding.fact} [{finding.source}]" for finding in investigation.findings
        ]
        lines.append(
            "\n".join(
                [
                    ticket_id,
                    "findings:",
                    *(rendered or ["  (none)"]),
                    f"related={investigation.related_ticket_ids}",
                    f"kb={investigation.kb_article_ids}",
                    f"missing_info={investigation.missing_info}",
                    f"dropped_claims={investigation.dropped_claims}",
                ]
            )
        )
        blob = " ".join(
            [finding.fact + " " + finding.source for finding in investigation.findings]
        )
        if ticket_id == "T-001" and "88421" not in blob and "C-1001" not in blob:
            failures.append(f"T-001 did not cite 88421 or C-1001: {blob}")
        if ticket_id == "T-009" and not any(
            "C-1009" in item for item in investigation.missing_info
        ):
            failures.append(
                f"T-009 missing_info did not mention C-1009: {investigation.missing_info}"
            )

    report = "\n\n".join(lines)
    print(report)
    assert not failures, report
