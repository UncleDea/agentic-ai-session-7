"""Triage Agent tests. The model is always a stub, so the suite needs no API key."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from triage_agent import TicketInput, TicketState, run_triage
from triage_agent.schemas import TriageClassification
from triage_agent.tests.fixtures import FIXTURES, RECEIVED_AT, TicketFixture

UNAMBIGUOUS = tuple(
    fixture
    for fixture in FIXTURES
    if fixture.call_model and fixture.stub_confidence >= 0.6
)


class _StructuredStub:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeModel:
    """Stands in for a LangChain chat model with structured output."""

    def __init__(self, result):
        self.structured = _StructuredStub(result)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


class ExplodingModel:
    def with_structured_output(self, schema):
        raise AssertionError("model should not be called")


def _ticket(fixture: TicketFixture) -> TicketInput:
    return TicketInput(
        ticket_id=fixture.ticket_id,
        raw_text=fixture.raw_text,
        customer_id=fixture.customer_id,
        channel=fixture.channel,
        received_at=RECEIVED_AT,
    )


def _model_for(fixture: TicketFixture):
    if not fixture.call_model:
        return ExplodingModel()
    classification = TriageClassification(
        category=fixture.stub_category,
        severity=fixture.stub_severity,
        confidence=fixture.stub_confidence,
        reasoning=fixture.stub_reasoning,
    )
    return FakeModel(classification)


def _assert_state(state: TicketState, fixture: TicketFixture) -> None:
    TicketState.model_validate(state.model_dump())
    assert state.ticket_id == fixture.ticket_id
    assert state.raw_text == fixture.raw_text
    assert state.customer_id == fixture.customer_id
    assert state.channel == fixture.channel
    assert state.received_at == RECEIVED_AT
    assert state.category == fixture.expect_category
    assert state.severity == fixture.expect_severity
    assert state.confidence == fixture.expect_confidence
    assert state.status == "triaged"
    assert state.next_node_hint == fixture.expect_hint
    assert state.triage_reasoning


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_every_fixture_returns_valid_ticket_state(fixture: TicketFixture):
    state = run_triage(_ticket(fixture), model=_model_for(fixture))
    _assert_state(state, fixture)


@pytest.mark.parametrize("fixture", UNAMBIGUOUS, ids=lambda fixture: fixture.name)
def test_unambiguous_fixtures_keep_stub_labels(fixture: TicketFixture):
    state = run_triage(_ticket(fixture), model=_model_for(fixture))
    assert state.category == fixture.stub_category
    assert state.severity == fixture.stub_severity
    assert state.triage_reasoning == fixture.stub_reasoning


def test_double_charge_is_billing_high():
    fixture = next(item for item in FIXTURES if item.name == "billing_double_charge")
    state = run_triage(_ticket(fixture), model=_model_for(fixture))
    assert "charged twice" in state.raw_text
    assert state.category == "billing"
    assert state.severity == "high"


def test_low_confidence_fallback_flags_human_review():
    fixture = next(item for item in FIXTURES if item.name == "ambiguous_help")
    state = run_triage(_ticket(fixture), model=_model_for(fixture))
    assert state.category == "other"
    assert state.severity == "medium"
    assert state.confidence == 0.3
    assert state.next_node_hint == "hitl"
    reasoning = state.triage_reasoning.lower()
    assert "low-confidence" in reasoning
    assert "0.60" in state.triage_reasoning
    assert "human should review" in reasoning


def test_schema_failure_uses_conservative_fallback():
    model = FakeModel(ValueError("structured output rejected"))
    ticket = TicketInput(
        ticket_id="TKT-BAD-SCHEMA",
        raw_text="I was charged twice for Pro this month.",
        customer_id="cust-x",
        channel="email",
        received_at=RECEIVED_AT,
    )
    state = run_triage(ticket, model=model)
    assert state.category == "other"
    assert state.severity == "medium"
    assert state.confidence == 0.0
    assert state.status == "triaged"
    assert state.next_node_hint == "hitl"
    assert "schema validation" in state.triage_reasoning
    assert "human should review" in state.triage_reasoning.lower()
    assert "Error: ValueError" in state.triage_reasoning
    assert model.structured.calls == 1


def test_invalid_structured_dict_falls_back():
    model = FakeModel({"category": "not-a-real-category", "confidence": 0.99})
    ticket = TicketInput(
        ticket_id="TKT-BAD-DICT",
        raw_text="Please change the email on my profile.",
        customer_id="cust-y",
        channel="form",
        received_at=RECEIVED_AT,
    )
    state = run_triage(ticket, model=model)
    assert state.category == "other"
    assert state.severity == "medium"
    assert state.confidence == 0.0
    assert "human should review" in state.triage_reasoning.lower()


def test_empty_text_does_not_call_the_model():
    fixture = next(item for item in FIXTURES if item.name == "empty_text")
    model = _model_for(fixture)
    state = run_triage(_ticket(fixture), model=model)
    assert state.category == "other"
    assert state.severity == "medium"
    assert state.confidence == 0.0
    assert "empty" in state.triage_reasoning
    assert "human should review" in state.triage_reasoning.lower()


def test_logs_classification_as_json(capsys):
    fixture = next(item for item in FIXTURES if item.name == "billing_double_charge")
    run_triage(_ticket(fixture), model=_model_for(fixture))
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload == {
        "ticket_id": "TKT-BILL-HIGH",
        "category": "billing",
        "severity": "high",
        "confidence": 0.95,
    }


def test_dict_input_is_accepted():
    fixture = next(item for item in FIXTURES if item.name == "chat_dashboard")
    payload = {
        "ticket_id": fixture.ticket_id,
        "raw_text": fixture.raw_text,
        "customer_id": fixture.customer_id,
        "channel": fixture.channel,
        "received_at": RECEIVED_AT,
    }
    state = run_triage(payload, model=_model_for(fixture))
    assert isinstance(state, TicketState)
    assert state.category == "technical"
    assert state.channel == "chat"


def test_invalid_input_raises_and_does_not_guess():
    with pytest.raises(ValidationError):
        run_triage(
            {
                "raw_text": "I was charged twice",
                "channel": "email",
                "received_at": datetime.now(timezone.utc),
            },
            model=ExplodingModel(),
        )


def test_public_exports_are_the_supervisor_contract():
    import triage_agent

    assert set(triage_agent.__all__) == {"TicketInput", "TicketState", "run_triage"}
