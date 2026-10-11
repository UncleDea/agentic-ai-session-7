"""Triage node. The model is stubbed unless the test is marked live."""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path

import pytest

from triage_agent.config import get_settings
from triage_agent.nodes.triage import build_messages, example_label, triage
from triage_agent.state import (
    Category,
    Priority,
    Sentiment,
    Ticket,
    TicketState,
    TriageResult,
    initial_state,
)

_SAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_tickets.json"
_LABEL_FIELDS = ("category", "priority", "sentiment", "language", "needs_human")
_PROMPT_SENTENCE = "Do not draft a customer reply."


class FakeModel:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.schemas: list = []
        self.messages = None
        self.invoke_count = 0

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self

    def invoke(self, messages):
        self.invoke_count += 1
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    records = json.loads(_SAMPLES_PATH.read_text(encoding="utf-8"))
    assert len(records) == 15
    return records


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _by_id(samples: list[dict], ticket_id: str) -> dict:
    return next(record for record in samples if record["id"] == ticket_id)


def _state_for(record: dict) -> TicketState:
    payload = {key: value for key, value in record.items() if key != "expected"}
    return initial_state(Ticket.model_validate(payload))


def _result(**overrides) -> TriageResult:
    data = {
        "category": Category.billing,
        "priority": Priority.P2,
        "sentiment": Sentiment.neutral,
        "language": "en",
        "confidence": 0.9,
        "needs_human": False,
    }
    data.update(overrides)
    return TriageResult(**data)


def _system(fake: FakeModel) -> str:
    assert fake.messages is not None
    return fake.messages[0].content


def _human(fake: FakeModel) -> str:
    assert fake.messages is not None
    return fake.messages[1].content


def test_labeled_samples_have_explicit_fields(samples: list[dict]) -> None:
    labeled = [record for record in samples if "category" in record["expected"]]
    assert len(labeled) == 13
    for record in labeled:
        missing = [field for field in _LABEL_FIELDS if field not in record["expected"]]
        assert missing == [], record["id"]
    assert _by_id(samples, "T-009")["expected"] == {"edge": "empty"}
    assert _by_id(samples, "T-010")["expected"] == {"edge": "very_short"}


def test_example_label_uses_file_values_and_rejects_gaps(samples: list[dict]) -> None:
    label = example_label(_by_id(samples, "T-014"))
    assert label["priority"] == "P1"
    assert label["sentiment"] == "negative"
    assert label["language"] == "en"
    assert label["needs_human"] is False
    incomplete = {
        "id": "T-014",
        "subject": "s",
        "body": "b",
        "expected": {"category": "billing"},
    }
    with pytest.raises(KeyError, match="sentiment"):
        example_label(incomplete)


def test_returns_only_triage(samples: list[dict]) -> None:
    fake = FakeModel(_result())
    update = triage(_state_for(_by_id(samples, "T-001")), model=fake)
    assert set(update) == {"triage"}
    assert update["triage"].category is Category.billing
    assert fake.invoke_count == 1
    assert fake.schemas == [TriageResult]


def test_does_not_construct_a_client(
    samples: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("get_chat_model was called")

    module = importlib.import_module("triage_agent.nodes.triage")
    monkeypatch.setattr(module, "get_chat_model", fail)
    fake = FakeModel(_result(category=Category.technical))
    update = triage(_state_for(_by_id(samples, "T-002")), model=fake)
    assert update["triage"].category is Category.technical


@pytest.mark.parametrize(
    ("ticket_id", "category"),
    [
        ("T-001", Category.billing),
        ("T-002", Category.technical),
        ("T-003", Category.account),
        ("T-004", Category.shipping),
        ("T-005", Category.refund),
        ("T-006", Category.feedback),
        ("T-007", Category.spam),
        ("T-008", Category.other),
        ("T-013", Category.security),
        ("T-015", Category.legal),
    ],
)
def test_stores_scripted_category(
    samples: list[dict], ticket_id: str, category: Category
) -> None:
    if category in {Category.security, Category.legal}:
        scripted = _result(
            category=category,
            priority=Priority.P1,
            needs_human=True,
            needs_human_reason=category.value,
        )
    else:
        scripted = _result(category=category)
    fake = FakeModel(scripted)
    update = triage(_state_for(_by_id(samples, ticket_id)), model=fake)
    assert update["triage"].category is category


def test_spam_ticket_is_stored(samples: list[dict]) -> None:
    fake = FakeModel(_result(category=Category.spam, priority=Priority.P4))
    update = triage(_state_for(_by_id(samples, "T-007")), model=fake)
    assert update["triage"].category is Category.spam


def test_non_english_body_is_sent_and_language_is_stored(samples: list[dict]) -> None:
    record = _by_id(samples, "T-011")
    fake = FakeModel(_result(category=Category.account, language="es"))
    update = triage(_state_for(record), model=fake)
    assert record["body"] in _human(fake)
    assert update["triage"].language == "es"


def test_multi_issue_keeps_primary_and_secondary(samples: list[dict]) -> None:
    fake = FakeModel(
        _result(
            category=Category.refund,
            priority=Priority.P2,
            secondary_categories=[Category.shipping],
        )
    )
    update = triage(_state_for(_by_id(samples, "T-012")), model=fake)
    result = update["triage"]
    assert result.category is Category.refund
    assert result.secondary_categories == [Category.shipping]


@pytest.mark.parametrize("ticket_id", ["T-009", "T-010"])
def test_empty_and_short_tickets_still_call_the_model(
    samples: list[dict], ticket_id: str
) -> None:
    record = _by_id(samples, ticket_id)
    fake = FakeModel(
        _result(
            category=Category.other,
            priority=Priority.P4,
            confidence=0.2,
            needs_human=True,
            needs_human_reason="empty or unintelligible",
        )
    )
    update = triage(_state_for(record), model=fake)
    assert fake.invoke_count == 1
    assert update["triage"].category is Category.other
    assert update["triage"].needs_human is True
    assert f"body: {record['body']}" in _human(fake)


def test_few_shot_excludes_the_current_ticket(samples: list[dict]) -> None:
    record = _by_id(samples, "T-001")
    other = _by_id(samples, "T-002")
    fake = FakeModel(_result())
    triage(_state_for(record), model=fake)
    system = _system(fake)
    assert _PROMPT_SENTENCE in system
    assert "Never set a priority below the category default." in system
    assert "money charged repeatedly" in system
    assert f"Example {other['id']}" in system
    assert other["expected"]["category"] in system
    assert record["body"] not in system
    assert record["subject"] not in _system(fake)
    assert record["body"] in _human(fake)


def test_prompt_allows_raising_priority_one_level() -> None:
    prompt = Path(__file__).resolve().parents[1].joinpath(
        "src", "triage_agent", "prompts", "triage.md"
    ).read_text(encoding="utf-8")
    assert "Raise that default by one level, and only one level" in prompt
    for signal in (
        "an outage affecting many users",
        "data loss",
        "money charged repeatedly",
        "a stated deadline",
    ):
        assert signal in prompt


def test_validation_failure_is_an_error(samples: list[dict]) -> None:
    fake = FakeModel(
        {
            "category": "billing",
            "priority": "P2",
            "sentiment": "neutral",
            "language": "en",
            "confidence": 0.5,
            "needs_human": True,
            "needs_human_reason": "  ",
        }
    )
    update = triage(_state_for(_by_id(samples, "T-001")), model=fake)
    assert set(update) == {"errors"}
    assert update["errors"][0].startswith("triage: ValidationError:")
    assert "triage" not in update


def test_model_exception_is_an_error(samples: list[dict]) -> None:
    fake = FakeModel(error=RuntimeError("provider down"))
    update = triage(_state_for(_by_id(samples, "T-001")), model=fake)
    assert update == {"errors": ["triage: RuntimeError: provider down"]}


def test_missing_ticket_is_an_error() -> None:
    fake = FakeModel(_result())
    update = triage({}, model=fake)
    assert update == {"errors": ["triage: state has no ticket"]}
    assert fake.invoke_count == 0


def test_logs_category_and_ticket_id(samples: list[dict], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="triage_agent.triage")
    fake = FakeModel(_result())
    triage(_state_for(_by_id(samples, "T-001")), model=fake)
    assert "node=triage" in caplog.text
    assert "ticket_id=T-001" in caplog.text
    assert "decision=billing" in caplog.text
    assert "latency_ms=" in caplog.text


def _ticket_from(record: dict) -> Ticket:
    payload = {key: value for key, value in record.items() if key != "expected"}
    return Ticket.model_validate(payload)


def _report_line(ticket_id: str, expected: dict, predicted: dict | None, notes: list[str]) -> str:
    status = "ok" if not notes else "miss"
    predicted_text = predicted if predicted is not None else "error"
    return f"{ticket_id} {status} expected={expected} predicted={predicted_text} notes={notes}"


@pytest.mark.live
def test_live_sample_accuracy(samples: list[dict]) -> None:
    settings = get_settings()
    if not os.getenv("AWS_ACCESS_KEY_ID", "").strip() or not os.getenv(
        "AWS_SECRET_ACCESS_KEY", ""
    ).strip():
        pytest.skip("AWS credentials are not set")

    lines: list[str] = []
    category_hits = 0
    category_total = 0
    field_failures: list[str] = []
    priority_hits = 0
    priority_total = 0

    for record in samples:
        expected = record["expected"]
        update = triage(initial_state(_ticket_from(record)))
        result = update.get("triage")
        notes: list[str] = []
        predicted = None
        if not isinstance(result, TriageResult):
            notes.append(f"error:{update.get('errors')}")
            if "category" in expected:
                category_total += 1
                field_failures.append(f"{record['id']} produced no triage")
            lines.append(_report_line(record["id"], expected, None, notes))
            continue

        predicted = {
            "category": result.category.value,
            "priority": result.priority.value,
            "sentiment": result.sentiment.value,
            "language": result.language.lower(),
            "needs_human": result.needs_human,
            "secondary_categories": [item.value for item in result.secondary_categories],
            "confidence": result.confidence,
        }

        if "category" in expected:
            category_total += 1
            priority_total += 1
            if predicted["category"] == expected["category"]:
                category_hits += 1
            else:
                notes.append("category")
            if predicted["priority"] == expected["priority"]:
                priority_hits += 1
            else:
                notes.append("priority")
            for field in ("sentiment", "language", "needs_human"):
                actual = predicted[field]
                if field == "language":
                    actual = actual.lower()
                if actual != expected[field]:
                    notes.append(field)
                    field_failures.append(
                        f"{record['id']} {field}: expected {expected[field]!r} got {actual!r}"
                    )
            expected_secondary = set(expected.get("secondary_categories", []))
            missing_secondary = expected_secondary - set(predicted["secondary_categories"])
            if missing_secondary:
                notes.append("secondary_categories")
                field_failures.append(
                    f"{record['id']} secondary_categories missing {sorted(missing_secondary)}"
                )
        else:
            if result.category is not Category.other:
                notes.append("category")
            if result.needs_human is not True:
                notes.append("needs_human")
            if result.confidence >= settings.confidence_threshold:
                notes.append("confidence")
            if notes:
                field_failures.append(f"{record['id']} edge contract failed: {notes}")

        lines.append(_report_line(record["id"], expected, predicted, notes))

    report = "\n".join(
        [
            *lines,
            f"category accuracy: {category_hits}/{category_total}",
            f"priority accuracy (reported, not gated): {priority_hits}/{priority_total}",
            f"field failures: {field_failures or 'none'}",
        ]
    )
    print(report)
    assert category_hits >= 11, report
    assert not field_failures, report


def test_build_messages_matches_node_path(samples: list[dict]) -> None:
    ticket = _ticket_from(_by_id(samples, "T-006"))
    messages = build_messages(ticket, samples)
    assert _PROMPT_SENTENCE in messages[0].content
    assert "Example T-006" not in messages[0].content
    assert ticket.body in messages[1].content
