"""Responder drafts. The model is stubbed unless the test is marked live."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from triage_agent.config import Settings, get_settings
from triage_agent.graph import build_graph, thread_config
from triage_agent.memory.episodic import seed_episodes
from triage_agent.memory.procedural import seed_procedures
from triage_agent.memory.semantic import seed_articles
from triage_agent.memory.store import FAKE_DIMS, FAKE_MODEL, fake_embeddings, open_store
from triage_agent.nodes.responder import fact_check, responder
from triage_agent.nodes.supervisor import supervisor
from triage_agent.state import (
    Category,
    Channel,
    Draft,
    Finding,
    Investigation,
    Priority,
    Review,
    ReviewStatus,
    Sentiment,
    Ticket,
    TriageResult,
    initial_state,
)

_DATA = Path(__file__).resolve().parents[1] / "data"
_SAMPLES = _DATA / "sample_tickets.json"
_LIVE_IDS = ("T-001", "T-005", "T-011", "T-014")
_LIVE_KB = {
    "T-001": ["KB-001"],
    "T-005": ["KB-003"],
    "T-011": ["KB-004"],
    "T-014": [],
}
_BLAND = "Thanks for writing. We are looking into this."


class FakeModel:
    def __init__(self, draft: Draft, error: Exception | None = None) -> None:
        self.draft = draft
        self.error = error
        self.messages = None
        self.schema = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.draft


def _system(fake: FakeModel) -> str:
    return fake.messages[0].content


def _human(fake: FakeModel) -> str:
    return fake.messages[1].content


@pytest.fixture
def store():
    opened = open_store(
        ":memory:",
        fake_embeddings(),
        dims=FAKE_DIMS,
        model=FAKE_MODEL,
    )
    try:
        yield opened
    finally:
        opened.conn.close()


def _ticket(**overrides) -> Ticket:
    data = {
        "id": "T-001",
        "customer_id": "C-1001",
        "channel": Channel.email,
        "subject": "Charged twice for order 88421",
        "body": (
            "I was billed twice on March 2 for order 88421, "
            "once for $49.00 and again for $49.00."
        ),
        "created_at": datetime(2026, 3, 3, tzinfo=timezone.utc),
    }
    data.update(overrides)
    return Ticket(**data)


def _triage(**overrides) -> TriageResult:
    data = {
        "category": Category.billing,
        "priority": Priority.P3,
        "sentiment": Sentiment.neutral,
        "language": "en",
        "confidence": 0.9,
        "needs_human": False,
    }
    data.update(overrides)
    return TriageResult(**data)


def _finding() -> Finding:
    return Finding(
        fact="Order 88421 was charged $49.00 on 2026-03-02.",
        source="order:88421",
    )


def _state(**overrides):
    state = initial_state(_ticket())
    state["triage"] = _triage()
    state["investigation"] = Investigation(
        findings=[_finding()],
        kb_article_ids=["KB-001"],
    )
    state.update(overrides)
    return state


def _draft(reply: str, confidence: float = 0.95) -> Draft:
    return Draft(reply=reply, confidence=confidence)


def _seed(store) -> None:
    seed_articles(store)
    seed_episodes(store)
    seed_procedures(store)


def test_procedure_text_is_in_the_system_message(store) -> None:
    _seed(store)
    fake = FakeModel(_draft(_BLAND))
    responder(_state(), model=fake, settings=Settings(), store=store)
    system = _system(fake)
    assert "Support Team" in system
    assert "Never promise a delivery date" in system
    assert "Do not escalate" in system
    assert "The ticket language is en." in system


def test_missing_procedure_still_drafts(store) -> None:
    fake = FakeModel(_draft(_BLAND))
    update = responder(_state(), model=fake, settings=Settings(), store=store)
    assert update["draft"].reply == _BLAND
    assert update["draft"].memory_ids == []
    assert "Support Team" not in _system(fake)


def test_memory_ids_keep_applicable_articles_and_episodes(store) -> None:
    _seed(store)
    fake = FakeModel(_draft(_BLAND))
    state = _state()
    state["investigation"] = Investigation(
        findings=[_finding()],
        kb_article_ids=["KB-001", "KB-007"],
    )
    update = responder(state, model=fake, settings=Settings(), store=store)
    assert update["draft"].memory_ids == ["v1", "EP-001", "KB-001"]
    assert "Gift wrap" not in _human(fake)
    assert "EP-001" in _human(fake)
    assert "style only, not facts" in _human(fake)


def test_empty_kb_ids_search_for_an_applicable_article(store) -> None:
    _seed(store)
    fake = FakeModel(_draft(_BLAND))
    state = _state()
    state["investigation"] = Investigation(findings=[_finding()], kb_article_ids=[])
    update = responder(state, model=fake, settings=Settings(), store=store)
    assert "KB-001" in update["draft"].memory_ids
    assert "KB-007" not in update["draft"].memory_ids


def test_episode_reply_is_not_a_fact_source(store) -> None:
    _seed(store)
    reply = "I confirmed both charges on the order and reversed the extra one."
    fake = FakeModel(_draft(reply))
    state = _state()
    state["investigation"] = Investigation(
        findings=[_finding()],
        kb_article_ids=["KB-007"],
    )
    update = responder(state, model=fake, settings=Settings(), store=store)
    assert "reversed the extra one" in _human(fake)
    assert "KB-007" not in update["draft"].memory_ids
    assert any("revers" in claim.lower() for claim in update["draft"].flagged_claims)
    assert update["draft"].confidence < 0.7


def test_t001_reworded_date_is_not_flagged(store) -> None:
    _seed(store)
    corpus = "Order 88421 was charged $49.00 on 2026-03-02."
    assert fact_check("The charge was on March 2.", corpus) == []
    assert fact_check("The charge was on 2 de marzo.", corpus) == []
    assert fact_check("The charge was on 2026-03-02.", corpus) == []
    assert fact_check("The charge was on March 2, 2025.", corpus) == ["March 2, 2025"]
    reply = (
        "I've passed this to our billing team to review the extra charge "
        "on order 88421 of $49.00 from March 2."
    )
    fake = FakeModel(_draft(reply))
    update = responder(_state(), model=fake, settings=Settings(), store=store)
    draft = update["draft"]
    assert draft.flagged_claims == []
    assert draft.confidence == 0.95
    assert draft.escalates is False
    assert "KB-001" in draft.memory_ids
    assert draft.reply.startswith("I've passed this")


def test_completed_action_is_flagged_even_when_the_article_allows_it(store) -> None:
    _seed(store)
    reply = "I reversed the extra charge on order 88421 of $49.00 on March 2."
    fake = FakeModel(_draft(reply))
    update = responder(_state(), model=fake, settings=Settings(), store=store)
    draft = update["draft"]
    assert "KB-001" in draft.memory_ids
    assert any("reversed" in claim.lower() for claim in draft.flagged_claims)
    assert not any("march" in claim.lower() for claim in draft.flagged_claims)
    assert draft.confidence < 0.7
    allowed = "A damaged item can be refunded to the original payment method."
    assert any(
        "refunded" in claim.lower()
        for claim in fact_check("The headphones have been refunded.", allowed)
    )
    assert fact_check("The headphones can be refunded to the original payment method.", allowed) == []
    assert fact_check("Ya reembolsé el pedido.", "se puede reembolsar") == ["reembolsé"]
    assert any(
        "cancell" in claim.lower()
        for claim in fact_check("I cancelled the plan.", "status canceled")
    )
    assert fact_check("Your monthly plan is marked as canceled.", "status canceled") == []


def test_invented_amount_order_and_date_are_flagged(store) -> None:
    _seed(store)
    reply = "I reversed order 99999 for $99.00 on 2026-04-01."
    fake = FakeModel(_draft(reply))
    update = responder(_state(), model=fake, settings=Settings(), store=store)
    draft = update["draft"]
    blob = " ".join(draft.flagged_claims).lower()
    assert "99999" in blob
    assert "99.00" in blob
    assert "2026-04-01" in blob
    assert draft.confidence < 0.7
    assert draft.reply == reply


def test_delivery_date_promise_is_flagged_when_the_article_only_forbids_it(store) -> None:
    _seed(store)
    fake = FakeModel(_draft("Your delivery date is Friday."))
    state = _state(
        ticket=_ticket(
            id="T-004",
            customer_id="C-1004",
            subject="Order 90210 has not moved",
            body="Tracking for order 90210 has said label created for six days.",
        ),
        triage=_triage(category=Category.shipping),
        investigation=Investigation(
            findings=[
                Finding(
                    fact="Order 90210 status is label created.",
                    source="order:90210",
                )
            ],
            kb_article_ids=["KB-002"],
        ),
    )
    update = responder(state, model=fake, settings=Settings(), store=store)
    assert "KB-002" in update["draft"].memory_ids
    assert any("delivery date" in claim.lower() for claim in update["draft"].flagged_claims)
    assert update["draft"].confidence < 0.7


def test_refusal_to_promise_a_delivery_date_is_not_flagged(store) -> None:
    _seed(store)
    fake = FakeModel(_draft("I cannot promise a delivery date."))
    update = responder(_state(), model=fake, settings=Settings(), store=store)
    assert update["draft"].flagged_claims == []
    assert update["draft"].confidence == 0.95


def test_t005_refund_timeframe_is_flagged_even_with_kb_003(store) -> None:
    _seed(store)
    record = next(
        row
        for row in json.loads(_SAMPLES.read_text(encoding="utf-8"))
        if row["id"] == "T-005"
    )
    ticket = Ticket.model_validate({key: value for key, value in record.items() if key != "expected"})
    reply = "your refund will arrive within 5 business days"
    fake = FakeModel(_draft(reply))
    state = initial_state(ticket)
    state["triage"] = _triage(category=Category.refund, priority=Priority.P2)
    state["investigation"] = Investigation(
        findings=[
            Finding(
                fact="Order 7712 status delivered amount 80.00 headphones delivered.",
                source="order:7712",
            )
        ],
        kb_article_ids=["KB-003"],
    )
    update = responder(state, model=fake, settings=Settings(), store=store)
    draft = update["draft"]
    assert "KB-003" in draft.memory_ids
    assert any("business days" in claim.lower() for claim in draft.flagged_claims)
    assert not any(claim.lower() == "refund" for claim in draft.flagged_claims)
    assert draft.confidence < 0.7


@pytest.mark.parametrize(
    ("reply", "corpus", "snippet"),
    [
        ("Ship it within 5 business days.", "refunded", "within 5 business days"),
        ("Llega dentro de 2 días.", "reembolso", "dentro de 2 días"),
        ("I will send it by Friday.", "label created", "by Friday"),
        ("Son 3 horas de espera.", "cuenta activa", "3 horas"),
    ],
)
def test_unsupported_timeframes_are_flagged(reply: str, corpus: str, snippet: str) -> None:
    assert snippet in fact_check(reply, corpus)


def test_timeframe_in_the_corpus_is_kept() -> None:
    assert fact_check("It has been 6 days.", "label created for 6 days") == []


def test_spanish_ticket_sets_the_reply_language(store) -> None:
    fake = FakeModel(_draft("Gracias, estamos revisando la cuenta."))
    state = _state(triage=_triage(language="es", category=Category.account))
    responder(state, model=fake, settings=Settings(), store=store)
    assert "The ticket language is es." in _system(fake)
    assert "language: es" in _human(fake)


def test_revision_uses_feedback_and_the_previous_draft(store) -> None:
    _seed(store)
    previous = "We refunded invoice 42."
    fake = FakeModel(_draft("Here is a more specific update."))
    state = _state(
        draft=Draft(reply=previous, confidence=0.95),
        review=Review(status=ReviewStatus.rejected, reviewer_feedback="be specific"),
        revision_count=1,
    )
    update = responder(state, model=fake, settings=Settings(), store=store)
    assert previous in _human(fake)
    assert "be specific" in _human(fake)
    assert "review" in update and update["review"] is None
    assert update["revision_count"] == 2
    assert "revision_count" in update


def test_pending_review_is_not_a_revision(store) -> None:
    fake = FakeModel(_draft(_BLAND))
    state = _state(review=Review(status=ReviewStatus.pending), revision_count=0)
    update = responder(state, model=fake, settings=Settings(), store=store)
    assert "review" not in update
    assert "revision_count" not in update
    assert "Previous draft" not in _human(fake)


def test_missing_ticket_is_an_error() -> None:
    update = responder({}, model=FakeModel(_draft(_BLAND)), settings=Settings())
    assert update["errors"] == ["responder: state has no ticket"]


def test_model_error_is_recorded(store) -> None:
    fake = FakeModel(_draft(_BLAND), error=RuntimeError("provider down"))
    update = responder(_state(), model=fake, settings=Settings(), store=store)
    assert update["errors"] == ["responder: RuntimeError: provider down"]
    assert "draft" not in update


def test_logs_ticket_id_and_flagged_claim(store, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="triage_agent.responder")
    fake = FakeModel(_draft("The extra charge was $99.00."))
    responder(_state(), model=fake, settings=Settings(), store=store)
    assert "node=responder" in caplog.text
    assert "ticket_id=T-001" in caplog.text
    assert "decision=draft flagged=1" in caplog.text
    assert "flagged_claim=$99.00" in caplog.text


def test_t011_escalation_routes_to_hitl(store) -> None:
    _seed(store)
    record = next(row for row in _samples() if row["id"] == "T-011")
    ticket = Ticket.model_validate(
        {key: value for key, value in record.items() if key != "expected"}
    )
    reply = "Voy a escalar esto a un especialista, quien hará seguimiento."
    fake = FakeModel(_draft(reply, confidence=0.9))
    state = initial_state(ticket)
    state["triage"] = _triage(
        category=Category.account,
        priority=Priority.P3,
        language="es",
    )
    state["investigation"] = Investigation(
        findings=[
            Finding(
                fact="Jules Vega email jules.vega@example.com plan plus status active.",
                source="customer:C-1011",
            )
        ],
        kb_article_ids=["KB-004"],
    )
    update = responder(state, model=fake, settings=Settings(), store=store)
    draft = update["draft"]
    assert draft.escalates is True
    assert draft.flagged_claims == []
    assert draft.confidence == 0.9
    state["draft"] = draft
    decision = supervisor(state, settings=Settings())
    assert decision["next_node"] == "hitl"
    assert decision["route_history"][0].reason == "escalation promised"


def test_flagged_draft_routes_to_hitl(store) -> None:
    fake = FakeModel(_draft("The extra charge was $99.00."))
    update = responder(_state(), model=fake, settings=Settings(), store=store)
    state = _state()
    state["draft"] = update["draft"]
    decision = supervisor(state, settings=Settings())
    assert decision["next_node"] == "hitl"
    assert decision["route_history"][0].reason == "draft confidence below threshold"


def test_graph_passes_the_store(store, tmp_path) -> None:
    seed_procedures(store)

    class FeedbackModel:
        def __init__(self) -> None:
            self.schema = None

        def bind_tools(self, tools, **kwargs):
            return self

        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        def invoke(self, messages):
            if self.schema is Draft:
                return Draft(reply=_BLAND, confidence=0.95)
            return _triage(
                category=Category.feedback,
                priority=Priority.P4,
                sentiment=Sentiment.positive,
            )

    graph = build_graph(
        model=FeedbackModel(),
        settings=Settings(
            outbox_dir=str(tmp_path / "outbox"),
            escalations_dir=str(tmp_path / "escalations"),
        ),
        store=store,
    )
    result = graph.invoke(initial_state(_ticket()), thread_config("T-001", "store"))
    assert [entry.node for entry in result["route_history"]] == [
        "triage",
        "responder",
        "send",
    ]
    assert result["draft"].memory_ids == ["v1"]


def _samples() -> list[dict]:
    return json.loads(_SAMPLES.read_text(encoding="utf-8"))


def _live_state(ticket_id: str):
    record = next(row for row in _samples() if row["id"] == ticket_id)
    ticket = Ticket.model_validate(
        {key: value for key, value in record.items() if key != "expected"}
    )
    expected = record["expected"]
    customers = json.loads((_DATA / "customers.json").read_text(encoding="utf-8"))
    orders = json.loads((_DATA / "orders.json").read_text(encoding="utf-8"))
    customer = next(row for row in customers if row["customer_id"] == ticket.customer_id)
    findings = [
        Finding(
            fact=(
                f"{customer['name']} email {customer['email']} "
                f"plan {customer['plan']} status {customer['status']}."
            ),
            source=customer["source"],
        )
    ]
    for order in orders:
        if order["customer_id"] != ticket.customer_id:
            continue
        findings.append(
            Finding(
                fact=(
                    f"Order {order['order_id']} status {order['status']} "
                    f"amount {order['amount']} placed_on {order['placed_on']}. "
                    f"{order['detail']}"
                ),
                source=order["source"],
            )
        )
    state = initial_state(ticket)
    state["triage"] = TriageResult(
        category=expected["category"],
        priority=expected["priority"],
        sentiment=expected["sentiment"],
        language=expected["language"],
        confidence=0.9,
        needs_human=expected["needs_human"],
        needs_human_reason=expected.get("needs_human_reason", ""),
    )
    state["investigation"] = Investigation(
        findings=findings,
        kb_article_ids=list(_LIVE_KB[ticket_id]),
    )
    return state


def _looks_spanish(reply: str) -> bool:
    lowered = reply.lower()
    if any(word in lowered for word in ("cuenta", "contraseña", "contrasena", "sesión", "sesion")):
        return True
    return re.search(r"[áéíóúñ¿¡]", reply) is not None


@pytest.mark.live
def test_live_replies_for_four_tickets(store) -> None:
    settings = get_settings()
    if not os.getenv("AWS_ACCESS_KEY_ID", "").strip() or not os.getenv(
        "AWS_SECRET_ACCESS_KEY", ""
    ).strip():
        pytest.skip("AWS credentials are not set")
    _seed(store)
    lines: list[str] = []
    failures: list[str] = []
    for ticket_id in _LIVE_IDS:
        update = responder(_live_state(ticket_id), settings=settings, store=store)
        draft = update.get("draft")
        if not isinstance(draft, Draft):
            failures.append(f"{ticket_id} produced no draft: {update.get('errors')}")
            lines.append(f"{ticket_id} error={update.get('errors')}")
            continue
        lines.append(
            "\n".join(
                [
                    ticket_id,
                    f"reply: {draft.reply}",
                    f"memory_ids: {draft.memory_ids}",
                    f"confidence: {draft.confidence}",
                    f"escalates: {draft.escalates}",
                    f"flagged_claims: {draft.flagged_claims}",
                ]
            )
        )
        if ticket_id == "T-011":
            if not _looks_spanish(draft.reply):
                failures.append(f"T-011 reply was not Spanish: {draft.reply}")
            if draft.escalates:
                failures.append(f"T-011 escalated: {draft.reply}")
        elif draft.escalates is not True:
            failures.append(f"{ticket_id} did not escalate: {draft.reply}")
    report = "\n\n".join(lines)
    print(report)
    assert not failures, report
