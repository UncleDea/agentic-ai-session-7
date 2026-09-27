"""Responder tests. The draft model is a stub, so the suite needs no API key."""

import json
from datetime import datetime, timezone

from investigator.schemas import Evidence, Investigation
from triage_agent.schemas import TicketState
from responder.agent import run_response
from responder.schemas import ReplyText, ResponseDraft

RECEIVED_AT = datetime(2026, 5, 10, 9, 14, 22, tzinfo=timezone.utc)


class _Structured:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeModel:
    def __init__(self, result):
        self.structured = _Structured(result)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


def _ticket(raw_text: str, customer_id: str | None = "priya.s@example.com") -> TicketState:
    return TicketState(
        ticket_id="TKT-NEW",
        raw_text=raw_text,
        customer_id=customer_id,
        channel="email",
        received_at=RECEIVED_AT,
        category="account",
        severity="medium",
        confidence=0.9,
        triage_reasoning="test",
        status="triaged",
        next_node_hint="investigator",
    )


def _investigation(**overrides) -> Investigation:
    payload = {
        "status": "investigated",
        "summary": "Matched auth-service.",
        "service": "auth-service",
        "runbook_path": "support/runbooks/mfa-loop.md",
        "related_ticket_ids": ["TKT-1001"],
        "evidence": [
            Evidence(
                source="runbook",
                ref="support/runbooks/mfa-loop.md",
                excerpt="MFA loop on login",
            )
        ],
    }
    payload.update(overrides)
    return Investigation(**payload)


def _prompt_text(model: FakeModel) -> str:
    return "\n".join(message.content for message in model.structured.messages)


def test_mfa_draft_uses_support_memory_and_copies_the_model():
    model = FakeModel(
        ReplyText(subject="Re: MFA loop", body="Please reopen your authenticator app and try again.")
    )
    draft = run_response(
        _ticket("Authenticator codes loop back to the MFA page."),
        _investigation(),
        model=model,
    )
    prompt = _prompt_text(model)
    assert isinstance(draft, ResponseDraft)
    assert draft.status == "drafted"
    assert draft.domain == "support"
    assert draft.subject == "Re: MFA loop"
    assert draft.body == "Please reopen your authenticator app and try again."
    assert draft.memory_used == ["semantic", "episodic", "procedural"]
    assert draft.needs_review is False
    assert draft.next_node_hint == "send"
    assert "Never promise refunds" in prompt
    assert "close the authenticator app" in prompt
    assert "Monk Support" in prompt


def test_vpn_draft_uses_the_it_helpdesk_prompt():
    model = FakeModel(ReplyText(subject="Re: VPN", body="Please reinstall the VPN client."))
    draft = run_response(
        _ticket("Corporate VPN drops every hour.", customer_id="alex@example.com"),
        _investigation(
            summary="Matched vpn-gateway.",
            service="vpn-gateway",
            runbook_path="it-helpdesk/runbooks/vpn-disconnect.md",
            related_ticket_ids=["IT-220"],
            evidence=[
                Evidence(
                    source="runbook",
                    ref="it-helpdesk/runbooks/vpn-disconnect.md",
                    excerpt="VPN disconnects hourly",
                )
            ],
        ),
        model=model,
    )
    assert draft.domain == "it-helpdesk"
    assert "IT helpdesk" in _prompt_text(model)
    assert "procedural" in draft.memory_used


def test_checkout_draft_uses_the_oncall_prompt():
    model = FakeModel(ReplyText(subject="Re: checkout", body="Checkout is returning 503. We are checking the database pool."))
    draft = run_response(
        _ticket("Checkout is returning 503 on /pay.", customer_id="oncall@example.com"),
        _investigation(
            summary="Matched checkout-service.",
            service="checkout-service",
            runbook_path="oncall/runbooks/checkout-503.md",
            related_ticket_ids=["INC-401"],
            evidence=[
                Evidence(
                    source="log",
                    ref="oncall/checkout-service",
                    excerpt="ERROR db_pool exhausted",
                )
            ],
        ),
        model=model,
    )
    assert draft.domain == "oncall"
    assert "on-call incident responder" in _prompt_text(model)


def test_model_failure_uses_a_safe_draft():
    draft = run_response(
        _ticket("Authenticator codes loop back to the MFA page."),
        _investigation(),
        model=FakeModel(RuntimeError("model down")),
    )
    assert draft.needs_review is True
    assert draft.subject == "Re: TKT-NEW"
    assert "specialist needs to review" in draft.body
    assert "have not changed your account" in draft.body


def test_refund_promise_is_replaced():
    model = FakeModel(ReplyText(subject="Re: billing", body="We issued a refund of $79 today."))
    draft = run_response(
        _ticket("I was charged twice.", customer_id="marc.w@example.com"),
        _investigation(
            summary="Matched billing-service.",
            service="billing-service",
            runbook_path=None,
            related_ticket_ids=[],
            evidence=[
                Evidence(source="log", ref="support/billing-service", excerpt="charged_usd=79.00")
            ],
        ),
        model=model,
    )
    assert draft.domain == "support"
    assert draft.needs_review is True
    assert "refund" not in draft.body.casefold()


def test_no_evidence_is_flagged_for_review():
    model = FakeModel(ReplyText(subject="Re: sticker", body="A specialist will review this request."))
    draft = run_response(
        _ticket("Please send me a sticker.", customer_id="nobody@example.com"),
        _investigation(
            summary="No local evidence matched this ticket.",
            service=None,
            runbook_path=None,
            related_ticket_ids=[],
            evidence=[],
        ),
        model=model,
    )
    assert draft.needs_review is True
    assert draft.domain == "support"
    assert "episodic" not in draft.memory_used
    assert "procedural" not in draft.memory_used


def test_logs_the_draft(capsys):
    run_response(
        _ticket("Please send me a sticker.", customer_id="nobody@example.com"),
        _investigation(
            summary="No local evidence matched this ticket.",
            service=None,
            runbook_path=None,
            evidence=[],
            related_ticket_ids=[],
        ),
        model=FakeModel(ReplyText(subject="Re: sticker", body="A specialist will review this request.")),
    )
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["ticket_id"] == "TKT-NEW"
    assert payload["domain"] == "support"
    assert payload["needs_review"] is True
