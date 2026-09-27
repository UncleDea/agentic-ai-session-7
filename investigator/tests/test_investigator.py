"""Investigator tests against the repo's local data files. No API calls."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from triage_agent.schemas import TicketState
from investigator import Investigation, run_investigation

RECEIVED_AT = datetime(2026, 5, 10, 9, 14, 22, tzinfo=timezone.utc)


def _ticket(
    *,
    ticket_id: str = "TKT-NEW",
    raw_text: str,
    customer_id: str | None,
    category: str = "technical",
    severity: str = "medium",
) -> TicketState:
    return TicketState(
        ticket_id=ticket_id,
        raw_text=raw_text,
        customer_id=customer_id,
        channel="email",
        received_at=RECEIVED_AT,
        category=category,
        severity=severity,
        confidence=0.9,
        triage_reasoning="test",
        status="triaged",
        next_node_hint="investigator",
    )


def _refs(result: Investigation, source: str) -> list[str]:
    return [item.ref for item in result.evidence if item.source == source]


def test_mfa_ticket_finds_history_logs_and_runbook():
    result = run_investigation(
        _ticket(
            raw_text="Cannot log in. Authenticator codes loop back to the MFA page.",
            customer_id="priya.s@example.com",
            category="account",
            severity="high",
        )
    )
    assert result.status == "investigated"
    assert result.service == "auth-service"
    assert result.related_ticket_ids == ["TKT-1001", "TKT-0950", "TKT-0901"]
    assert result.runbook_path == "support/runbooks/mfa-loop.md"
    assert "support/runbooks/mfa-loop.md" in _refs(result, "runbook")
    assert any("mfa_verify" in item.excerpt for item in result.evidence)
    assert result.next_node_hint == "responder"


def test_billing_ticket_finds_the_renewal_charge():
    result = run_investigation(
        _ticket(
            ticket_id="TKT-1002",
            raw_text="Cancelled Pro but I was still charged $79.",
            customer_id="marc.w@example.com",
            category="billing",
        )
    )
    assert result.service == "billing-service"
    assert "TKT-1002" not in result.related_ticket_ids
    assert any("charged_usd=79.00" in item.excerpt for item in result.evidence)
    assert any(item.source == "metric" for item in result.evidence)


def test_export_ticket_finds_the_render_timeout():
    result = run_investigation(
        _ticket(
            raw_text="The report export hangs at 50%.",
            customer_id="lisa.f@example.com",
            category="technical",
        )
    )
    assert result.service == "reports-service"
    assert result.related_ticket_ids == ["TKT-1004"]
    assert any("timeout=30s" in item.excerpt for item in result.evidence)


def test_vpn_ticket_finds_the_it_runbook():
    result = run_investigation(
        _ticket(
            raw_text="Corporate VPN drops every hour.",
            customer_id="alex@example.com",
            category="technical",
        )
    )
    assert result.service == "vpn-gateway"
    assert "IT-220" in result.related_ticket_ids
    assert result.runbook_path == "it-helpdesk/runbooks/vpn-disconnect.md"


def test_checkout_alert_finds_the_oncall_runbook():
    result = run_investigation(
        _ticket(
            raw_text="Checkout is returning 503 on /pay.",
            customer_id="oncall@example.com",
            category="technical",
            severity="critical",
        )
    )
    assert result.service == "checkout-service"
    assert result.related_ticket_ids == ["INC-401"]
    assert result.runbook_path == "oncall/runbooks/checkout-503.md"


def test_unknown_ticket_does_not_invent_evidence():
    result = run_investigation(
        _ticket(
            raw_text="Please send me a sticker.",
            customer_id="nobody@example.com",
            category="other",
            severity="low",
        )
    )
    assert result.service is None
    assert result.evidence == []
    assert result.related_ticket_ids == []
    assert result.runbook_path is None
    assert result.summary == "No local evidence matched this ticket."


def test_logs_a_json_line(capsys):
    run_investigation(
        _ticket(
            ticket_id="TKT-LOG",
            raw_text="Please send me a sticker.",
            customer_id="nobody@example.com",
            category="other",
        )
    )
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {
        "ticket_id": "TKT-LOG",
        "service": None,
        "related_tickets": 0,
        "evidence": 0,
    }


def test_missing_data_dir_raises(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        run_investigation(
            _ticket(raw_text="VPN down", customer_id="alex@example.com"),
            data_dir=missing,
        )
