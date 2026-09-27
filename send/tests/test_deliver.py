"""Outbox delivery tests. The mail transport is fake, so nothing is emailed."""

import json
from pathlib import Path

import pytest

from send.__main__ import main
from send.deliver import deliver_pending


class FakeMail:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, record: dict) -> None:
        self.sent.append(record)


def _outbox(path: Path, *records: dict) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _record(outbound_id: str, recipient: str = "priya.s@example.com") -> dict:
    return {
        "outbound_id": outbound_id,
        "ticket_id": "TKT-1",
        "recipient": recipient,
        "subject": "Re: MFA loop",
        "body": "Please reopen your authenticator app.",
        "sent_at": "2026-09-26T00:00:00+00:00",
    }


def test_deliver_sends_only_new_records(tmp_path: Path):
    outbox = tmp_path / "sent.jsonl"
    ledger = tmp_path / "delivered.jsonl"
    _outbox(outbox, _record("OUT-1"), _record("OUT-2"))
    ledger.write_text(json.dumps({"outbound_id": "OUT-1", "delivered_at": "2026-09-26T00:00:00+00:00"}) + "\n", encoding="utf-8")
    mail = FakeMail()
    sent = deliver_pending(outbox=outbox, ledger=ledger, transport=mail)
    assert [row["outbound_id"] for row in sent] == ["OUT-2"]
    assert [row["outbound_id"] for row in mail.sent] == ["OUT-2"]
    again = deliver_pending(outbox=outbox, ledger=ledger, transport=mail)
    assert again == []
    assert len(mail.sent) == 1


def test_failed_send_is_not_marked_delivered(tmp_path: Path):
    outbox = tmp_path / "sent.jsonl"
    ledger = tmp_path / "delivered.jsonl"
    _outbox(outbox, _record("OUT-9"))

    class Boom:
        def send(self, record: dict) -> None:
            raise RuntimeError("mailbox unavailable")

    with pytest.raises(RuntimeError):
        deliver_pending(outbox=outbox, ledger=ledger, transport=Boom())
    assert not ledger.exists()


def test_command_refuses_to_deliver_without_smtp(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["deliver"], outbox=tmp_path / "sent.jsonl", ledger=tmp_path / "delivered.jsonl")
    assert "SMTP_HOST" in str(exc.value)


def test_command_prints_delivered_ids(tmp_path: Path, capsys):
    outbox = tmp_path / "sent.jsonl"
    _outbox(outbox, _record("OUT-3"))
    code = main(
        ["deliver"],
        transport=FakeMail(),
        outbox=outbox,
        ledger=tmp_path / "delivered.jsonl",
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {"delivered": 1, "outbound_ids": ["OUT-3"]}
