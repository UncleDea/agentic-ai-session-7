"""Deliver outbox records by email. The outbox write itself stays local."""

from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from send.agent import default_outbox


class MailTransport(Protocol):
    def send(self, record: dict) -> None:
        """Deliver one outbox record."""


def default_ledger() -> Path:
    return default_outbox().with_name("delivered.jsonl")


def deliver_pending(
    *,
    outbox: Path | None = None,
    ledger: Path | None = None,
    transport: MailTransport,
) -> list[dict]:
    """Send every outbox row that is not already in the ledger."""
    outbox_path = outbox or default_outbox()
    ledger_path = ledger or default_ledger()
    done = _delivered_ids(ledger_path)
    sent: list[dict] = []
    for record in _read_jsonl(outbox_path):
        outbound_id = str(record.get("outbound_id", ""))
        if not outbound_id or outbound_id in done:
            continue
        transport.send(record)
        _append_jsonl(
            ledger_path,
            {
                "outbound_id": outbound_id,
                "recipient": record.get("recipient"),
                "delivered_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        done.add(outbound_id)
        sent.append(record)
    return sent


def smtp_transport_from_env() -> "SmtpTransport | None":
    host = os.environ.get("SMTP_HOST", "").strip()
    if not host:
        return None
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("MAIL_FROM", "").strip() or user
    if not user or not password or not sender:
        raise SystemExit("Set SMTP_USER, SMTP_PASSWORD, and MAIL_FROM before delivering mail.")
    port = int(os.environ.get("SMTP_PORT", "587"))
    return SmtpTransport(host=host, port=port, user=user, password=password, sender=sender)


class SmtpTransport:
    def __init__(self, *, host: str, port: int, user: str, password: str, sender: str) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.sender = sender

    def send(self, record: dict) -> None:
        recipient = str(record.get("recipient", "")).strip()
        if not recipient:
            raise ValueError(f"{record.get('outbound_id')} has no recipient.")
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = recipient
        message["Subject"] = str(record.get("subject", ""))
        message.set_content(str(record.get("body", "")))
        with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(self.user, self.password)
            smtp.send_message(message)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _delivered_ids(path: Path) -> set[str]:
    return {str(row.get("outbound_id", "")) for row in _read_jsonl(path)}


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
