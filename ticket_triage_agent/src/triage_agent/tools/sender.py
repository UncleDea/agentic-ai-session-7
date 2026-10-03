"""Outbound delivery and escalation records. JSON files stand in for a real channel."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class OutboundMessage(BaseModel):
    """One sent reply. The file is the record of this run."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    thread_id: str
    final_reply: str
    sent_at: datetime


class EscalationRecord(BaseModel):
    """An open follow-up. status stays open until a later phase closes it."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    thread_id: str
    reasons: list[str]
    reply: str
    created_at: datetime
    status: Literal["open"] = "open"


class Sender(Protocol):
    """Delivers one reply per run. Email or Slack can replace the outbox."""

    def get(self, thread_id: str) -> OutboundMessage | None:
        """Return this run's message, or None when it was not sent."""

    def send(self, message: OutboundMessage) -> OutboundMessage:
        """Deliver the message. If this run was already sent, return that record."""


class EscalationLog(Protocol):
    """Tracks promised follow-ups. A queue can replace the JSON files."""

    def get(self, thread_id: str) -> EscalationRecord | None:
        """Return this run's case, or None when none was opened."""

    def open_case(self, record: EscalationRecord) -> EscalationRecord:
        """Open a case. If this run already has one, return it unchanged."""


class OutboxSender:
    """Writes one JSON file per run under the outbox directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, thread_id: str) -> OutboundMessage | None:
        return _read(self._path(thread_id), OutboundMessage)

    def send(self, message: OutboundMessage) -> OutboundMessage:
        path = self._path(message.thread_id)
        if not _create(path, _dump(message)):
            existing = _read(path, OutboundMessage)
            if existing is None:
                raise ValueError(f"outbox record {path.name} is unreadable")
            return existing
        return message

    def _path(self, thread_id: str) -> Path:
        return self.directory / _filename(thread_id)


class JsonEscalationLog:
    """Writes one JSON file per run under the escalations directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, thread_id: str) -> EscalationRecord | None:
        return _read(self._path(thread_id), EscalationRecord)

    def open_case(self, record: EscalationRecord) -> EscalationRecord:
        path = self._path(record.thread_id)
        if not _create(path, _dump(record)):
            existing = _read(path, EscalationRecord)
            if existing is None:
                raise ValueError(f"escalation record {path.name} is unreadable")
            return existing
        return record

    def _path(self, thread_id: str) -> Path:
        return self.directory / _filename(thread_id)


def list_open(directory: Path) -> tuple[list[EscalationRecord], list[str]]:
    """Open escalation records, plus file names that could not be read.

    A record whose status is not open fails validation and is skipped.
    """
    if not directory.is_dir():
        return [], []
    records: list[EscalationRecord] = []
    skipped: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = EscalationRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, ValidationError):
            skipped.append(path.name)
            continue
        if record.status == "open":
            records.append(record)
    records.sort(key=lambda row: (row.created_at, row.thread_id))
    return records, skipped


def _filename(thread_id: str) -> str:
    cleaned = _UNSAFE.sub("_", thread_id).strip("._")
    if not cleaned:
        raise ValueError(f"unsafe thread id: {thread_id!r}")
    return f"{cleaned}.json"


def _dump(model: BaseModel) -> str:
    return model.model_dump_json(indent=2) + "\n"


def _read(path: Path, model: type[OutboundMessage] | type[EscalationRecord]):
    if not path.is_file():
        return None
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _create(path: Path, text: str) -> bool:
    """Create the file only when it is absent. False means another write won."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(path, flags)
    except FileExistsError:
        return False
    try:
        os.write(fd, text.encode("utf-8"))
    except Exception:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    os.close(fd)
    return True
