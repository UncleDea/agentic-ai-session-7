"""Runtime invoke parsing. No AWS calls."""

from __future__ import annotations

import json

import pytest

from triage_agent.runtime_client import invoke_runtime, parse_runtime_body


class _Body:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def read(self) -> bytes:
        return self.raw


class _Client:
    def __init__(self) -> None:
        self.kwargs = None

    def invoke_agent_runtime(self, **kwargs):
        self.kwargs = kwargs
        return {"response": _Body(json.dumps({"paused": []}).encode("utf-8"))}


def test_parse_runtime_body_reads_a_stream() -> None:
    parsed = parse_runtime_body(_Body(b'{"outcome": "paused"}'))
    assert parsed == {"outcome": "paused"}


def test_invoke_runtime_sends_the_shared_session(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:000000000000:runtime/ticket")
    monkeypatch.setenv("AGENT_SESSION_ID", "ticket-review-session-000000000001")
    client = _Client()
    body = invoke_runtime({"action": "list"}, client=client)
    assert body == {"paused": []}
    assert client.kwargs["runtimeSessionId"] == "ticket-review-session-000000000001"
    assert json.loads(client.kwargs["payload"]) == {"action": "list"}


def test_invoke_runtime_rejects_a_short_session(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:000000000000:runtime/ticket")
    monkeypatch.setenv("AGENT_SESSION_ID", "short")
    with pytest.raises(RuntimeError, match="33"):
        invoke_runtime({"action": "list"}, client=_Client())
