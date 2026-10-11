"""Call the AgentCore runtime. The review page uses this when an ARN is set."""

from __future__ import annotations

import json
import os
from typing import Any

_MIN_SESSION = 33


def parse_runtime_body(body: Any) -> dict:
    """Turn an invoke body into the action dict."""
    if hasattr(body, "read"):
        body = body.read()
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    parsed = json.loads(body)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError("runtime response was not an object")
    return parsed


def invoke_runtime(payload: dict, *, client=None) -> dict:
    """Invoke the configured runtime. The session id is shared by every call."""
    arn = os.environ.get("AGENT_RUNTIME_ARN", "").strip()
    session_id = os.environ.get("AGENT_SESSION_ID", "").strip()
    if not arn:
        raise RuntimeError("AGENT_RUNTIME_ARN is not set.")
    if len(session_id) < _MIN_SESSION:
        raise RuntimeError("AGENT_SESSION_ID must be at least 33 characters.")
    if client is None:
        import boto3

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        client = boto3.client("bedrock-agentcore", region_name=region)
    response = client.invoke_agent_runtime(
        agentRuntimeArn=arn,
        runtimeSessionId=session_id,
        payload=json.dumps(payload).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )
    return parse_runtime_body(response["response"])
