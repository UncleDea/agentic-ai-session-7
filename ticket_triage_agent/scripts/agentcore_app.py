"""AgentCore entrypoint. The review page calls run, list, get, and resume."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from triage_agent.config import get_settings  # noqa: E402
from triage_agent.graph import build_graph  # noqa: E402
from triage_agent.runtime_api import SESSION_ROOT, handle_action, prepare_session_files  # noqa: E402

_SOURCE_MEMORY = _ROOT / "data" / "memory.sqlite"
_graph = None


def _ready_graph():
    global _graph
    if _graph is None:
        prepare_session_files(SESSION_ROOT, _SOURCE_MEMORY)
        os.environ.setdefault("AWS_REGION", "us-east-1")
        os.environ.setdefault("MODEL_PROVIDER", "bedrock_converse")
        os.environ.setdefault("MODEL_NAME", "us.anthropic.claude-sonnet-4-6")
        os.environ.setdefault("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        os.environ.setdefault("EMBEDDING_DIMS", "1024")
        get_settings.cache_clear()
        _graph = build_graph(settings=get_settings())
    return _graph


def main() -> None:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload, context=None):
        body = payload if isinstance(payload, dict) else {}
        return handle_action(_ready_graph(), body)

    app.run()


if __name__ == "__main__":
    main()
