"""Test isolation. Settings must not read the project `.env`."""

from __future__ import annotations

import pytest

from triage_agent.config import get_settings

# Names `get_settings` copies out of the environment, plus the API keys in `.env`.
_SETTING_ENV = (
    "MODEL_PROVIDER",
    "MODEL_NAME",
    "TEMPERATURE",
    "CONFIDENCE_THRESHOLD",
    "MAX_ITERATIONS",
    "MAX_REVISIONS",
    "MAX_TOOL_CALLS",
    "HITL_PRIORITIES",
    "HITL_SENTIMENTS",
    "HITL_CATEGORIES",
    "INVESTIGATION_CATEGORIES",
    "MEMORY_DB_PATH",
    "CHECKPOINT_DB_PATH",
    "OUTBOX_DIR",
    "ESCALATIONS_DIR",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMS",
    "KB_MIN_SCORE",
    "KB_RELATIVE_MARGIN",
    "MEMORY_SEARCH_LIMIT",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _settings_ignore_dotenv(monkeypatch: pytest.MonkeyPatch):
    """Drop project dotenv values before each test.

    `get_settings` loads `.env` when a name is missing, so deleting the
    variables is not enough. Tests that need a value set it with monkeypatch.
    """

    def _skip_dotenv(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr("triage_agent.config.load_dotenv", _skip_dotenv)
    for name in _SETTING_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
