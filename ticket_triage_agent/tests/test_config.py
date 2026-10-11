"""Offline settings tests. No API key is required."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from triage_agent.config import get_settings
from triage_agent.state import Category, Priority, Sentiment

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
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMS",
    "KB_MIN_SCORE",
    "KB_RELATIVE_MARGIN",
    "MEMORY_SEARCH_LIMIT",
    "AWS_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SETTING_ENV:
        monkeypatch.delenv(name, raising=False)


def test_defaults_load_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    settings = get_settings()
    assert settings.model_provider == "bedrock_converse"
    assert settings.model_name == "us.anthropic.claude-sonnet-4-6"
    assert settings.temperature == 0.0
    assert settings.confidence_threshold == 0.7
    assert settings.max_iterations == 10
    assert settings.max_tool_calls == 8
    assert settings.hitl_priorities == [Priority.P1]
    assert settings.hitl_sentiments == [Sentiment.negative]
    assert settings.hitl_categories == [
        Category.refund,
        Category.security,
        Category.legal,
    ]
    assert settings.investigation_categories == [
        Category.billing,
        Category.technical,
        Category.account,
        Category.shipping,
        Category.refund,
        Category.security,
        Category.legal,
        Category.other,
    ]
    assert settings.memory_db_path is None
    assert settings.checkpoint_db_path is None
    assert settings.embedding_model is None
    assert settings.embedding_dims == 1024
    assert settings.kb_min_score == 0.2
    assert settings.kb_relative_margin == 0.1
    assert settings.memory_search_limit == 3


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "bedrock_converse")
    monkeypatch.setenv("MODEL_NAME", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("MAX_ITERATIONS", "3")
    monkeypatch.setenv("INVESTIGATION_CATEGORIES", "feedback")
    monkeypatch.setenv("MEMORY_DB_PATH", "data/memory.sqlite")
    monkeypatch.setenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")
    monkeypatch.setenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("KB_MIN_SCORE", "0.4")
    monkeypatch.setenv("KB_RELATIVE_MARGIN", "0.25")
    monkeypatch.setenv("MEMORY_SEARCH_LIMIT", "2")
    settings = get_settings()
    assert settings.model_provider == "bedrock_converse"
    assert settings.model_name == "us.anthropic.claude-sonnet-4-6"
    assert settings.confidence_threshold == 0.4
    assert settings.max_iterations == 3
    assert settings.investigation_categories == [Category.feedback]
    assert settings.memory_db_path == "data/memory.sqlite"
    assert settings.checkpoint_db_path == "data/checkpoints.sqlite"
    assert settings.embedding_model == "amazon.titan-embed-text-v2:0"
    assert settings.kb_min_score == 0.4
    assert settings.kb_relative_margin == 0.25
    assert settings.memory_search_limit == 2


def test_unknown_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "azure")
    with pytest.raises(ValidationError):
        get_settings()
