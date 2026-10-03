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
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
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
    assert settings.model_provider == "openai"
    assert settings.model_name == "gpt-5.5"
    assert settings.temperature == 0.0
    assert settings.confidence_threshold == 0.7
    assert settings.max_iterations == 10
    assert settings.hitl_priorities == [Priority.P1]
    assert settings.hitl_sentiments == [Sentiment.negative]
    assert settings.hitl_categories == [
        Category.refund,
        Category.security,
        Category.legal,
    ]


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("MAX_ITERATIONS", "3")
    settings = get_settings()
    assert settings.model_provider == "anthropic"
    assert settings.confidence_threshold == 0.4
    assert settings.max_iterations == 3


def test_unknown_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "azure")
    with pytest.raises(ValidationError):
        get_settings()
