"""Factory wiring only. The model is never invoked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from triage_agent.config import Settings, get_settings
from triage_agent.llm import get_chat_model


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_factory_passes_provider_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    calls: dict = {}

    def fake_init(model_name: str, **kwargs):
        calls["model_name"] = model_name
        calls["kwargs"] = kwargs
        return model

    monkeypatch.setattr("triage_agent.llm.init_chat_model", fake_init)
    settings = Settings(
        model_provider="bedrock_converse",
        model_name="us.anthropic.claude-sonnet-4-6",
        temperature=0.0,
    )
    result = get_chat_model(settings)
    assert result is model
    assert calls["model_name"] == "us.anthropic.claude-sonnet-4-6"
    assert calls["kwargs"]["model_provider"] == "bedrock_converse"
    assert calls["kwargs"]["temperature"] == 0.0
    model.invoke.assert_not_called()


def test_unknown_provider_does_not_construct_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("init_chat_model was called")

    monkeypatch.setattr("triage_agent.llm.init_chat_model", fail_if_called)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("MODEL_PROVIDER", "not-a-provider")
    with pytest.raises(ValidationError):
        get_chat_model()
