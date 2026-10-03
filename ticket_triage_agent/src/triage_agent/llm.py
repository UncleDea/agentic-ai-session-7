"""Chat-model factory. Provider and model name come from settings."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from triage_agent.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Build a chat model. This does not call the provider."""
    resolved = settings if settings is not None else get_settings()
    return init_chat_model(
        resolved.model_name,
        model_provider=resolved.model_provider,
        temperature=resolved.temperature,
    )
