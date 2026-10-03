"""Runtime settings. Thresholds and model choice live here, not in nodes."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from triage_agent.state import Category, Priority, Sentiment

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseModel):
    model_provider: Literal["openai", "anthropic"] = "openai"
    model_name: str = "gpt-5.5"
    temperature: float = 0.0
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    max_iterations: int = Field(default=10, ge=1)
    max_revisions: int = Field(default=2, ge=0)
    max_tool_calls: int = Field(default=6, ge=1)
    hitl_priorities: list[Priority] = Field(default_factory=lambda: [Priority.P1])
    hitl_sentiments: list[Sentiment] = Field(
        default_factory=lambda: [Sentiment.negative]
    )
    hitl_categories: list[Category] = Field(
        default_factory=lambda: [Category.refund, Category.security, Category.legal]
    )
    investigation_categories: list[Category] = Field(
        default_factory=lambda: [
            Category.billing,
            Category.technical,
            Category.account,
            Category.shipping,
            Category.refund,
            Category.security,
            Category.legal,
            Category.other,
        ]
    )


def _csv(name: str) -> list[str] | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _overrides() -> dict[str, object]:
    data: dict[str, object] = {}
    scalar = {
        "MODEL_PROVIDER": "model_provider",
        "MODEL_NAME": "model_name",
        "TEMPERATURE": "temperature",
        "CONFIDENCE_THRESHOLD": "confidence_threshold",
        "MAX_ITERATIONS": "max_iterations",
        "MAX_REVISIONS": "max_revisions",
        "MAX_TOOL_CALLS": "max_tool_calls",
    }
    for env_name, field_name in scalar.items():
        value = os.getenv(env_name)
        if value is not None and value.strip() != "":
            data[field_name] = value.strip()
    lists = {
        "HITL_PRIORITIES": "hitl_priorities",
        "HITL_SENTIMENTS": "hitl_sentiments",
        "HITL_CATEGORIES": "hitl_categories",
        "INVESTIGATION_CATEGORIES": "investigation_categories",
    }
    for env_name, field_name in lists.items():
        values = _csv(env_name)
        if values is not None:
            data[field_name] = values
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from the environment. Existing variables win over `.env`."""
    load_dotenv(_ENV_PATH, override=False)
    return Settings(**_overrides())
