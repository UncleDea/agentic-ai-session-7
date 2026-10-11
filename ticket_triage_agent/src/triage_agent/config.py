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
    model_provider: Literal["bedrock_converse"] = "bedrock_converse"
    model_name: str = "us.anthropic.claude-sonnet-4-6"
    temperature: float = 0.0
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    max_iterations: int = Field(default=10, ge=1)
    max_revisions: int = Field(default=2, ge=0)
    max_tool_calls: int = Field(default=8, ge=1)
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
    memory_db_path: str | None = None
    checkpoint_db_path: str | None = None
    outbox_dir: str = "outbox"
    escalations_dir: str = "escalations"
    embedding_model: str | None = None
    embedding_dims: int = Field(default=1024, ge=1)
    kb_min_score: float = Field(default=0.2, ge=0, le=1)
    kb_relative_margin: float = Field(default=0.1, ge=0, le=1)
    memory_search_limit: int = Field(default=3, ge=1)


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
        "MEMORY_DB_PATH": "memory_db_path",
        "CHECKPOINT_DB_PATH": "checkpoint_db_path",
        "OUTBOX_DIR": "outbox_dir",
        "ESCALATIONS_DIR": "escalations_dir",
        "EMBEDDING_MODEL": "embedding_model",
        "EMBEDDING_DIMS": "embedding_dims",
        "KB_MIN_SCORE": "kb_min_score",
        "KB_RELATIVE_MARGIN": "kb_relative_margin",
        "MEMORY_SEARCH_LIMIT": "memory_search_limit",
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


def resolve_dir(path: str) -> Path:
    """Absolute paths stay absolute. Relative paths start at the package root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _ENV_PATH.parent / candidate


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from the environment. Existing variables win over `.env`."""
    load_dotenv(_ENV_PATH, override=False)
    return Settings(**_overrides())
