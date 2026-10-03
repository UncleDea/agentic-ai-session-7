"""Load kb articles, seed episodes, and seed procedures into the dev store."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from langchain.embeddings import init_embeddings  # noqa: E402

from triage_agent.config import get_settings  # noqa: E402
from triage_agent.memory.episodic import seed_episodes  # noqa: E402
from triage_agent.memory.procedural import seed_procedures  # noqa: E402
from triage_agent.memory.semantic import seed_articles  # noqa: E402
from triage_agent.memory.store import DATA_DIR, open_store  # noqa: E402


def _openai_key() -> str:
    get_settings()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    os.environ["OPENAI_API_KEY"] = key
    return key


def main() -> None:
    _openai_key()
    settings = get_settings()
    model = settings.embedding_model or "text-embedding-3-small"
    dims = settings.embedding_dims if settings.embedding_model else 1536
    path = settings.memory_db_path or str(DATA_DIR / "memory.sqlite")
    embeddings = init_embeddings(model, provider="openai")
    store = open_store(path, embeddings, dims=dims, model=model)
    try:
        articles = seed_articles(store)
        episodes = seed_episodes(store)
        seed_procedures(store)
    finally:
        store.conn.close()
    print(f"seeded {articles} articles, {episodes} episodes, procedures into {path}")


if __name__ == "__main__":
    main()
