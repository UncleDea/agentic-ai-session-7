"""Load kb articles, seed episodes, and seed procedures into the dev store."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from triage_agent.config import get_settings  # noqa: E402
from triage_agent.memory.episodic import seed_episodes  # noqa: E402
from triage_agent.memory.procedural import seed_procedures  # noqa: E402
from triage_agent.memory.semantic import seed_articles  # noqa: E402
from triage_agent.memory.store import DATA_DIR, bedrock_embeddings, open_store  # noqa: E402

_DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
_DEFAULT_EMBEDDING_DIMS = 1024


def main() -> None:
    settings = get_settings()
    model = settings.embedding_model or _DEFAULT_EMBEDDING_MODEL
    dims = settings.embedding_dims if settings.embedding_model else _DEFAULT_EMBEDDING_DIMS
    path = settings.memory_db_path or str(DATA_DIR / "memory.sqlite")
    embeddings = bedrock_embeddings(model, dimensions=dims)
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
