# Phase 4 — Memory layer

No Responder, HITL learning, or graph `compile(store=...)`. Those stay for Phases 5 and 6. This phase adds the store, the three namespaces, seed data, retrieval, and `search_kb` reading the semantic namespace.

## Versions checked in the project venv

Interpreter: `ticket_triage_agent\.venv\Scripts\python` (Python 3.14.7).

Already in the venv, matching `ticket_triage_agent/pyproject.toml`:

- `langgraph==1.2.12` — before the install, `langgraph.store` contained only `base` and `memory`
- `langgraph-checkpoint==4.2.0`
- `langchain==1.4.3` — `init_embeddings(model, *, provider=None, **kwargs)` imports and is usable
- `langchain-core==1.6.6`
- `langchain-openai==1.6.7`

Installed into this venv only (not the global interpreter):

- `langgraph-checkpoint-sqlite==3.1.1`
- `sqlite-vec==0.1.9`
- `aiosqlite==0.22.1` came in as a dependency of the SQLite package. Our code uses sync `SqliteStore`, so `aiosqlite` stays transitive and is not a direct pin.

`pyproject.toml` still needs the two direct pins. Plan mode blocked that file edit; it is the first change once this plan is approved:

```toml
"langgraph-checkpoint-sqlite==3.1.1",
"sqlite-vec==0.1.9",
```

## Store API, from the installed packages

`langgraph.store.base.BaseStore` in this venv:

- `put(namespace, key, value, index=None, *, ttl=NOT_GIVEN)`
- `get(namespace, key, *, refresh_ttl=None) -> Item | None`
- `search(namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, refresh_ttl=None) -> list[SearchItem]`

`SearchItem` attributes: `key`, `value`, `namespace`, `score`, `created_at`, `updated_at`, plus `dict()`.

`InMemoryStore` index config uses `dims`, `embed`, and `fields`. The installed `SqliteStore` does not. Its `_ensure_index_config` reads `text_fields` and otherwise embeds the whole JSON (`$`). A venv smoke test confirmed that: `text_fields: ["text"]` ranked an `alpha` row at score `1.0`; the same call with `fields: ["text"]` embedded the whole document and scored `0.14`.

`SqliteStore.from_conn_string` is a context manager and closes the connection on exit. It opens SQLite with `check_same_thread=False` and `isolation_level=None`. A connection left at the default isolation level fails on `put` with `sqlite3.OperationalError: cannot start a transaction within a transaction`, because the store issues its own `BEGIN`.

Open the dev store like this, and keep the connection for the process:

```python
conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
store = SqliteStore(conn, index={
    "dims": dims,
    "embed": embeddings,       # Embeddings instance or callable
    "text_fields": ["text"],
})
store.setup()
```

`setup()` is required before first use and is safe to call again. A second connection to the same file, after `close()`, returned the stored row.

Dev file: `ticket_triage_agent/data/memory.sqlite`, gitignored. Tests use `:memory:` or a temp file, opened the same way.

Live embeddings: `init_embeddings("openai:text-embedding-3-small")` from `langchain.embeddings` (1536 dims). Chat can still be Anthropic; the live memory test needs `OPENAI_API_KEY`.

Offline embeddings: a deterministic callable `(list[str]) -> list[list[float]]`. `IndexConfig.embed` accepts that, and the venv smoke test used one. Same function for put and search. No API calls.

## Namespaces and value shapes

Keys are stable ids. `put` on the same key overwrites, which is what makes article and episode seeding safe to run twice.

- Semantic `("semantic", "kb")`
  - Key: article id, such as `KB-002`
  - Value: `id`, `kind` (`kb`), `title`, `body`, `source` (`kb:KB-002`), `text` (title plus body). `text` is the only embedded field.
- Episodic `("episodic", "resolved_tickets")`
  - Key: episode id, such as `EP-001`
  - Value: `id`, `ticket_id`, `category`, `subject`, `body`, `approved_reply`, `text` (subject plus body only, so search matches the problem)
- Procedural `("procedural", "response_rules")`
  - Keys `v1`, `v2`, ... value: `version` (int), `text` (the style guide), `note`
  - Key `current` value: `{"version": N}` and no `text`, so the pointer is not embedded
- Metadata `("meta", "store")` key `embeddings`: `model`, `dims`. `open_store` refuses to search a file stamped with a different model.

`search_kb` stays `search_kb(self, query: str) -> LookupResult`. It searches `("semantic", "kb")`, drops scores below `kb_min_score` (start at `0.2`), dedupes by key, and returns at most 3 records with `id`, `source`, `title`, and `body`. `rank_keywords` stays for past-ticket search only.

## Procedural versioning and rollback

- `get_current(store)` reads `current`, then gets `v{N}`, and returns that version's text and number.
- `put_version(store, text, note)` finds the highest `vN`, writes `v{N+1}`, and points `current` at it. Older rows stay.
- `rollback(store, version)` points `current` at an existing `v{version}`. It does not delete later versions. A missing version raises.
- `list_versions(store)` returns every `vN` row, newest version first.
- `seed_procedures`: if `current` is missing, write `v1` with note `seed` and point `current` at 1. If `current` is still 1, overwrite the `v1` text in place. If `current` is already later, do nothing, so a second run cannot add a duplicate version or undo a rollback.

## Files

New:

- `ticket_triage_agent/src/triage_agent/memory/store.py` — namespace constants, `fake_embeddings()`, `open_store(path, embeddings)`, model stamp
- `ticket_triage_agent/src/triage_agent/memory/semantic.py` — `upsert_article`, `get_article`, `search_articles`, `seed_articles`
- `ticket_triage_agent/src/triage_agent/memory/episodic.py` — `upsert_episode`, `get_episode`, `search_episodes` (limit 3), `seed_episodes`
- `ticket_triage_agent/src/triage_agent/memory/procedural.py` — `seed_procedures`, `get_current`, `put_version`, `rollback`, `list_versions`
- `ticket_triage_agent/data/seed_episodes.json` — resolved billing, stuck-tracking, and damaged-item examples with approved replies
- `ticket_triage_agent/data/seed_procedures.md` — tone, sign-off, never promise a delivery date or a refund that is not in the findings, escalation phrasing
- `ticket_triage_agent/scripts/seed_memory.py` — loads `kb_articles.json`, `seed_episodes.json`, and `seed_procedures.md` into the dev sqlite
- `ticket_triage_agent/tests/test_memory.py`

Edit:

- `ticket_triage_agent/pyproject.toml` — pin the two packages above
- `ticket_triage_agent/src/triage_agent/memory/__init__.py` — export the public functions
- `ticket_triage_agent/src/triage_agent/tools/knowledge_base.py` — replace `JsonKnowledgeBase` with `StoreKnowledgeBase(store)`. Lookup reads the store, not a second in-memory copy of the articles.
- `ticket_triage_agent/src/triage_agent/tools/__init__.py` — default KB is the store. Tool text says search is by meaning.
- `ticket_triage_agent/src/triage_agent/nodes/investigator.py` — pass `settings` into `build_tools`
- `ticket_triage_agent/src/triage_agent/config.py` — `memory_db_path` (default `None`, meaning `:memory:`), `embedding_model` (default `None`, meaning fake), `embedding_dims` (1536 when live), `kb_min_score` (`0.2`), `memory_search_limit` (3)
- `ticket_triage_agent/.env.example` — `MEMORY_DB_PATH=data/memory.sqlite`, `EMBEDDING_MODEL=text-embedding-3-small`, `EMBEDDING_DIMS=1536`
- `ticket_triage_agent/.gitignore` — `data/memory.sqlite`
- `ticket_triage_agent/tests/test_investigator.py` — the three direct KB tests seed a temp store with fake embeddings, then call `search_kb`
- `ticket_triage_agent/tests/test_config.py` — cover the new settings and clear the new env vars

`get_settings()` reads `.env`. Tests that pass `Settings()` stay on `:memory:` plus fake embeddings, so `search_kb("zzzznomatch")` in the existing investigator graph test does not call the network or open the dev database.

Article and episode seeds: `put` the same key again. A second run updates text and leaves the count unchanged.

## Test cases

Offline, in `tests/test_memory.py`, on `:memory:` or a temp file with fake embeddings:

- Semantic put/get round-trip
- Semantic search ranks a close fact above an unrelated one
- `seed_articles` twice leaves 7 rows
- `search_kb("charged twice")` includes `KB-001` and excludes `KB-007`
- `search_kb("zzzznomatch")` is `found=False`
- T-004 subject `"Order 90210 has not moved"` returns `KB-002` first
- T-005 subject `"Refund for order 7712"` excludes `KB-001` and `KB-002`
- Episodic put/get, search limit 3, seed twice does not duplicate, `approved_reply` is on the value
- Procedural seed sets version 1; second seed does not create `v2`
- `put_version` makes version 2 current and leaves version 1 readable
- `rollback(1)` makes version 1 current and keeps version 2
- Rollback to a missing version raises
- Persistence: write an episode to a temp sqlite file, close, reopen with a new connection, `get` returns it

Live, `@pytest.mark.live`, skipped when `OPENAI_API_KEY` is unset:

- Seed `kb_articles.json` into a temp sqlite with `text-embedding-3-small`
- `search_kb("my package hasn't budged in a week")` returns `KB-002` first
- The test asserts that query shares no content tokens with KB-002, so a keyword matcher cannot satisfy it

Default `pytest` stays on `-m 'not live'`. The Phase 5 xfail stays untouched.
