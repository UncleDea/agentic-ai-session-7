"""Memory store. Embeddings are fake unless the test is marked live."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
from langchain.embeddings import init_embeddings

from triage_agent.config import Settings, get_settings
from triage_agent.nodes.investigator import kb_query
from triage_agent.memory.episodic import (
    get_episode,
    search_episodes,
    seed_episodes,
    upsert_episode,
)
from triage_agent.memory.procedural import (
    get_current,
    list_versions,
    put_version,
    rollback,
    seed_procedures,
)
from triage_agent.memory.semantic import (
    get_article,
    search_articles,
    seed_articles,
    upsert_article,
)
from triage_agent.memory.store import (
    EPISODIC,
    FAKE_DIMS,
    FAKE_MODEL,
    PROCEDURAL,
    SEMANTIC,
    fake_embeddings,
    open_store,
    select_hits,
)
from triage_agent.tools.knowledge_base import (
    EMPTY_KB_WARNING,
    StoreKnowledgeBase,
    content_tokens,
)

_CHARGE = {
    "id": "KB-001",
    "title": "Duplicate charge",
    "body": "When a customer was charged twice for one order, confirm both amounts.",
    "source": "kb:KB-001",
}
_GIFT = {
    "id": "KB-007",
    "title": "Gift wrap in stores",
    "body": "Stores offer gift wrap. The company was founded in 2014.",
    "source": "kb:KB-007",
}
_EPISODE = {
    "id": "EP-001",
    "ticket_id": "T-SEED-001",
    "category": "billing",
    "subject": "Charged twice for one order",
    "body": "The customer was billed twice and asked for the extra charge to be reversed.",
    "approved_reply": "I confirmed both charges on the order and reversed the extra one.",
}


@pytest.fixture
def store():
    opened = open_store(
        ":memory:",
        fake_embeddings(),
        dims=FAKE_DIMS,
        model=FAKE_MODEL,
    )
    try:
        yield opened
    finally:
        opened.conn.close()


def _keys(opened, namespace: tuple[str, ...]) -> set[str]:
    return {item.key for item in opened.search(namespace, limit=100)}


def _kb(opened) -> StoreKnowledgeBase:
    return StoreKnowledgeBase(opened, settings=Settings())


def test_relative_margin_keeps_hits_near_the_top() -> None:
    scored = [
        (0.50, {"id": "top"}),
        (0.46, {"id": "near"}),
        (0.44, {"id": "far"}),
        (0.19, {"id": "below-min"}),
        (0.90, {"id": "extra-1"}),
        (0.88, {"id": "extra-2"}),
        (0.86, {"id": "extra-3"}),
        (0.84, {"id": "extra-4"}),
    ]
    kept = select_hits(scored[:4], min_score=0.2, relative_margin=0.1, limit=3)
    assert [value["id"] for _, value in kept] == ["top", "near"]
    wide = select_hits(scored[4:], min_score=0.2, relative_margin=0.1, limit=3)
    assert [value["id"] for _, value in wide] == ["extra-1", "extra-2", "extra-3"]


def test_semantic_put_and_get(store) -> None:
    upsert_article(store, _CHARGE)
    got = get_article(store, "KB-001")
    assert got is not None
    assert got["source"] == "kb:KB-001"
    assert got["kind"] == "kb"
    assert "charged twice" in got["text"]


def test_semantic_search_ranks_a_close_fact_first(store) -> None:
    upsert_article(store, _CHARGE)
    upsert_article(store, _GIFT)
    hits = search_articles(store, "charged twice", limit=2, min_score=0.0)
    assert [value["id"] for _, value in hits][0] == "KB-001"
    assert hits[0][0] > hits[1][0]


def test_seed_articles_twice_leaves_seven_rows(store) -> None:
    assert seed_articles(store) == 7
    assert seed_articles(store) == 7
    assert _keys(store, SEMANTIC) == {f"KB-00{index}" for index in range(1, 8)}


def test_search_kb_matches_a_charge_and_skips_a_miss(store) -> None:
    seed_articles(store)
    kb = _kb(store)
    found = kb.search_kb("charged twice")
    ids = [row["id"] for row in found.records]
    assert "KB-001" in ids
    assert "KB-007" not in ids
    assert set(found.records[0]) == {"id", "source", "title", "body"}
    empty = kb.search_kb("zzzznomatch")
    assert empty.found is False
    assert empty.records == []


def test_tracking_subject_returns_kb_002_first(store) -> None:
    seed_articles(store)
    ids = [row["id"] for row in _kb(store).search_kb("Order 90210 has not moved").records]
    assert ids[0] == "KB-002"
    assert "KB-001" not in ids


def test_refund_subject_skips_charge_and_tracking(store) -> None:
    seed_articles(store)
    ids = [row["id"] for row in _kb(store).search_kb("Refund for order 7712").records]
    assert "KB-001" not in ids
    assert "KB-002" not in ids


def test_empty_semantic_namespace_warns(caplog: pytest.LogCaptureFixture) -> None:
    def embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError("embedder should not run when the knowledge base is empty")

    opened = open_store(":memory:", embed, dims=4, model=FAKE_MODEL)
    try:
        kb = StoreKnowledgeBase(opened)
        with caplog.at_level(logging.WARNING, logger="triage_agent.knowledge_base"):
            result = kb.search_kb("charged twice")
    finally:
        opened.conn.close()
    assert result.found is False
    assert result.records == []
    assert EMPTY_KB_WARNING in caplog.text


def test_episode_put_get_and_search_limit(store) -> None:
    upsert_episode(store, _EPISODE)
    got = get_episode(store, "EP-001")
    assert got is not None
    assert got["approved_reply"] == _EPISODE["approved_reply"]
    seed_episodes(store)
    seed_episodes(store)
    assert _keys(store, EPISODIC) == {"EP-001", "EP-002", "EP-003"}
    upsert_episode(
        store,
        {
            "id": "EP-004",
            "ticket_id": "T-SEED-004",
            "category": "billing",
            "subject": "Another duplicate charge",
            "body": "Charged twice again on the same order.",
            "approved_reply": "Reversed the extra charge.",
        },
    )
    hits = search_episodes(store, "charged twice", limit=3, min_score=0.0)
    assert len(hits) == 3
    assert hits[0][1]["id"] == "EP-001"
    assert hits[0][1]["approved_reply"]


def test_procedure_seed_is_version_one_and_stays_there(store) -> None:
    seed_procedures(store)
    seed_procedures(store)
    current = get_current(store)
    assert current is not None
    assert current["version"] == 1
    assert current["note"] == "seed"
    assert "Never promise a delivery date" in current["text"]
    assert [row["version"] for row in list_versions(store)] == [1]
    assert _keys(store, PROCEDURAL) == {"v1", "current"}


def test_procedure_seed_updates_v1_text_in_place(store, tmp_path) -> None:
    path = tmp_path / "rules.md"
    path.write_text("Be brief.", encoding="utf-8")
    seed_procedures(store, path)
    path.write_text("Be specific.", encoding="utf-8")
    seed_procedures(store, path)
    current = get_current(store)
    assert current is not None
    assert current["version"] == 1
    assert current["text"] == "Be specific."
    assert len(list_versions(store)) == 1


def test_put_version_and_rollback(store) -> None:
    seed_procedures(store)
    original = get_current(store)
    assert original is not None
    updated = put_version(store, "Never promise a delivery date.", "reviewer")
    assert updated["version"] == 2
    assert get_current(store)["version"] == 2
    assert get_current(store)["text"] == "Never promise a delivery date."
    assert any(row["version"] == 1 and row["text"] == original["text"] for row in list_versions(store))
    restored = rollback(store, 1)
    assert restored["text"] == original["text"]
    assert get_current(store)["version"] == 1
    assert any(row["version"] == 2 for row in list_versions(store))


def test_rollback_of_a_missing_version_raises(store) -> None:
    seed_procedures(store)
    with pytest.raises(LookupError):
        rollback(store, 9)


def test_seed_does_not_overwrite_a_later_procedure(store) -> None:
    seed_procedures(store)
    put_version(store, "A later rule.", "reviewer")
    seed_procedures(store)
    assert get_current(store)["version"] == 2
    assert get_current(store)["text"] == "A later rule."
    assert [row["version"] for row in list_versions(store)] == [2, 1]


def test_episode_persists_across_connections(tmp_path) -> None:
    path = str(tmp_path / "memory.sqlite")
    embeddings = fake_embeddings()
    opened = open_store(path, embeddings, dims=FAKE_DIMS, model=FAKE_MODEL)
    upsert_episode(opened, _EPISODE)
    opened.conn.close()
    reopened = open_store(path, embeddings, dims=FAKE_DIMS, model=FAKE_MODEL)
    try:
        got = get_episode(reopened, "EP-001")
    finally:
        reopened.conn.close()
    assert got is not None
    assert got["approved_reply"] == _EPISODE["approved_reply"]


@pytest.mark.live
def test_live_search_kb_paraphrase_and_neighbors() -> None:
    get_settings.cache_clear()
    get_settings()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        pytest.skip("OPENAI_API_KEY is not set")
    os.environ["OPENAI_API_KEY"] = key
    embeddings = init_embeddings("text-embedding-3-small", provider="openai")
    opened = open_store(
        ":memory:",
        embeddings,
        dims=1536,
        model="text-embedding-3-small",
    )
    try:
        seed_articles(opened)
        kb = StoreKnowledgeBase(opened, settings=Settings())
        samples = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "sample_tickets.json").read_text(
                encoding="utf-8"
            )
        )
        ticket = next(row for row in samples if row["id"] == "T-005")
        refund_query = kb_query(ticket["subject"], ticket["body"])
        queries = (
            "my package hasn't budged in a week",
            "Order 90210 has not moved",
            refund_query,
            "charged twice",
        )
        for query in queries:
            hits = opened.search(SEMANTIC, query=query, limit=10)
            print(f"query: {query}")
            for hit in hits:
                score = float(hit.score) if hit.score is not None else float("nan")
                print(f"  {hit.key} {score:.4f}")
        article = get_article(opened, "KB-002")
        assert article is not None
        paraphrase = queries[0]
        assert content_tokens(paraphrase).isdisjoint(
            content_tokens(f"{article['title']} {article['body']}")
        )
        assert kb.search_kb(paraphrase).records[0]["id"] == "KB-002"
        moved = [row["id"] for row in kb.search_kb(queries[1]).records]
        assert moved[0] == "KB-002"
        refund = [row["id"] for row in kb.search_kb(queries[2]).records]
        assert refund[0] == "KB-003"
        assert "KB-002" not in refund
        charged = [row["id"] for row in kb.search_kb(queries[3]).records]
        assert "KB-007" not in charged
    finally:
        opened.conn.close()
