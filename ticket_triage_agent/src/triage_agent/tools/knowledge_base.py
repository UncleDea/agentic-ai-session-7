"""Knowledge-base search over the semantic memory store."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from triage_agent.config import Settings
from triage_agent.memory.semantic import public_article, search_articles
from triage_agent.memory.store import SEMANTIC, store_for_settings
from triage_agent.tools.ticket_db import LookupResult

logger = logging.getLogger("triage_agent.knowledge_base")

EMPTY_KB_WARNING = "knowledge base is empty, run scripts/seed_memory.py"

_WORD = re.compile(r"[a-z0-9]+")
_TITLE_WEIGHT = 3
_BODY_WEIGHT = 1
_MIN_SCORE = 2
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "it",
        "its",
        "this",
        "that",
        "my",
        "me",
        "i",
        "we",
        "you",
        "your",
        "our",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "so",
        "than",
        "then",
        "too",
        "very",
        "can",
        "will",
        "just",
        "into",
        "over",
        "after",
        "before",
        "about",
        "up",
        "out",
        "off",
        "again",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "when",
        "where",
        "who",
        "what",
        "which",
        "how",
        "order",
    }
)


class KnowledgeBase(Protocol):
    def search_kb(self, query: str) -> LookupResult: ...


def content_tokens(text: str) -> set[str]:
    return {
        word
        for word in _WORD.findall(text.lower())
        if word not in _STOPWORDS
    }


def rank_keywords(query: str, rows: list[dict], title_of, body_of, limit: int = 3) -> list[dict]:
    """Rank rows by keyword overlap. Title hits outweigh body hits.

    Past-ticket subjects are titles. A hit must clear the minimum score:
    one title word does, one body word does not.
    """
    query_tokens = content_tokens(query)
    if not query_tokens:
        return []
    scored: list[tuple[int, dict]] = []
    for row in rows:
        title_tokens = content_tokens(title_of(row))
        body_tokens = content_tokens(body_of(row))
        score = 0
        for token in query_tokens:
            if token in title_tokens:
                score += _TITLE_WEIGHT
            elif token in body_tokens:
                score += _BODY_WEIGHT
        if score >= _MIN_SCORE:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:limit]]


class StoreKnowledgeBase:
    def __init__(self, store, settings: Settings | None = None) -> None:
        self._store = store
        self._settings = settings if settings is not None else Settings()

    def search_kb(self, query: str) -> LookupResult:
        if not self._store.search(SEMANTIC, limit=1):
            logger.warning(EMPTY_KB_WARNING)
            return LookupResult(
                found=False,
                records=[],
                note="knowledge base is empty",
            )
        ranked = search_articles(
            self._store,
            query,
            limit=self._settings.memory_search_limit,
            min_score=self._settings.kb_min_score,
            relative_margin=self._settings.kb_relative_margin,
        )
        if not ranked:
            return LookupResult(
                found=False,
                records=[],
                note="no matching knowledge base articles",
            )
        return LookupResult(
            found=True,
            records=[public_article(value) for _, value in ranked],
        )


def knowledge_base_for_settings(settings: Settings | None = None) -> StoreKnowledgeBase:
    active = settings if settings is not None else Settings()
    return StoreKnowledgeBase(store_for_settings(active), settings=active)
