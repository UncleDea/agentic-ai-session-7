"""Knowledge-base search. Keyword match now; an embedding store can replace it later."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from triage_agent.tools.ticket_db import LookupResult

_DATA = Path(__file__).resolve().parents[3] / "data"
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


def _tokens(text: str) -> set[str]:
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
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    scored: list[tuple[int, dict]] = []
    for row in rows:
        title_tokens = _tokens(title_of(row))
        body_tokens = _tokens(body_of(row))
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


class JsonKnowledgeBase:
    def __init__(self, path: Path | None = None) -> None:
        kb_path = path or _DATA / "kb_articles.json"
        self._rows = json.loads(kb_path.read_text(encoding="utf-8"))

    def search_kb(self, query: str) -> LookupResult:
        matches = rank_keywords(
            query,
            self._rows,
            lambda row: str(row.get("title", "")),
            lambda row: str(row.get("body", "")),
        )
        if not matches:
            return LookupResult(
                found=False,
                records=[],
                note="no matching knowledge base articles",
            )
        return LookupResult(found=True, records=matches)
