"""Responder. Drafts from memory, then checks the reply against the facts."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.sqlite import SqliteStore

from triage_agent.config import Settings, get_settings
from triage_agent.llm import get_chat_model
from triage_agent.memory.episodic import search_episodes
from triage_agent.memory.procedural import get_current
from triage_agent.memory.semantic import get_article, search_articles
from triage_agent.memory.store import store_for_settings
from triage_agent.nodes.investigator import kb_query
from triage_agent.state import (
    Draft,
    Finding,
    Investigation,
    Review,
    ReviewStatus,
    Ticket,
    TicketState,
    TriageResult,
)
from triage_agent.tools.knowledge_base import content_tokens

logger = logging.getLogger("triage_agent.responder")

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "responder.md"

_AMOUNT = re.compile(r"\$\s?\d+(?:\.\d{1,2})?|\b\d+\.\d{2}\b")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONTH_DATE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
_ES_MONTH = (
    r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|"
    r"octubre|noviembre|diciembre"
)
_ES_DATE = re.compile(
    rf"\b(?:{_ES_MONTH})\s+de\s+\d{{1,2}}(?:\s+de\s+\d{{4}})?\b"
    rf"|\b\d{{1,2}}\s+de\s+(?:{_ES_MONTH})(?:\s+de\s+\d{{4}})?\b",
    re.IGNORECASE,
)
_ORDER = re.compile(r"\b(?:order|pedido|orden)\s+#?(\d{3,})\b", re.IGNORECASE)
_BARE_ORDER = re.compile(r"\b\d{4,}\b")
_DURATION = re.compile(
    r"\b(?:within\s+)?\d+\s+(?:business\s+days?|business\s+hours?|days?|hours?|weeks?)\b",
    re.IGNORECASE,
)
_BY_WHEN = re.compile(
    r"\bby\s+(?:tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|mañana|lunes|martes|miércoles|miercoles|jueves|viernes|"
    r"sábado|sabado|domingo)\b",
    re.IGNORECASE,
)
_ES_DURATION = re.compile(
    r"\b(?:dentro\s+de\s+)?\d+\s+(?:d[ií]as?\s+h[aá]biles|d[ií]as?|horas?|semanas?)\b",
    re.IGNORECASE,
)
_PROMISES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refund", re.compile(r"\b(?:refund(?:ed|ing)?|reembols(?:o|ar|ado|a))\b", re.I)),
    ("credit", re.compile(r"\b(?:credit(?:ed|ing)?|cr[eé]dito)\b", re.I)),
    (
        "replacement",
        re.compile(r"\b(?:replace(?:ment|d)?|reemplaz(?:o|ar|ado))\b", re.I),
    ),
    (
        "reverse",
        re.compile(r"\b(?:revers(?:e|ed|ing|al)|revertir|reversar)\b", re.I),
    ),
    ("delivery date", re.compile(r"\b(?:delivery date|fecha de entrega)\b", re.I)),
)
_NEGATION = re.compile(
    r"(?:before promising|do not|cannot|can't|don't|no puedo|no prometas|"
    r"\bnunca\b|\bnever\b|\bwithout\b|\bnot\b|\bsin\b|\bno\b)",
    re.IGNORECASE,
)
_SPAN_PATTERNS = (
    _AMOUNT,
    _DURATION,
    _BY_WHEN,
    _ES_DURATION,
    _ORDER,
)
_COMPLETED = re.compile(
    r"(?:"
    r"\b(?:i|we|we['’]ve)\s+(?:have\s+)?(?:refunded|reversed|cancell?ed)\b"
    r"|\b(?:has|have)\s+been\s+(?:refunded|reversed|cancell?ed)\b"
    r"|\breembols(?:é|e)\b"
    r"|\b(?:he|hemos|ha sido|fue|se ha|ya se)\s+reembolsado\b"
    r"|\brevert(?:í|i)\b"
    r"|\b(?:he|hemos|ha sido|fue)\s+revertido\b"
    r"|\bcancel(?:é|e)\b"
    r"|\b(?:he|hemos|ha sido|fue|se ha)\s+cancelado\b"
    r")",
    re.IGNORECASE,
)
_ESCALATION = re.compile(
    r"(?:"
    r"\bescalat(?:e|ing|ion)\b"
    r"|\bescalar(?:é|e|á|a)?\b"
    r"|will follow up"
    r"|har[aá] seguimiento"
    r"|\bespecialista\b"
    r")",
    re.IGNORECASE,
)
_EN_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_ES_MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
_Date = tuple[int | None, int, int]


def load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log(ticket_id: str | None, decision: str, started: float) -> None:
    logger.info(
        "node=responder ticket_id=%s decision=%s latency_ms=%d",
        ticket_id,
        decision,
        _latency_ms(started),
    )


def _applies(article: dict[str, Any], ticket: Ticket, findings: list[Finding]) -> bool:
    haystack = " ".join(
        [
            ticket.subject,
            ticket.body,
            *(finding.fact for finding in findings),
            *(finding.source for finding in findings),
        ]
    )
    article_text = f"{article.get('title', '')}\n{article.get('body', '')}"
    return bool(content_tokens(haystack) & content_tokens(article_text))


def _search_articles(
    store: SqliteStore,
    ticket: Ticket,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Search the full ticket, then the subject if that query is too dilute."""
    queries = [kb_query(ticket.subject, ticket.body)]
    subject = kb_query(ticket.subject, "")
    if subject and subject not in queries:
        queries.append(subject)
    for query in queries:
        if not query:
            continue
        hits = search_articles(
            store,
            query,
            limit=settings.memory_search_limit,
            min_score=settings.kb_min_score,
            relative_margin=settings.kb_relative_margin,
        )
        if hits:
            return [value for _, value in hits]
    return []


def select_articles(
    store: SqliteStore,
    ticket: Ticket,
    investigation: Investigation | None,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Investigation articles, or a search when that list is empty.

    An article with no token in common with the ticket or the findings is dropped.
    """
    findings = list(investigation.findings) if investigation is not None else []
    ids = list(investigation.kb_article_ids) if investigation is not None else []
    rows: list[dict[str, Any]] = []
    if ids:
        for article_id in ids:
            value = get_article(store, article_id)
            if value is not None:
                rows.append(value)
    else:
        rows = _search_articles(store, ticket, settings)
    return [row for row in rows if _applies(row, ticket, findings)]


def select_episodes(
    store: SqliteStore,
    ticket: Ticket,
    triage: TriageResult | None,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Top similar episodes. Prefer the ticket category when any hit shares it."""
    query = f"{ticket.subject}\n{ticket.body}"
    hits = search_episodes(
        store,
        query,
        limit=settings.memory_search_limit,
        min_score=settings.kb_min_score,
    )
    episodes = [value for _, value in hits]
    if triage is None:
        return episodes
    same = [row for row in episodes if row.get("category") == triage.category.value]
    if same:
        return same[: settings.memory_search_limit]
    return episodes


def build_messages(
    ticket: Ticket,
    triage: TriageResult | None,
    investigation: Investigation | None,
    procedure: dict[str, Any] | None,
    episodes: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    review: Review | None,
    previous: Draft | None,
) -> list:
    language = triage.language if triage is not None else "en"
    category = triage.category.value if triage is not None else "unknown"
    system = load_prompt()
    if procedure is not None:
        system += "\n\nCurrent response rules:\n\n" + str(procedure["text"])
    system += (
        f"\n\nThe ticket language is {language}. "
        "Write the entire reply in that language."
    )
    findings = list(investigation.findings) if investigation is not None else []
    missing = list(investigation.missing_info) if investigation is not None else []
    finding_lines = [
        f"- {finding.fact} [{finding.source}]" for finding in findings
    ] or ["- (none)"]
    article_lines = [
        f"{article['id']}: {article['title']}\n{article['body']}" for article in articles
    ] or ["(none)"]
    episode_lines = [
        "\n".join(
            [
                f"{episode['id']} ({episode.get('category', '')})",
                f"Subject: {episode.get('subject', '')}",
                f"Approved reply: {episode.get('approved_reply', '')}",
            ]
        )
        for episode in episodes
    ] or ["(none)"]
    human = "\n".join(
        [
            f"id: {ticket.id}",
            f"customer_id: {ticket.customer_id}",
            f"channel: {ticket.channel.value}",
            f"category: {category}",
            f"language: {language}",
            f"subject: {ticket.subject}",
            f"body: {ticket.body}",
            "",
            "Findings:",
            *finding_lines,
            "",
            "Missing info:",
            *(missing or ["(none)"]),
            "",
            "Knowledge-base articles:",
            *article_lines,
            "",
            "Example replies (style only, not facts):",
            *episode_lines,
        ]
    )
    if (
        review is not None
        and review.status is ReviewStatus.rejected
        and previous is not None
    ):
        human += (
            "\n\nPrevious draft:\n"
            f"{previous.reply}\n\n"
            "Reviewer feedback:\n"
            f"{review.reviewer_feedback}\n\n"
            "Revise the previous draft using the feedback. "
            "Do not add facts that are not in the findings or knowledge-base articles."
        )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def _overlaps(start: int, end: int, claimed: list[tuple[int, int]]) -> bool:
    return any(start < right and end > left for left, right in claimed)


def _negated(text: str, start: int) -> bool:
    window = text[max(0, start - 40) : start]
    return _NEGATION.search(window) is not None


def _affirmative_spans(text: str, pattern: re.Pattern[str]) -> list[re.Match[str]]:
    return [match for match in pattern.finditer(text) if not _negated(text, match.start())]


def _calendar(year: int | None, month: int, day: int) -> _Date | None:
    if month < 1 or month > 12 or day < 1:
        return None
    if month == 2 and day > 29:
        return None
    if month in (4, 6, 9, 11) and day > 30:
        return None
    if day > 31:
        return None
    if year is not None:
        try:
            datetime(year, month, day)
        except ValueError:
            return None
    return (year, month, day)


def _parse_english_date(snippet: str) -> _Date | None:
    match = re.match(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:,\s*(?P<year>\d{4}))?$",
        snippet,
    )
    if match is None:
        return None
    month = _EN_MONTHS.get(match.group("month").lower())
    if month is None:
        return None
    year = int(match.group("year")) if match.group("year") else None
    return _calendar(year, month, int(match.group("day")))


def _parse_spanish_date(snippet: str) -> _Date | None:
    day_first = re.match(
        r"(?P<day>\d{1,2})\s+de\s+(?P<month>[A-Za-záéíóúñ]+)(?:\s+de\s+(?P<year>\d{4}))?$",
        snippet,
        re.IGNORECASE,
    )
    month_first = re.match(
        r"(?P<month>[A-Za-záéíóúñ]+)\s+de\s+(?P<day>\d{1,2})(?:\s+de\s+(?P<year>\d{4}))?$",
        snippet,
        re.IGNORECASE,
    )
    match = day_first or month_first
    if match is None:
        return None
    month = _ES_MONTHS.get(match.group("month").lower())
    if month is None:
        return None
    year = int(match.group("year")) if match.group("year") else None
    return _calendar(year, month, int(match.group("day")))


def _same_day(left: _Date, right: _Date) -> bool:
    left_year, left_month, left_day = left
    right_year, right_month, right_day = right
    if left_month != right_month or left_day != right_day:
        return False
    if left_year is not None and right_year is not None and left_year != right_year:
        return False
    return True


def _dates(text: str) -> list[tuple[int, int, str, _Date]]:
    found: list[tuple[int, int, str, _Date]] = []
    for match in _ISO_DATE.finditer(text):
        year_s, month_s, day_s = match.group(0).split("-")
        parsed = _calendar(int(year_s), int(month_s), int(day_s))
        if parsed is not None:
            found.append((match.start(), match.end(), match.group(0), parsed))
    for match in _MONTH_DATE.finditer(text):
        parsed = _parse_english_date(match.group(0))
        if parsed is not None:
            found.append((match.start(), match.end(), match.group(0), parsed))
    for match in _ES_DATE.finditer(text):
        parsed = _parse_spanish_date(match.group(0))
        if parsed is not None:
            found.append((match.start(), match.end(), match.group(0), parsed))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str, _Date]] = []
    for start, end, snippet, parsed in found:
        if _overlaps(start, end, [(item[0], item[1]) for item in kept]):
            continue
        kept.append((start, end, snippet, parsed))
    return kept


def promises_escalation(reply: str) -> bool:
    """True when the reply says a person will take the ticket from here."""
    return _ESCALATION.search(reply) is not None


def fact_check(reply: str, corpus: str) -> list[str]:
    """Snippets in the reply that the findings and used articles do not support.

    Dates match by calendar day, so March 2, 2026-03-02, and 2 de marzo are
    the same day. A completed refund, reversal, or cancellation is flagged
    even when an article allows that action.
    """
    corpus_l = corpus.lower()
    flagged: list[tuple[int, str]] = []
    claimed: list[tuple[int, int]] = []
    corpus_dates = [item[3] for item in _dates(corpus)]

    def take(match: re.Match[str], needle: str | None = None) -> None:
        if _overlaps(match.start(), match.end(), claimed):
            return
        claimed.append((match.start(), match.end()))
        token = (needle if needle is not None else match.group(0)).lower()
        if token not in corpus_l:
            flagged.append((match.start(), match.group(0)))

    for start, end, snippet, parsed in _dates(reply):
        if _overlaps(start, end, claimed):
            continue
        claimed.append((start, end))
        if not any(_same_day(parsed, known) for known in corpus_dates):
            flagged.append((start, snippet))

    for pattern in _SPAN_PATTERNS:
        for match in pattern.finditer(reply):
            if pattern is _AMOUNT:
                number = match.group(0).lstrip("$").strip().lower()
                if match.group(0).lower() in corpus_l or number in corpus_l:
                    if not _overlaps(match.start(), match.end(), claimed):
                        claimed.append((match.start(), match.end()))
                    continue
            needle = match.group(1) if pattern is _ORDER else None
            take(match, needle)
    for match in _BARE_ORDER.finditer(reply):
        take(match)

    for match in _COMPLETED.finditer(reply):
        if _negated(reply, match.start()) or _overlaps(match.start(), match.end(), claimed):
            continue
        flagged.append((match.start(), match.group(0)))
        claimed.append((match.start(), match.end()))

    for _, pattern in _PROMISES:
        reply_hits = _affirmative_spans(reply, pattern)
        if reply_hits and not _affirmative_spans(corpus, pattern):
            match = reply_hits[0]
            if not _overlaps(match.start(), match.end(), claimed):
                flagged.append((match.start(), match.group(0)))
                claimed.append((match.start(), match.end()))

    flagged.sort(key=lambda item: item[0])
    unique: list[str] = []
    for _, snippet in flagged:
        if snippet not in unique:
            unique.append(snippet)
    return unique


def _corpus(findings: list[Finding], articles: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for finding in findings:
        parts.append(finding.fact)
        parts.append(finding.source)
    for article in articles:
        parts.append(str(article.get("title", "")))
        parts.append(str(article.get("body", "")))
    return "\n".join(parts)


def _memory_ids(
    procedure: dict[str, Any] | None,
    episodes: list[dict[str, Any]],
    articles: list[dict[str, Any]],
) -> list[str]:
    ids: list[str] = []
    if procedure is not None:
        ids.append(f"v{int(procedure['version'])}")
    ids.extend(str(episode["id"]) for episode in episodes)
    ids.extend(str(article["id"]) for article in articles)
    return ids


def responder(
    state: TicketState,
    model: Any | None = None,
    settings: Settings | None = None,
    store: SqliteStore | None = None,
) -> dict:
    """Draft state['ticket']. On a rejected review, clear it and count the revision."""
    started = time.perf_counter()
    ticket = state.get("ticket")
    if not isinstance(ticket, Ticket):
        _log(None, "error", started)
        return {"errors": ["responder: state has no ticket"]}
    review = state.get("review")
    revising = isinstance(review, Review) and review.status is ReviewStatus.rejected
    try:
        active = settings if settings is not None else get_settings()
        chat = model if model is not None else get_chat_model()
        memory = store if store is not None else store_for_settings(active)
        triage = state.get("triage")
        investigation = state.get("investigation")
        procedure = get_current(memory)
        episodes = select_episodes(memory, ticket, triage, active)
        articles = select_articles(memory, ticket, investigation, active)
        previous = state.get("draft") if revising else None
        messages = build_messages(
            ticket,
            triage,
            investigation,
            procedure,
            episodes,
            articles,
            review if revising else None,
            previous if isinstance(previous, Draft) else None,
        )
        structured = chat.with_structured_output(Draft)
        raw = structured.invoke(messages)
        parsed = raw if isinstance(raw, Draft) else Draft.model_validate(raw)
        findings = list(investigation.findings) if investigation is not None else []
        flagged = fact_check(parsed.reply, _corpus(findings, articles))
        escalates = promises_escalation(parsed.reply)
        confidence = parsed.confidence
        if flagged:
            confidence = min(confidence, max(0.0, active.confidence_threshold - 0.01))
        draft = Draft(
            reply=parsed.reply,
            memory_ids=_memory_ids(procedure, episodes, articles),
            flagged_claims=flagged,
            escalates=escalates,
            confidence=confidence,
        )
        for claim in flagged:
            logger.info(
                "node=responder ticket_id=%s flagged_claim=%s",
                ticket.id,
                claim,
            )
        kind = "revision" if revising else "draft"
        _log(ticket.id, f"{kind} flagged={len(flagged)}", started)
        update: dict = {"draft": draft}
        if revising:
            update["review"] = None
            update["revision_count"] = state.get("revision_count", 0) + 1
        return update
    except Exception as exc:
        _log(ticket.id, "error", started)
        return {"errors": [f"responder: {type(exc).__name__}: {exc}"]}
