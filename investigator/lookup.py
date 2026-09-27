"""Search the local data directory. No network and no model calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from triage_agent.schemas import TicketState

from investigator.schemas import Evidence, Investigation

_SERVICE_TERMS: dict[str, tuple[str, ...]] = {
    "auth-service": ("mfa", "authenticator", "login"),
    "billing-service": ("charge", "refund", "invoice", "billing"),
    "reports-service": ("export", "report"),
    "vpn-gateway": ("vpn",),
    "identity-service": ("password",),
    "checkout-service": ("checkout", "503"),
    "api-gateway": ("503", "gateway"),
}

_RUNBOOK_ANCHORS: dict[str, tuple[str, ...]] = {
    "mfa-loop.md": ("mfa", "authenticator"),
    "vpn-disconnect.md": ("vpn",),
    "checkout-503.md": ("checkout", "503"),
}

_MAX_LOGS = 5
_MAX_METRICS = 4


@dataclass
class _Ticket:
    domain: str
    ticket_id: str
    user_id: str
    subject: str
    body: str
    received_at: str


@dataclass
class _Log:
    domain: str
    service: str
    level: str
    message: str


@dataclass
class _Metric:
    domain: str
    service: str
    name: str
    current: object
    avg: object
    trend: str


@dataclass
class _Runbook:
    domain: str
    filename: str
    relative_path: str
    title: str


@dataclass
class Catalog:
    tickets: list[_Ticket] = field(default_factory=list)
    logs: list[_Log] = field(default_factory=list)
    metrics: list[_Metric] = field(default_factory=list)
    runbooks: list[_Runbook] = field(default_factory=list)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def load_catalog(data_dir: Path) -> Catalog:
    root = data_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Investigation data directory does not exist: {root}")

    catalog = Catalog()
    for domain_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        _load_domain(catalog, domain_dir)
    return catalog


def investigate(ticket: TicketState, catalog: Catalog) -> Investigation:
    text = ticket.raw_text.casefold()
    customer = (ticket.customer_id or "").casefold()
    keyword_hits = _keyword_hits(text)
    user_logs = _logs_for_customer(catalog, customer) if customer else []
    related = _related_tickets(catalog, customer, ticket.ticket_id)
    runbooks = _matching_runbooks(catalog, text)
    service = _pick_service(keyword_hits, user_logs)

    evidence: list[Evidence] = []
    evidence.extend(_ticket_evidence(related))
    evidence.extend(_log_evidence(catalog, user_logs, service))
    evidence.extend(_metric_evidence(catalog, service))
    evidence.extend(_runbook_evidence(runbooks))

    runbook_path = runbooks[0].relative_path if runbooks else None
    return Investigation(
        status="investigated",
        summary=_summary(service, related, runbook_path, evidence),
        service=service,
        runbook_path=runbook_path,
        related_ticket_ids=[item.ticket_id for item in related],
        evidence=evidence,
    )


def _load_domain(catalog: Catalog, domain_dir: Path) -> None:
    domain = domain_dir.name
    tickets_path = domain_dir / "historical_tickets.jsonl"
    if tickets_path.is_file():
        for line in tickets_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            catalog.tickets.append(
                _Ticket(
                    domain=domain,
                    ticket_id=str(row["id"]),
                    user_id=str(row.get("user_id", "")),
                    subject=str(row.get("subject", "")),
                    body=str(row.get("body", "")),
                    received_at=str(row.get("received_at", "")),
                )
            )

    logs_path = domain_dir / "mock_logs.json"
    if logs_path.is_file():
        payload = json.loads(logs_path.read_text(encoding="utf-8"))
        for service, entries in payload.items():
            for entry in entries:
                catalog.logs.append(
                    _Log(
                        domain=domain,
                        service=str(service),
                        level=str(entry.get("level", "")),
                        message=str(entry.get("message", "")),
                    )
                )

    metrics_path = domain_dir / "mock_metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        for service, metrics in payload.items():
            for name, metric in metrics.items():
                catalog.metrics.append(
                    _Metric(
                        domain=domain,
                        service=str(service),
                        name=str(name),
                        current=metric.get("current"),
                        avg=metric.get("avg"),
                        trend=str(metric.get("trend", "")),
                    )
                )

    runbook_dir = domain_dir / "runbooks"
    if runbook_dir.is_dir():
        for path in sorted(runbook_dir.glob("*.md")):
            title = _heading(path.read_text(encoding="utf-8"))
            catalog.runbooks.append(
                _Runbook(
                    domain=domain,
                    filename=path.name,
                    relative_path=f"{domain}/runbooks/{path.name}",
                    title=title,
                )
            )


def _heading(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _keyword_hits(text: str) -> dict[str, int]:
    hits: dict[str, int] = {}
    for service, terms in _SERVICE_TERMS.items():
        count = sum(1 for term in terms if term in text)
        if count:
            hits[service] = count
    return hits


def _logs_for_customer(catalog: Catalog, customer: str) -> list[_Log]:
    needle = f"user={customer}"
    return [log for log in catalog.logs if needle in log.message.casefold()]


def _related_tickets(catalog: Catalog, customer: str, current_id: str) -> list[_Ticket]:
    if not customer:
        return []
    matched = [
        ticket
        for ticket in catalog.tickets
        if ticket.user_id.casefold() == customer and ticket.ticket_id != current_id
    ]
    return sorted(matched, key=lambda ticket: ticket.received_at, reverse=True)


def _matching_runbooks(catalog: Catalog, text: str) -> list[_Runbook]:
    matched = []
    for runbook in catalog.runbooks:
        anchors = _RUNBOOK_ANCHORS.get(runbook.filename, ())
        if any(anchor in text for anchor in anchors):
            matched.append(runbook)
    return matched


def _pick_service(keyword_hits: dict[str, int], user_logs: list[_Log]) -> str | None:
    log_counts: dict[str, int] = {}
    for log in user_logs:
        log_counts[log.service] = log_counts.get(log.service, 0) + 1
    candidates = set(keyword_hits) | set(log_counts)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda name: (log_counts.get(name, 0), keyword_hits.get(name, 0), name),
    )


def _ticket_evidence(tickets: list[_Ticket]) -> list[Evidence]:
    return [
        Evidence(
            source="ticket",
            ref=ticket.ticket_id,
            excerpt=f"{ticket.subject}: {ticket.body}",
        )
        for ticket in tickets
    ]


def _log_evidence(
    catalog: Catalog,
    user_logs: list[_Log],
    service: str | None,
) -> list[Evidence]:
    if user_logs:
        chosen = [log for log in user_logs if log.service == service][:_MAX_LOGS] or user_logs[:_MAX_LOGS]
    elif service:
        chosen = [
            log
            for log in catalog.logs
            if log.service == service
            and log.level in {"WARN", "ERROR"}
            and "user=" not in log.message
        ][:_MAX_LOGS]
    else:
        chosen = []
    return [
        Evidence(
            source="log",
            ref=f"{log.domain}/{log.service}",
            excerpt=f"{log.level} {log.message}",
        )
        for log in chosen
    ]


def _metric_evidence(catalog: Catalog, service: str | None) -> list[Evidence]:
    if service is None:
        return []
    evidence = []
    for metric in catalog.metrics:
        if metric.service != service:
            continue
        if not _metric_is_notable(metric):
            continue
        evidence.append(
            Evidence(
                source="metric",
                ref=f"{metric.domain}/{metric.service}/{metric.name}",
                excerpt=(
                    f"{metric.service} {metric.name} "
                    f"current={metric.current} avg={metric.avg} trend={metric.trend}"
                ),
            )
        )
        if len(evidence) >= _MAX_METRICS:
            break
    return evidence


def _metric_is_notable(metric: _Metric) -> bool:
    if metric.trend == "up":
        return True
    try:
        return float(metric.current) > float(metric.avg)
    except (TypeError, ValueError):
        return False


def _runbook_evidence(runbooks: list[_Runbook]) -> list[Evidence]:
    return [
        Evidence(source="runbook", ref=runbook.relative_path, excerpt=runbook.title)
        for runbook in runbooks
    ]


def _summary(
    service: str | None,
    related: list[_Ticket],
    runbook_path: str | None,
    evidence: list[Evidence],
) -> str:
    if not evidence:
        return "No local evidence matched this ticket."
    parts = [f"Matched {service}." if service else "Matched local evidence."]
    if related:
        ids = ", ".join(ticket.ticket_id for ticket in related)
        parts.append(f"{len(related)} related ticket(s): {ids}.")
    if runbook_path:
        parts.append(f"Runbook {runbook_path}.")
    return " ".join(parts)
