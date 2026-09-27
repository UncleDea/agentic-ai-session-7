"""Semantic, episodic, and procedural memory loaded from local files."""

from __future__ import annotations

import json
from pathlib import Path

from investigator.schemas import Investigation

from responder.config import _EPISODIC_ANCHORS, _SERVICE_DOMAIN


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def domain_for(investigation: Investigation) -> str:
    if investigation.runbook_path:
        prefix = investigation.runbook_path.split("/", 1)[0]
        if prefix in {"support", "it-helpdesk", "oncall"}:
            return prefix
    if investigation.service in _SERVICE_DOMAIN:
        return _SERVICE_DOMAIN[investigation.service]
    return "support"


def gather_memory(
    investigation: Investigation,
    raw_text: str,
    customer_id: str | None,
    data_dir: Path,
) -> dict[str, str]:
    """Return the memory sections that should be shown to the draft model."""
    domain = domain_for(investigation)
    sections: dict[str, str] = {}

    semantic = _semantic(domain, data_dir)
    if semantic:
        sections["semantic"] = semantic

    episodic = _episodic(raw_text, customer_id, data_dir)
    if episodic:
        sections["episodic"] = episodic

    procedural = _procedural(investigation.runbook_path, data_dir)
    if procedural:
        sections["procedural"] = procedural

    return sections


def _semantic(domain: str, data_dir: Path) -> str:
    prompt = _latest_prompt(domain, data_dir)
    taxonomy = _taxonomy(domain, data_dir)
    parts = [part for part in (prompt, taxonomy) if part]
    return "\n".join(parts)


def _latest_prompt(domain: str, data_dir: Path) -> str:
    path = data_dir / "prompts" / f"responder_{domain}.json"
    if not path.is_file():
        return ""
    history = json.loads(path.read_text(encoding="utf-8"))["history"]
    latest = max(history, key=lambda item: int(item["version"]))
    return str(latest["prompt"])


def _taxonomy(domain: str, data_dir: Path) -> str:
    path = data_dir / domain / "taxonomy.yaml"
    if not path.is_file():
        return ""
    lines: list[str] = []
    name = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("- name:"):
            name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("description:") and name:
            description = stripped.split(":", 1)[1].strip()
            lines.append(f"{name}: {description}")
            name = ""
    if not lines:
        return ""
    return "Categories:\n" + "\n".join(lines)


def _episodic(raw_text: str, customer_id: str | None, data_dir: Path) -> str:
    path = data_dir / "sent_responses.jsonl"
    if not path.is_file():
        return ""
    customer = (customer_id or "").casefold()
    text = raw_text.casefold()
    anchors = [anchor for anchor in _EPISODIC_ANCHORS if anchor in text]
    recipient_hits: list[str] = []
    anchor_hits: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        body = str(row.get("body", "")).strip()
        if not body or body in seen:
            continue
        seen.add(body)
        recipient = str(row.get("recipient", "")).casefold()
        if customer and recipient == customer:
            recipient_hits.append(body)
        elif anchors and any(anchor in body.casefold() for anchor in anchors):
            anchor_hits.append(body)
    chosen = (recipient_hits or anchor_hits)[:1]
    if not chosen:
        return ""
    excerpt = chosen[0]
    if len(excerpt) > 700:
        excerpt = excerpt[:700].rstrip() + "..."
    return "Previous reply:\n" + excerpt


def _procedural(runbook_path: str | None, data_dir: Path) -> str:
    if not runbook_path:
        return ""
    path = data_dir / runbook_path
    if not path.is_file():
        return ""
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("<!--"):
            continue
        kept.append(line)
    text = "\n".join(kept).strip()
    if len(text) > 2000:
        text = text[:2000].rstrip() + "..."
    return text
