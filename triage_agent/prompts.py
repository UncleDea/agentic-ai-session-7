"""Classification prompts. Kept separate from agent control flow."""

from langchain_core.messages import HumanMessage, SystemMessage

from triage_agent.config import CATEGORIES, CONFIDENCE_THRESHOLD, SEVERITIES

SYSTEM_PROMPT = f"""You are the Triage Agent in a customer support pipeline.
Classify the ticket from its text alone. Do not look anything up.

Pick exactly one category: {", ".join(CATEGORIES)}.
Pick exactly one severity: {", ".join(SEVERITIES)}.

When a ticket mentions more than one issue, choose the single primary issue.
Set confidence between 0 and 1.
If the text is empty, vague, or you cannot tell the primary issue, set confidence below {CONFIDENCE_THRESHOLD}.

Return only the structured fields: category, severity, confidence, and a short reasoning string.
"""


def classification_messages(raw_text: str) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=raw_text),
    ]
