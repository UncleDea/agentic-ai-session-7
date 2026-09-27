"""Runtime settings for the Triage Agent."""

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0
CONFIDENCE_THRESHOLD = 0.6

CATEGORIES = (
    "billing",
    "technical",
    "account",
    "feature_request",
    "spam",
    "other",
)

SEVERITIES = (
    "low",
    "medium",
    "high",
    "critical",
)
