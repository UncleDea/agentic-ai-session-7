"""Runtime settings for the Responder."""

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0

_SERVICE_DOMAIN = {
    "auth-service": "support",
    "billing-service": "support",
    "reports-service": "support",
    "vpn-gateway": "it-helpdesk",
    "identity-service": "it-helpdesk",
    "checkout-service": "oncall",
    "api-gateway": "oncall",
}

_EPISODIC_ANCHORS = ("mfa", "vpn", "checkout", "refund", "export", "password")
