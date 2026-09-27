"""Record of an attempted delivery."""

from typing import Literal

from pydantic import BaseModel


class Delivery(BaseModel):
    status: Literal["sent", "not_sent"]
    outbound_id: str | None = None
    recipient: str | None = None
    subject: str
    body: str
    reason: str | None = None
    sent_at: str | None = None
