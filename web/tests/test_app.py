"""The triage page classifies one ticket. The model is a stub, so no API key is used."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from triage_agent import run_triage
from triage_agent.schemas import TriageClassification
from triage_agent.tests.test_agent import FakeModel
from web.server import make_server
from web.service import Desk


def _classifier(result):
    model = FakeModel(result)

    def triage(ticket):
        return run_triage(ticket, model=model)

    return triage


def _start(triage_fn):
    server = make_server(Desk(triage_fn=triage_fn), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _post(server, payload: dict) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{server.server_address[1]}/api/triage"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_page_is_the_triage_agent():
    server = _start(_classifier(TriageClassification(
        category="technical", severity="low", confidence=0.8, reasoning="Minor question."
    )))
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/") as response:
            page = response.read().decode("utf-8")
        assert response.status == 200
        assert "Ticket triage" in page
        assert "Classify" in page
        assert "Email the queue" not in page
    finally:
        server.shutdown()


def test_classify_returns_category_and_severity():
    server = _start(_classifier(TriageClassification(
        category="technical",
        severity="high",
        confidence=0.9,
        reasoning="Authenticator codes loop back to the MFA page.",
    )))
    try:
        status, payload = _post(server, {
            "text": "Authenticator codes loop back to the MFA page.",
            "customer_id": "priya.s@example.com",
            "channel": "email",
        })
        assert status == 200
        assert payload["category"] == "technical"
        assert payload["severity"] == "high"
        assert payload["confidence"] == 0.9
        assert payload["status"] == "triaged"
        assert payload["next_node_hint"] == "hitl"
        assert payload["classification_failed"] is False
    finally:
        server.shutdown()


def test_model_failure_is_marked():
    server = _start(_classifier(RuntimeError("invalid api key")))
    try:
        _, payload = _post(server, {"text": "I cannot log in."})
        assert payload["category"] == "other"
        assert payload["severity"] == "medium"
        assert payload["confidence"] == 0.0
        assert payload["classification_failed"] is True
        assert payload["next_node_hint"] == "hitl"
    finally:
        server.shutdown()


def test_blank_message_is_refused():
    server = _start(_classifier(TriageClassification(
        category="other", severity="low", confidence=0.9, reasoning="unused"
    )))
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/triage"
        request = Request(url, data=b'{"text":"  "}', headers={"Content-Type": "application/json"}, method="POST")
        try:
            urlopen(request)
            raised = False
        except HTTPError as exc:
            raised = True
            assert exc.code == 400
            detail = json.loads(exc.read().decode("utf-8"))
            assert "message" in detail["error"].lower()
        assert raised
    finally:
        server.shutdown()
