"""Local page for the ticket triage agent. It listens only on this computer."""

from __future__ import annotations

import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from web.service import Desk, DeskError

_PAGE = Path(__file__).resolve().parent / "static" / "index.html"
_MAX_BODY = 1_000_000


def make_server(desk: Desk, host: str = "127.0.0.1", port: int = 8010) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        desk_app = desk

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                _send(self, 200, _PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            _send_json(self, 404, {"error": "Not found."})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = _read_json(self)
                if path == "/api/triage":
                    customer_id = body.get("customer_id")
                    ticket_id = body.get("ticket_id")
                    result = self.desk_app.classify(
                        str(body.get("text", "")),
                        customer_id=None if customer_id in (None, "") else str(customer_id),
                        channel=str(body.get("channel") or "email"),
                        ticket_id=None if ticket_id in (None, "") else str(ticket_id),
                    )
                    _send_json(self, 200, result)
                    return
                _send_json(self, 404, {"error": "Not found."})
            except DeskError as exc:
                _send_json(self, 400, {"error": exc.message})
            except (json.JSONDecodeError, UnicodeError, ValueError):
                _send_json(self, 400, {"error": "Send the ticket as JSON."})

        def log_message(self, fmt: str, *args) -> None:
            return

    return _ExclusiveServer((host, port), Handler)


class _ExclusiveServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(host: str = "127.0.0.1", port: int = 8010) -> None:
    try:
        server = make_server(Desk(), host, port)
    except OSError as exc:
        raise SystemExit(
            f"Port {port} is already in use on this computer. "
            "Another program is serving that address."
        ) from exc
    print(f"Ticket triage: http://{host}:{port}")
    print("Leave this window open. The OpenAI key stays in this window.")
    server.serve_forever()


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length < 0 or length > _MAX_BODY:
        raise ValueError("body too large")
    raw = handler.rfile.read(length) if length else b"{}"
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("expected an object")
    return parsed


def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    _send(handler, status, json.dumps(payload).encode("utf-8"), "application/json")
