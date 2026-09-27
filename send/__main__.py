"""python -m send deliver"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from send.deliver import deliver_pending, smtp_transport_from_env


def main(argv: list[str] | None = None, *, transport=None, outbox: Path | None = None, ledger: Path | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m send")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("deliver", help="Email outbox records that have not been delivered")
    args = parser.parse_args(argv)
    if args.command != "deliver":
        return 2
    mailer = transport if transport is not None else smtp_transport_from_env()
    if mailer is None:
        raise SystemExit(
            "Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, and MAIL_FROM "
            "before delivering the outbox."
        )
    sent = deliver_pending(outbox=outbox, ledger=ledger, transport=mailer)
    print(json.dumps({"delivered": len(sent), "outbound_ids": [row["outbound_id"] for row in sent]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
