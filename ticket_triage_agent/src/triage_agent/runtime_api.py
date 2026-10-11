"""Actions the AgentCore entrypoint serves. No AWS client."""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

from triage_agent.cli import find_ticket, row_from_values
from triage_agent.graph import list_paused_runs, thread_config
from triage_agent.review_ui import decision_from_action, resume_thread
from triage_agent.state import initial_state

SESSION_ROOT = Path("/mnt/data")
QUEUE_PREFIX = "ticket-review/queue/"
_SQLITE_NAMES = ("checkpoints.sqlite", "memory.sqlite")
_JSON_DIRS = ("outbox", "escalations")
_SAVED_ACTIONS = frozenset({"list", "get", "run", "resume"})


def queue_bucket() -> str:
    return os.environ.get("QUEUE_BUCKET", "").strip()


def prepare_session_files(root: Path, source_memory: Path, *, s3_client=None) -> None:
    """Create the session directories and copy the seeded store once."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "outbox").mkdir(exist_ok=True)
    (root / "escalations").mkdir(exist_ok=True)
    restore_queue(root, client=s3_client)
    memory = root / "memory.sqlite"
    if not memory.exists() and source_memory.is_file():
        shutil.copyfile(source_memory, memory)
    os.environ["CHECKPOINT_DB_PATH"] = str(root / "checkpoints.sqlite")
    os.environ["MEMORY_DB_PATH"] = str(memory)
    os.environ["OUTBOX_DIR"] = str(root / "outbox")
    os.environ["ESCALATIONS_DIR"] = str(root / "escalations")


def restore_queue(root: Path, *, client=None) -> None:
    """Download the shared queue. No bucket means the local files stay."""
    bucket = queue_bucket()
    if not bucket:
        return
    root.mkdir(parents=True, exist_ok=True)
    for name in _SQLITE_NAMES:
        for suffix in ("-wal", "-shm"):
            sidecar = root / f"{name}{suffix}"
            if sidecar.exists():
                sidecar.unlink()
    store = client or _s3_client()
    for key in _list_keys(store, bucket, QUEUE_PREFIX):
        relative = _relative_key(key)
        if relative is None:
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        body = store.get_object(Bucket=bucket, Key=key)["Body"].read()
        destination.write_bytes(body)


def save_queue(root: Path, *, client=None, graph=None) -> None:
    """Upload a consistent snapshot. No bucket means nothing is uploaded."""
    bucket = queue_bucket()
    if not bucket:
        return
    store = client or _s3_client()
    snapshot = root / ".queue-snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir()
    try:
        flushed = _backup_open_databases(graph, snapshot)
        for name in _SQLITE_NAMES:
            if name in flushed:
                _put_file(store, bucket, f"{QUEUE_PREFIX}{name}", snapshot / name)
                continue
            source = root / name
            if not source.is_file():
                continue
            destination = snapshot / name
            _backup_sqlite(source, destination)
            _put_file(store, bucket, f"{QUEUE_PREFIX}{name}", destination)
        for folder in _JSON_DIRS:
            directory = root / folder
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                _put_file(store, bucket, f"{QUEUE_PREFIX}{folder}/{path.name}", path)
    finally:
        shutil.rmtree(snapshot, ignore_errors=True)


def handle_action(graph, payload: dict, *, queue_root: Path | None = None, s3_client=None) -> dict:
    """Run one review action. Errors are a dict with an error string."""
    action = str(payload.get("action") or "")
    if action == "list":
        result = {"paused": list_paused_runs(graph)}
    elif action == "get":
        result = _get(graph, str(payload.get("thread_id") or ""))
    elif action == "resume":
        result = _resume(graph, payload)
    elif action == "run":
        result = _run(graph, str(payload.get("ticket_id") or ""))
    else:
        return {"error": "Unknown action."}
    if action in _SAVED_ACTIONS:
        root = queue_root if queue_root is not None else (SESSION_ROOT if queue_bucket() else None)
        if root is not None and queue_bucket():
            save_queue(root, client=s3_client, graph=graph)
    return result


def _get(graph, thread_id: str) -> dict:
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.interrupts:
        return {"error": "That run is not waiting for review."}
    value = snapshot.interrupts[0].value
    if not isinstance(value, dict):
        return {"error": "That run is not waiting for review."}
    return {"payload": value}


def _resume(graph, payload: dict) -> dict:
    thread_id = str(payload.get("thread_id") or "")
    decision = decision_from_action(
        str(payload.get("decision_action") or ""),
        str(payload.get("text") or ""),
        str(payload.get("feedback") or ""),
        bool(payload.get("save_as_rule")),
    )
    if isinstance(decision, str):
        return {"error": decision}
    return {"outcome": resume_thread(graph, thread_id, decision)}


def _s3_client():
    import boto3

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    return boto3.client("s3", region_name=region)


def _list_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs: dict = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        keys.extend(item["Key"] for item in page.get("Contents") or [])
        if not page.get("IsTruncated"):
            return keys
        token = page.get("NextContinuationToken")


def _relative_key(key: str) -> Path | None:
    if not key.startswith(QUEUE_PREFIX):
        return None
    relative = key[len(QUEUE_PREFIX) :]
    if not relative or relative.endswith("/"):
        return None
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _backup_open_databases(graph, snapshot: Path) -> set[str]:
    """Copy from the connections the graph already has open."""
    if graph is None:
        return set()
    saved: set[str] = set()
    pairs = (
        ("checkpoints.sqlite", getattr(getattr(graph, "checkpointer", None), "conn", None)),
        ("memory.sqlite", getattr(getattr(graph, "store", None), "conn", None)),
    )
    for name, conn in pairs:
        if conn is None:
            continue
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        _backup_connection(conn, snapshot / name)
        saved.add(name)
    return saved


def _backup_connection(connection, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied = sqlite3.connect(destination)
    try:
        connection.backup(copied)
    finally:
        copied.close()


def _backup_sqlite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(source)
    try:
        dst = sqlite3.connect(destination)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _put_file(client, bucket: str, key: str, path: Path) -> None:
    client.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())


def _run(graph, ticket_id: str) -> dict:
    try:
        ticket = find_ticket(ticket_id)
    except SystemExit as exc:
        text = exc.code if isinstance(exc.code, str) else "Unknown ticket id."
        return {"error": text}
    config = thread_config(ticket.id)
    graph.invoke(initial_state(ticket), config, durability="sync")
    snapshot = graph.get_state(config)
    return {
        "thread_id": config["configurable"]["thread_id"],
        "paused": snapshot.next == ("hitl",),
        "row": row_from_values(dict(snapshot.values), ticket.id),
    }
