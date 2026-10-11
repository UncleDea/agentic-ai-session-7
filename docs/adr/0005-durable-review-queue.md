# 0005. The review queue survives a runtime restart

- Status: accepted
- Date: 2026-10-11

## Context

The review page and the graph already pause a ticket. The paused rows, the outbox, and the escalations lived on the AgentCore session disk under `/mnt/data`. That directory belongs to one session. A recycled session starts empty, so the review page lost its queue.

## Decision

Sqlite stays the store. When `QUEUE_BUCKET` is set, the runtime downloads the prefix `ticket-review/queue/` into the session directory before the graph opens. After `list`, `get`, `run`, and `resume`, it uploads a consistent snapshot of that prefix. Checkpoint and memory files are copied with SQLite's backup API from the connections the graph already has open. Outbox and escalation files are copied as the JSON already written. The bucket is the private bucket `ticket-review-queue-343223593675` in `us-east-1`. The runtime role may list that prefix and get or put objects only under `ticket-review/queue/*`. With `QUEUE_BUCKET` unset, tests and the laptop page keep using local files. Every session id shares one prefix.

## Consequences

A second session id sees the same paused rows, which is how a restart is checked. The review page can keep its session id. Two approvals at the same moment can still clash, because both writes land in one sqlite file. The snapshot has to come from the open database connections. A second connection can upload an empty checkpoint file while the first connection still holds the rows. `agentcore deploy` drops `QUEUE_BUCKET`, so that variable has to be set again after a deploy.
