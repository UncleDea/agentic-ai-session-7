# Ticket Triage Agent

Phase 0 scaffold for a multi-agent support triage system. Later phases add the graph, nodes, memory, and CLI.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env
```

`MODEL_PROVIDER` is `openai` or `anthropic`. Thresholds and loop caps are in `.env.example`.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

The default run skips tests marked `live`, so the suite stays offline.
