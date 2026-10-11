# Cursor Prompt — Ticket Triage Agent (Multi-Agent Supervisor)

> How to use: put this file in your repo root. In Cursor, open Plan Mode and say:
> "Read @cursor-prompt-ticket-triage-agent.md. Plan Phase 0 only. Do not write code until I approve the plan."
> Then go phase by phase. Don't let Cursor jump ahead.

---

## Role and goal

You are a senior Python engineer building a **multi-agent customer-support Ticket Triage Agent** with **LangGraph** and **LangChain**. A central **Supervisor** routes a ticket through specialist nodes (Triage, Investigator, Responder, HITL, Send). The **Responder** draws on three kinds of long-term memory: **Semantic**, **Episodic**, and **Procedural**.

Build it **incrementally, one component at a time**. Finish each component, test it on its own, and stop for my review before starting the next one. Every component must be decoupled: it reads from and writes to the shared state only, and it must be testable without the other components.

Before using any LangGraph or LangChain API, check the version installed in this project and follow the current official docs for that version. Do not guess at API signatures. If something in this spec conflicts with the current API, tell me and propose the closest equivalent.

---

## Architecture

```
                 ┌──────────┐
                 │  Triage  │
                 └────▲─────┘
                      │
 ┌──────┐       ┌─────┴──────┐       ┌──────────────┐
 │ Send │◄──────│ Supervisor │──────►│ Investigator │
 └──────┘       └─────┬──────┘       └──────────────┘
                 ┌────┴─────┐
            ┌────▼───┐  ┌───▼───────┐◄── Semantic memory
            │  HITL  │  │ Responder │◄── Episodic memory
            └────────┘  └───────────┘◄── Procedural memory
```

- **Hub and spoke:** every worker node returns control to the Supervisor, except Send, which ends the run.
- **Supervisor** decides the next node based on the current state.
- **Triage** classifies the ticket.
- **Investigator** gathers facts (customer, order or account details, related past tickets, knowledge-base articles).
- **Responder** drafts the reply using the investigation findings plus the three memory types.
- **HITL** pauses the graph for a human to approve, edit, or reject the draft.
- **Send** delivers the final reply and ends the run.

---

## Tech stack

- Python 3.11+
- `langgraph`, `langchain`, `langchain-core`, and one model provider package (OpenAI or Anthropic; read the choice from config)
- `pydantic` v2 for every schema and for structured LLM output
- LangGraph **checkpointer** for short-term or thread state (in-memory for dev, SQLite or Postgres later)
- LangGraph **Store** for long-term memory (in-memory store for dev, with embeddings-backed search for semantic memory)
- `pytest` for tests, `python-dotenv` for config
- Optional: LangSmith tracing, enabled through environment variables

Pin dependency versions in `pyproject.toml` (or `requirements.txt`).

---

## Project structure

```
ticket_triage_agent/
  pyproject.toml
  .env.example
  README.md
  src/triage_agent/
    config.py              # settings, model choice, thresholds, max loops
    state.py               # TicketState + pydantic sub-models
    llm.py                 # model factory
    graph.py               # builds and compiles the StateGraph
    nodes/
      supervisor.py
      triage.py
      investigator.py
      responder.py
      hitl.py
      send.py
    tools/
      ticket_db.py         # mock ticket/customer/order lookups behind an interface
      knowledge_base.py    # KB search behind an interface
      sender.py            # mock sender (writes to outbox/), real channel later
    memory/
      store.py             # store setup + namespaces
      semantic.py
      episodic.py
      procedural.py
    prompts/               # one prompt file per node
  data/
    sample_tickets.json    # ~15 realistic tickets covering every category and edge case
    kb_articles.json
    seed_episodes.json
    seed_procedures.md
  tests/
    test_state.py
    test_triage.py
    test_supervisor.py
    test_investigator.py
    test_memory.py
    test_responder.py
    test_hitl.py
    test_send.py
    test_graph_e2e.py
  scripts/
    run_ticket.py          # CLI: run one ticket end to end, prompt for HITL input
```

---

## Shared state (`state.py`)

Define a `TicketState` that every node reads from and writes to. It needs at least:

- `ticket`: id, customer_id, channel, subject, body, created_at
- `triage`: category (billing, technical, account, shipping, refund, feedback, spam, other), priority (P1–P4), sentiment, language, confidence (0–1), `needs_human` flag with a reason
- `investigation`: findings (list of fact plus source), related_ticket_ids, kb_article_ids, `missing_info` (list)
- `draft`: reply text, the memory items used (ids), self-assessed confidence
- `review`: status (pending, approved, edited, rejected), reviewer feedback, edited text
- `final_reply`, `sent` (bool), `sent_at`
- Control fields: `next_node`, `route_history` (list), `iteration_count`, `revision_count`, `errors` (list)

Use pydantic models for each sub-object and reducers where lists accumulate.

---

## Node specs

### 1. Supervisor
- Decides `next_node` from the state. Use **deterministic rules first**, and fall back to an LLM with structured output only for ambiguous cases.
- Routing rules:
  - No triage yet → **Triage**
  - Triage says spam → end (log it, no reply)
  - Triage done, no investigation, and category needs facts → **Investigator**
  - Investigation done, no draft → **Responder**
  - Draft exists and HITL criteria met (below) and review is not approved → **HITL**
  - Review rejected → **Responder** (revision, with the feedback), unless `revision_count` ≥ max, then **HITL**
  - Draft approved or edited, or HITL criteria not met → **Send**
- **HITL criteria** (configurable in `config.py`): priority P1, triage confidence or draft confidence below threshold, negative sentiment, refund or legal or security category, `needs_human` true, or investigation has unresolved `missing_info`.
- Guardrails: hard cap on `iteration_count` (for example 10). When the cap is hit, route to HITL. Append every decision to `route_history` with a one-line reason.

### 2. Triage
- LLM with structured output (a pydantic schema) that fills `triage`.
- Include a few-shot prompt built from `sample_tickets.json`.
- Must handle empty or very short tickets, non-English tickets, and multi-issue tickets (pick the primary category and note the others).

### 3. Investigator
- A tool-calling agent with tools: `get_customer(customer_id)`, `get_orders(customer_id)`, `search_past_tickets(customer_id, query)`, `search_kb(query)`.
- All tools are mocks backed by JSON in `data/`, behind interfaces, so real systems can be swapped in later.
- Writes concise findings, each with a source. Records anything it couldn't find in `missing_info`. Limit tool calls (for example 6).

### 4. Responder (with memory)
Before drafting, retrieve from the LangGraph Store:
- **Semantic memory** (namespace such as `("semantic", "kb")`): facts, policies, and product knowledge, retrieved by embedding search on the ticket plus findings.
- **Episodic memory** (`("episodic", "resolved_tickets")`): past resolved tickets with their approved replies, used as few-shot examples. Retrieve the top 2–3 similar ones.
- **Procedural memory** (`("procedural", "response_rules")`): the current response rules and style guide (tone, sign-off, what never to promise, escalation phrasing). It's loaded into the system prompt and must be updatable.

Then draft the reply. Fill `draft` with the text, the memory ids used, and a confidence score. On a revision, include the reviewer's feedback and the previous draft. The reply must never invent order details, refunds, or policies that aren't in the findings or semantic memory.

### 5. HITL
- Use LangGraph's **interrupt** mechanism (with a checkpointer and thread id) to pause and show the reviewer: the ticket, the triage result, the findings, the draft, and the reason for review.
- Resume with one of: **approve**, **edit** (with the new text), or **reject** (with feedback).
- **Memory learning on resume:**
  - Approved or edited replies get written to **episodic** memory as a resolved example.
  - Reject or edit feedback that states a general rule (for example "never promise a delivery date") gets proposed as an update to **procedural** memory. Log the change, and keep the previous version so it can be rolled back.

### 6. Send
- Calls `sender.send(ticket, final_reply)`. The dev implementation writes a JSON file to `outbox/` and logs it.
- Sets `sent` and `sent_at`, then ends the graph.
- Must be **idempotent**: if the ticket is already sent, don't send again.

---

## Cross-cutting requirements

- Every node is a pure function `(state) -> partial state update`. No hidden globals.
- Structured logging for each node (node name, ticket id, decision, latency).
- Errors are caught per node and appended to `errors`. When errors repeat, the Supervisor routes to HITL instead of crashing.
- Every prompt lives in `prompts/`, not inline in the code.
- All thresholds and limits live in `config.py`.
- Tests mock the LLM by default (fake or stub responses), so the test suite runs offline and costs nothing. Mark live-LLM tests with `@pytest.mark.live`.

---

## Build plan — one phase at a time

**Stop after each phase.** Show me the files changed and the test results, and wait for my "approved, next" before continuing.

- **Phase 0 — Scaffold:** project structure, dependencies, config, `state.py`, LLM factory, sample data. Tests: state validation.
- **Phase 1 — Triage node:** prompt plus structured output. Tests: each category, spam, non-English, empty ticket.
- **Phase 2 — Supervisor plus graph skeleton:** routing rules, with stub nodes for everything not built yet. Tests: every routing branch, the iteration cap, `route_history`.
- **Phase 3 — Investigator plus mock tools:** Tests: tool selection, `missing_info`, tool-call limit.
- **Phase 4 — Memory layer:** store setup, the three namespaces, seed scripts, retrieval functions. Tests: put/get/search for each type, and procedural versioning.
- **Phase 5 — Responder:** memory retrieval plus drafting plus revision mode. Tests: memory ids recorded, no invented facts on a fixture ticket, revision uses the feedback.
- **Phase 6 — HITL:** interrupt and resume with approve, edit, and reject, plus memory learning. Tests: each resume path, and the episodic and procedural writes.
- **Phase 7 — Send:** mock sender, idempotency. Tests: single send, duplicate guard.
- **Phase 8 — End to end:** `scripts/run_ticket.py` CLI (interactive HITL in the terminal), an e2e test over every sample ticket, and a README covering setup, the architecture diagram, how to run, and how to add real integrations.

For each phase, start by giving me a short plan: files, functions, test cases. Only write code after I approve the plan.

---

## Definition of done

- `pytest` passes offline.
- `python scripts/run_ticket.py --ticket-id <id>` runs a ticket end to end, pauses for review when the criteria are met, and writes the reply to `outbox/`.
- An approved reply shows up as an episodic example on the next similar ticket.
- A procedural rule added through HITL feedback changes the next draft's behaviour.
- The README explains how to swap the mocks for a real ticket system, knowledge base, and email or Slack sender.
