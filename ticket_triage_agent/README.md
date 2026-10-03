# Ticket Triage Agent

A supervisor routes each support ticket to triage, investigation, a draft reply, human review, and send. The responder reads three kinds of long-term memory: semantic (knowledge-base articles), episodic (approved replies), and procedural (the current response rules).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env` when `MODEL_PROVIDER` is `openai`. Thresholds, loop caps, and the embedding model are in `.env.example`. Commented names there are optional.

Seed the dev store once. This calls the embedding API and writes `data/memory.sqlite`:

```powershell
.venv\Scripts\python scripts\seed_memory.py
```

Check that every uncommented name in `.env.example` is set, and that the store was seeded with the configured embedding model:

```powershell
.venv\Scripts\python scripts\run_ticket.py check
```

The check prints missing setting names only. It does not print values. If the store was stamped with the fake embedder, delete `data/memory.sqlite`, set `EMBEDDING_MODEL` in `.env`, and run `scripts/seed_memory.py` again.

Offline tests skip anything marked `live`:

```powershell
.venv\Scripts\python -m pytest
```

## Architecture

```
                 ┌──────────┐
                 │  Triage  │
                 └────▲─────┘
                      │
 ┌──────┐       ┌─────┴──────┐       ┌──────────────┐
 │ Send │◄──────│ Supervisor │──────►│ Investigator │
 └──┬───┘       └─────┬──────┘       └──────────────┘
    │            ┌────┴─────┐
    │       ┌────▼───┐  ┌───▼───────┐◄── Semantic memory
    └──────►│  HITL  │  │ Responder │◄── Episodic memory
            └────────┘  └───────────┘◄── Procedural memory
```

The supervisor is the hub. Triage, the investigator, the responder, and HITL each return to it. Send ends the run, unless the draft makes a flagged or escalating claim that nobody approved; that path goes back to HITL. Spam, and an escalated run that was not approved, go through a close step that records an open escalation only when the run was escalated and nothing was sent.

Routing is deterministic. A draft needs review when priority is P1, triage or draft confidence is under the threshold, sentiment is negative, the category is refund, security, or legal, the ticket needs a human, investigation is missing a customer or a named order, or the reply promises an escalation. Those limits live in `.env`.

## How to run

From this directory:

```powershell
.venv\Scripts\python scripts\run_ticket.py run --ticket-id T-001
.venv\Scripts\python scripts\run_ticket.py run --all
.venv\Scripts\python scripts\run_ticket.py review
.venv\Scripts\python scripts\run_ticket.py rules
.venv\Scripts\python scripts\run_ticket.py rules rollback 1
.venv\Scripts\python scripts\run_ticket.py escalations
```

`run` uses the chat model from `.env` and the dev files `data/memory.sqlite` and `data/checkpoints.sqlite`. Each ticket gets a new thread, so a finished run does not block the next one. A ticket that meets the review rules pauses. It is not a failure. Finish it with `review`.

`run --all` walks `data/sample_tickets.json` and prints a table: ticket, category, route, why it went to HITL, whether it escalated, and whether it was sent. The totals are:

- handled without a human: the route never entered HITL, including spam that was closed with no reply
- sent to HITL: review was required
- escalated: the supervisor marked the run for manual handling (iteration cap, repeated errors, or the revision cap)

## Review and rule learning

`review` lists paused runs. Pick one to see the ticket, triage, findings, draft, flagged claims, and the reason for review. Then approve, edit, or reject.

- Approve sends the draft. Edit sends the text you provide. Both save a redacted episodic example (emails, phone numbers, the customer name, order ids, and amounts are replaced) so a later similar ticket can use it as a style example.
- Reject sends the draft back to the responder with your feedback, until the revision cap, which escalates to review again.
- If you add feedback, you can save it as a rule. That appends a new procedure version and makes it current. Older versions stay. `rules` lists them. `rules rollback <version>` points the current version at an older one without deleting the later text.

The next draft loads the current procedure into its system prompt.

## Swap the mocks

Lookups, the knowledge base, and delivery sit behind protocols. Nodes do not read the JSON files themselves.

- Customers, orders, and past tickets: implement `CustomerStore`, `OrderStore`, and `PastTicketStore` in `src/triage_agent/tools/ticket_db.py`. Each method returns a `LookupResult`. Pass the implementations to `build_tools`. The JSON classes (`JsonCustomerStore`, `JsonOrderStore`, `JsonPastTicketStore`) are the dev stand-ins, backed by `data/customers.json`, `data/orders.json`, and `data/past_tickets.json`.
- Knowledge base: `StoreKnowledgeBase.search_kb` reads the semantic namespace in the LangGraph store. Seed or upsert articles through `src/triage_agent/memory/semantic.py`, or replace `KnowledgeBase` with another search backend and pass it to `build_tools`.
- Email or Slack, and a follow-up queue: implement `Sender` and `EscalationLog` in `src/triage_agent/tools/sender.py`. `OutboxSender` writes one JSON file per run under `outbox/`. `JsonEscalationLog` writes open cases under `escalations/`. `send` calls those interfaces; point it at your channel by passing the implementations into the send node, or replace the two classes that `send` constructs from `OUTBOX_DIR` and `ESCALATIONS_DIR`.
