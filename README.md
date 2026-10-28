# TICKET TRIAGE AGENT

Supervisor-pattern customer support system. The Supervisor receives each ticket, calls Triage, and routes from that result.

Pipeline order:

1. Triage — classifies the ticket (built)
2. Supervisor — chooses the next node (built)
3. Investigator — looks up local tickets, logs, metrics, and runbooks (built)
4. Responder — drafts the reply from memory (built)
5. HITL — human review when severity is high (built)
6. Send — writes the reply to `data/outbox/sent.jsonl`, then can email it (built)

Open the triage agent in a browser from the repository root. Set `OPENAI_API_KEY` in that same terminal first. The page stays on this computer and only classifies the ticket.

```text
python -m triage_agent
```

Then open http://127.0.0.1:8010. Enter the customer message and choose Classify. The page shows category, severity, confidence, and whether the next step is a person or the investigator.

The Supervisor calls `run_triage(TicketInput) -> TicketState` and chooses the next node. High or critical severity, and any `hitl` hint, pauses for a person. `resume_review(thread_id, {"action": "approve"})` continues through Investigator, Responder, and Send. `reject` stops. A draft that needs review, or a ticket with no recipient, is held and not written. Other drafts are appended to the outbox. Email them with `python -m send deliver` after setting `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, and `MAIL_FROM`. A row is emailed once. The recipient must be a real mailbox.

Run one ticket:

```text
python -m supervisor run --text "Authenticator codes loop back to the MFA page." --customer-id priya.s@example.com
```

Set `OPENAI_API_KEY` first. A high or critical ticket prints a `resume:` command. Approve or reject it in a later command:

```text
python -m supervisor resume THREAD_ID --action approve --note "Looks right."
```

The paused review is stored in `data/reviews/`.

```text
send/
  agent.py      run_send writes data/outbox/sent.jsonl
  deliver.py    emails records not yet in delivered.jsonl
  schemas.py    Delivery

hitl/
  review.py     why the ticket was paused
  schemas.py    HumanDecision, ReviewRequest

responder/
  memory.py     semantic, episodic, and procedural memory
  prompts.py    draft prompt
  agent.py      run_response
  schemas.py    ResponseDraft
  tests/

investigator/
  lookup.py     local search, no network
  agent.py      run_investigation
  schemas.py    Investigation
  tests/

supervisor/
  graph.py      LangGraph: triage -> supervisor -> hitl | investigator
  routing.py    choose_next_node
  schemas.py    SupervisorResult
  tests/

triage_agent/
  schemas.py    TicketInput, TicketState
  prompts.py    classification prompt
  agent.py      run_triage
  config.py     model name, temperature, confidence threshold
  tests/
```

Classification uses the ticket text only. Tests inject a fake model, so the suite does not call an API.
