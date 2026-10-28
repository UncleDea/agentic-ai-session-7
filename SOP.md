# Standard operating procedure: support ticket pipeline

## 1. Purpose

Run one customer ticket from intake through a reviewed reply and a single email. The app is a command-line pipeline. It is not a website.

## 2. Scope

This procedure covers a live ticket on the operator's machine:

1. Triage classifies the ticket text.
2. The Supervisor routes it.
3. A person approves or rejects when the ticket is paused.
4. The Investigator looks up local evidence.
5. The Responder drafts a reply.
6. Send queues the reply.
7. Deliver emails queued replies that have not been emailed yet.

Local evidence lives under `data/support`, `data/it-helpdesk`, and `data/oncall`. The Investigator does not call the network. Tests do not call OpenAI or Gmail.

## 3. Roles

| Role | Responsibility |
| --- | --- |
| Operator | Sets keys in the shell, starts a ticket, reads the printed result, and runs deliver. |
| Reviewer | Approves or rejects a paused ticket. The same person may do both jobs. |
| Models | Classify the ticket and draft the reply. They do not choose the route or send mail. |

The Supervisor decides the route. A node's hint is only a suggestion. Severity `high` or `critical` always pauses for a person, even if the hint says investigator.

## 4. Before you start

Work in the repository root:

```powershell
cd C:\Users\Agwes\agentic-ai-session-7
```

Python 3.11 or newer is required. Install once:

```powershell
pip install -e .
```

Use one PowerShell window for the whole ticket. Values set with `$env:` exist only in that window. A new window needs them set again.

Set the model key. Copy it from [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys). Do not paste it into chat, a file in this repo, or a screenshot.

```powershell
$env:OPENAI_API_KEY = "paste-the-key"
```

Set mail only when you are ready to email. Gmail needs an App Password from [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords), with 2-Step Verification turned on. Use the mailbox that will send, and use `@gmail.com`.

```powershell
$env:SMTP_HOST = "smtp.gmail.com"
$env:SMTP_PORT = "587"
$env:SMTP_USER = "sender@gmail.com"
$env:SMTP_PASSWORD = "the-16-character-app-password"
$env:MAIL_FROM = "sender@gmail.com"
```

Leave the example words above as examples. `you@gmail.com`, `the-app-password`, `smtp.example.com`, and `sk-your-key` are not valid.

## 5. Secrets

- Do not create a project `.env`.
- Do not commit `data/outbox/` or `data/reviews/`. Those directories are gitignored because they contain live ticket text and recipient addresses.
- Do not append live sends to `data/sent_responses.jsonl`. That file is historical memory for the Responder.
- If a key or App Password appears in the terminal or in a screenshot, revoke it and create a new one. Setting a new value replaces it for later commands. It does not erase the old value from the scrollback.

## 6. Run one ticket

To classify a ticket in the browser, without investigation or email, start the triage agent from the repository root:

```powershell
python -m triage_agent
```

Leave that window open and go to http://127.0.0.1:8010. Enter the customer message and choose Classify. The page shows category, severity, confidence, and the suggested next step. It does not approve, investigate, draft, or send.

The commands below run the full pipeline from the terminal.

`--customer-id` is the recipient. It must be a real mailbox when you intend to email the reply. `--channel` defaults to `email`.

```powershell
python -m supervisor run --text "Authenticator codes loop back to the MFA page." --customer-id customer@gmail.com
```

Optional flags:

- `--ticket-id TKT-1001` sets the id. Letters, numbers, dots, underscores, and hyphens only. If omitted, the app assigns `TKT-` plus eight characters.
- `--channel email` is the default. Any non-blank channel is accepted.

Read the printed block before you continue. A good classification has a category such as `technical`, `billing`, or `account`, a confidence above `0.0`, and a reason that describes the customer's issue. It does not say `401`.

### 6.1 Low or medium, and no human pause

The app continues through Investigator, Responder, and Send in the same command. `delivery: sent` means the reply was written to `data/outbox/sent.jsonl`. It has not been emailed yet. Continue at section 9.

`delivery: not_sent` with a `hold:` line means the reply was not queued. Continue at section 8.

### 6.2 Paused for review

`routed_to: hitl` and `review: pending` mean a person must decide. The app prints a `resume:` line and saves the pause in `data/reviews/`.

Pause when any of these are true:

- Severity is `high` or `critical`.
- Triage's hint is `hitl`.
- The text is empty or whitespace.
- Confidence is below `0.6`.
- The model call fails or the classification does not match the schema. Those cases use category `other`, severity `medium`, and confidence `0.0`.

Approve only when the category, severity, and reason match the ticket text.

```powershell
python -m supervisor resume THREAD_ID --action approve --note "Looks right."
```

Replace `THREAD_ID` with the id from the `resume:` line. Approve continues through Investigator, Responder, and Send.

Reject stops. There is no investigation, no draft, and no email.

```powershell
python -m supervisor resume THREAD_ID --action reject --note "Not a real issue."
```

A ticket that is already approved or rejected prints `This ticket was already decided.` and does not run again.

Do not approve a failed classification. Reject it, or leave it pending, then start a new ticket after the key is fixed. Signs of failure:

- category `other` and confidence `0.0`
- the reason says `model call failed` or `401`
- the reason says the text was empty

## 7. How to read the result

| Line | Meaning |
| --- | --- |
| `category` | `billing`, `technical`, `account`, `feature_request`, `spam`, or `other`. |
| `severity` | `low`, `medium`, `high`, or `critical`. |
| `routed_to: hitl` | Waiting on a person, or rejected. |
| `routed_to: send` | Reply queued in the outbox. |
| `routed_to: responder` | Draft was held and was not queued. |
| `service` and `runbook` | Local evidence the Investigator matched. |
| `needs_review: false` | Draft may be queued. |
| `needs_review: true` | Draft must not be emailed. |
| `delivery: sent` | Queued. Not the same as delivered to the mailbox. |
| `outbound_id` | Id of the queued message, shaped like `OUT-19FC294F`. |
| `hold:` | Why the reply was not queued. |

The Investigator matches local tickets, logs, metrics, and runbooks from the ticket text and customer id. No matching evidence is allowed. The Responder then marks the draft for review.

On the support domain, a draft that promises a refund or credit is replaced with a specialist-review note and held. A model failure uses that same safe note and is held.

## 8. Held drafts

A held draft is not in the outbox, so deliver will not email it. Holds happen when:

- `needs_review` is true, including no evidence, a refund or credit promise, or a failed draft
- `--customer-id` was omitted or blank

Read the subject and body. If the hold is correct, stop. The reply stays unsent. If the ticket text or recipient was wrong, start a new ticket. Do not edit the outbox by hand.

## 9. Email the queue

Deliver emails every outbox row that is not already in `data/outbox/delivered.jsonl`. Read the subject and body from the command output first.

```powershell
python -m send deliver
```

Success looks like:

```text
{"delivered": 1, "outbound_ids": ["OUT-19FC294F"]}
```

Check the recipient inbox, including Spam. Running deliver again does not send that `outbound_id` a second time.

`{"delivered": 0, "outbound_ids": []}` means nothing new was waiting. That is expected after a successful send, and also when every draft was held.

## 10. Failures

| What you see | What to do |
| --- | --- |
| `Set OPENAI_API_KEY before running the ticket models.` | Set the key in this window, then run the ticket again. |
| Reason contains `401` or `invalid_api_key` | The key is missing, mistyped, or revoked. Do not approve. Revoke it if it was exposed, set a new key, and run a new ticket. |
| `No paused review for THREAD_ID.` | Use the exact thread id from the `resume:` line of that ticket. |
| `This ticket was already decided.` | Start a new ticket if more work is needed. |
| `Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, and MAIL_FROM` | Set mail in this window. Host, user, password, and sender are all required. Port defaults to `587` only after the host is set. |
| `getaddrinfo failed` | `SMTP_HOST` is not a real server. For Gmail use `smtp.gmail.com`. |
| `535` `Username and Password not accepted` | User or password is wrong, or the normal Gmail password was used. Set `SMTP_USER` to the real `@gmail.com` address and `SMTP_PASSWORD` to an App Password. |
| `534` `Application-specific password required` | Turn on 2-Step Verification and create an App Password. |
| Deliver raises an error after login | The message was not marked delivered. Fix the cause and run `python -m send deliver` again. |

A failed SMTP login does not write the ledger. A successful send does. Do not delete ledger rows to force a resend unless a person has confirmed the recipient did not get the message.

## 11. Records

| Path | Contents | Commit |
| --- | --- | --- |
| `data/reviews/*.json` | Paused and decided reviews | No |
| `data/outbox/sent.jsonl` | Queued replies | No |
| `data/outbox/delivered.jsonl` | Ids already emailed | No |
| `data/support`, `data/it-helpdesk`, `data/oncall` | Local evidence | Yes |
| `data/prompts/` | Responder prompt samples | Yes |
| `data/sent_responses.jsonl` | Historical replies used as memory | Yes |

## 12. Check the app without spending credits

From the repository root, with no API key and no mail settings:

```powershell
python -m pytest
```

The suite uses stand-in models. A passing run does not send email and does not prove the live key or Gmail login.

## 13. Done

A ticket is done when one of these is true:

- Deliver printed the `outbound_id`, and the recipient mailbox received the message.
- The reviewer rejected it.
- The draft was held, and a person accepted that it must not be emailed.
