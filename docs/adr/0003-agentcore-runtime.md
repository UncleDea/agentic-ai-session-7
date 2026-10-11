# 0003. The ticket graph runs on AgentCore

- Status: superseded by 0004
- Date: 2026-10-11

## Context

The review page needs a host, and the ticket graph needs a host that can call Bedrock. AgentCore Runtime serves an invoke endpoint. It does not serve the Streamlit page. The paused queue, memory store, outbox, and escalations have to stay together for one review session.

## Decision

The graph runs in an AgentCore Runtime container. Session files live under `/mnt/data` for one fixed session id. The Streamlit page runs on App Runner and calls that runtime with IAM. The page asks for a review token before it shows a ticket. With no runtime ARN, the page still reads the local checkpoint file. The terminal review command stays.

## Consequences

A reviewer can open the page on AWS and resume a pause that the runtime created. That queue lasts only as long as the runtime session, at most 8 hours. A new session id starts an empty queue. Reject can still call the chat model, because that action runs inside the runtime.
