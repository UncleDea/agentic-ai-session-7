# 0004. The review page runs behind a load balancer

- Status: accepted
- Date: 2026-10-11

## Context

The ticket graph stays on AgentCore Runtime. The review page is Streamlit, which opens a websocket after the first page load. App Runner accepts the health check and returns the page shell, then its proxy rejects the websocket upgrade with HTTP 403. The page never finishes loading.

## Decision

The Streamlit page runs as an ECS Express service in `us-east-1`, behind the load balancer that service creates. That load balancer accepts the websocket. The page still calls the AgentCore runtime with IAM, using one session id, and still asks for `REVIEW_TOKEN` before it shows a ticket. With no runtime ARN, the page still reads the local checkpoint file.

## Consequences

A reviewer can open the public page and see a pause created in that runtime session. The copy of that queue that survives a restart is recorded in 0005. The page task role may invoke this runtime only. It cannot call the models, and it has no sqlite files.
