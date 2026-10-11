# 0002. Review screen is a local Streamlit page

- Status: accepted
- Date: 2026-10-10

## Context

Paused ticket reviews were completed by typing approve, edit, or reject in the terminal. A reviewer needs to see the ticket, the findings, and the draft, then choose one of those three actions.

## Decision

A local Streamlit page lists paused runs and resumes the existing checkpoint. Approve, edit, and reject send the same resume value the terminal already uses. The page does not start a ticket. The terminal review command stays.

## Consequences

Review can be done from the page or the terminal. The page reads the checkpoint file, so a reload still shows the same pause. Reject can still call the chat model, because that action returns the draft to the responder.
