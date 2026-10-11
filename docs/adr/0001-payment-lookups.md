# 0001. Payment lookups stay read-only

- Status: accepted
- Date: 2026-10-09

## Context

The ticket investigator could list an order, but a payments company needs the charges and refunds on that order as records. A duplicate-charge reply also needs the payment procedure. Those lookups must not move money.

## Decision

The investigator gains `list_charges`, `list_refunds`, and `search_runbook`. Sample charges, refunds, and one duplicate-charge runbook live in JSON beside the other mock stores. The tool-call cap is 8 so those lookups fit with the existing customer, order, past-ticket, and knowledge-base calls. Log search and metrics stay on the on-call agent.

## Consequences

A finding can cite a charge, refund, or runbook source. An empty refund list means nothing has been reversed. A reversal still waits for a person. New payment procedures are added as runbook records, not as new tool names.
