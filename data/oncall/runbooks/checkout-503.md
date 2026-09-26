<!-- source: https://internal.example.com/runbooks/oncall/checkout-503 -->

# Checkout service 503 spike

## Symptoms

- API gateway returns 503 for `/pay` routes.
- Checkout error rate exceeds 5%.

## Common causes

1. Database connection pool exhausted on checkout-service.
2. Upstream dependency timeout during payment authorization.

## Resolution steps

1. Query checkout-service logs for `db_pool exhausted` or `request_failed`.
2. Check api-gateway metrics for error_rate and latency_ms trends.
3. Scale checkout-service replicas or restart pods to clear stuck connections.
4. Enable circuit breaker if upstream remains unhealthy.

## Escalation

Page the payments on-call if checkout error rate stays above 5% for more than 10 minutes.
