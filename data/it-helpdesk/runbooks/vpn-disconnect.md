<!-- source: https://internal.example.com/runbooks/it/vpn-disconnect -->

# VPN disconnects hourly

## Symptoms

- Corporate VPN drops every 60 minutes.
- User must re-authenticate to reconnect.

## Resolution steps

1. Confirm the user's VPN client is on the latest version.
2. Check identity-service logs for password-reset throttling or certificate mismatches.
3. Reissue the user's VPN certificate from the admin portal.

## Escalation

Escalate to network ops if multiple users on the same gateway report simultaneous drops.
