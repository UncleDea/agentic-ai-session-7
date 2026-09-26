<!-- source: https://internal.example.com/runbooks/auth/mfa-loop -->

# MFA loop on login

## Symptoms

- User enters their MFA code from an authenticator app.
- After submitting, they are redirected back to the login page.
- Repeats indefinitely.

## Common causes

1. Device clock drift. The authenticator app on the user's phone is more than 30 seconds out of sync with our servers. The generated TOTP code is therefore expired by the time it reaches us.
2. Replay attack protection misfired. If the user double-clicks Submit, the second submission is rejected as a replay.
3. Cached session token from a prior failed attempt is still in the user's browser.

## Resolution steps

1. Ask the user to fully close the authenticator app and reopen it. Most apps sync time on open.
2. Ask the user to clear cookies for our domain and try again from a fresh browser tab.
3. If still failing, manually reset the user's MFA device from the admin panel and walk them through re-enrolment.

## Escalation

Escalate to the auth-platform on-call if the user reports multiple MFA loops across multiple sessions in the same day - this could indicate a bug in the new TOTP service rollout.
