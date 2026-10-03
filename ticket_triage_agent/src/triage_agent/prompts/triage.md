Classify this support ticket into the schema. Do not draft a customer reply.

Choose one primary category, the action the customer wants:

- billing: a duplicate charge, a wrong amount, or a subscription billed after it was canceled. Asking to reverse or remove that incorrect charge is still billing.
- technical: the product or app fails, crashes, or behaves incorrectly.
- account: login, password, email address, or other account settings.
- shipping: delivery, tracking, or an order that has not arrived, when a refund is not the main action.
- refund: the customer wants the purchase price back because an item is damaged or unwanted, or because an order should be canceled. Undoing a mistaken or duplicate charge is billing, not refund.
- feedback: praise, thanks, or a comment with no request to fix something.
- spam: unsolicited promotion, phishing, or a message that is not a real support request.
- security: account takeover, an unexpected password or address change, or a request to lock an account.
- legal: a lawyer, a lawsuit, a chargeback, or a dispute filed with a card issuer. If any of those appear, legal is the primary category even when the customer also asks for money back.
- other: a real question that fits none of the categories above.

Put any other real issue in secondary_categories. Do not repeat the primary category there. Use an empty list when there is a single issue.

Priority starts at the default for the primary category:

- P1: security, legal
- P2: billing, refund
- P3: technical, account, shipping
- P4: feedback, spam, other

Raise that default by one level, and only one level, for a clear severity signal: an outage affecting many users, data loss, money charged repeatedly, or a stated deadline. Never set a priority below the category default. Do not raise P1.

Sentiment is the customer's wording toward the company, not how serious the topic is. Hostile, angry, or threatening wording is negative. Thanks or praise from a real customer is positive. A factual report is neutral, including a security incident told without insults. Spam and phishing are neutral: words such as congratulations in an unsolicited promotion are not customer sentiment. Language is the ISO 639-1 code of the body, such as en or es.

Set needs_human to true for a security incident, a legal or chargeback threat, or a body that is empty or only a few words, such as "help". A body that short does not say what to do, so treat it as something a person must read. Otherwise set needs_human to false. When it is true, needs_human_reason is a short phrase. When it is false, needs_human_reason is an empty string.

Use a high confidence for a clear ticket. Use a confidence below 0.7 when the body is empty or only a few words, and use category other for those tickets.
