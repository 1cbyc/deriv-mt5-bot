# personal notes

i found the root cause of why it keeps rejecting order
🔍 MT5 last_error: (-2, 'Invalid "comment" argument')
The "comment" field in your order request is invalid for Deriv's MT5 server. This is why every order fails, regardless of volume or other parameters.


How to Fix
The comment string i sent (e.g., python-mt5-bot-20250630-064557) is not accepted by Deriv's MT5 server.
Some brokers (including Deriv) have strict requirements for the comment field (e.g., only alphanumeric, max length, or no special characters).

Solution
i changed the comment to a simple, short, alphanumeric string.

For example: "pythonMT5bot" or "MT5bot"

i updated this line in my order request:
   "comment": "pythonMT5bot",

updated the code to use safe comment string