# The action gate

An agent about to refund money freezes the action into a 60-second transaction token, and asks
AuthFI's gate whether to run it.

## What it shows

- **Freeze, then decide.** The transaction token holds the action, its resource and its context.
  The gate refuses it for anything else, so a token for one refund cannot start another.
- **Limits are checked before the token is made.** An action over a limit never gets one.
- **Three answers.** `allow`, `deny`, or `approval_required` with the approval a person must give.

This gate is AuthFI's own. It is named after the IETF Transaction Tokens draft but does not follow
it.

## Run it

```bash
pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python refund_with_gate.py 4411 120
```
