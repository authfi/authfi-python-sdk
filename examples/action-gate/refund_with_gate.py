"""An agent freezes a high-risk action, asks AuthFI's gate, and runs it only on ``allow``.

Run it::

    pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
    cp .env.example .env    # then fill it in
    python refund_with_gate.py 4411 120
"""

from __future__ import annotations

import sys

from authfi import AuthFI


def refund(order: str, amount: int) -> None:
    """Refund ``amount`` on ``order``, if the gate allows it."""
    with AuthFI.from_env() as af:
        agent = af.agent_from_env()
        # Frozen for 60 seconds: the gate decides exactly this action, resource and context.
        txn = agent.transaction_token("refund", resource=f"order:{order}", context={"amount": amount})
        decision = agent.authorize_action(txn)
        if decision.decision == "allow":
            print(f"Refunding {amount} on order {order}.")
        elif decision.decision == "approval_required":
            print(f"A person must approve first (approval {decision.approval_id}).")
        else:
            print(f"Refused: {decision.reason or 'no reason given'}.")


if __name__ == "__main__":
    refund(sys.argv[1], int(sys.argv[2]))
