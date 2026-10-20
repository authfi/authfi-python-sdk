"""A voice agent refunds an order for a caller, with the caller's approval on their own phone.

The caller has no browser. The agent asks AuthFI to ask them (CIBA); they read the binding
message on their own device and approve or decline. The agent then holds a token whose subject is
the caller and whose actor is the agent.

Run it::

    pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
    cp .env.example .env    # then fill it in
    python voice_refund.py priya@acme.com 4411
"""

from __future__ import annotations

import os
import sys

import httpx

from authfi import AuthFI, ConsentDenied, ConsentExpired, WaitTimeout


def refund(caller: str, order: str) -> None:
    """Ask the caller to approve, then call the orders API as them."""
    with AuthFI.from_env() as af:
        agent = af.agent_from_env()
        request = agent.request_user_token(caller, binding_message=f"Refund order {order}")
        print("Asked the caller to approve on their phone.")
        try:
            as_caller = agent.await_user_token(
                request,
                timeout=120,
                resource=os.environ["ORDERS_API_CLIENT_ID"],
                send_to=[os.environ["ORDERS_API_ORIGIN"]],
            )
        except ConsentDenied:
            print("The caller declined.")
            return
        except ConsentExpired:
            print("The request ran out before the caller decided.")
            return
        except WaitTimeout:
            print("Still waiting; ask the caller to check their phone.")
            return
        print(f"Acting for {as_caller.subject} as {as_caller.actor}.")
        with httpx.Client(base_url=os.environ["ORDERS_API_ORIGIN"], auth=as_caller.httpx_auth()) as api:
            api.post(f"/orders/{order}/refund").raise_for_status()
        print("Refunded.")


if __name__ == "__main__":
    refund(sys.argv[1], sys.argv[2])
