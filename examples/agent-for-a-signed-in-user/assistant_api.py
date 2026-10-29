"""An assistant inside your app acts for the person who is signed in to it (token exchange).

The person calls your API with their own access token. The assistant exchanges it for a token
whose subject is the person and whose actor is the assistant, so the orders API (and AuthFI's
records) see both.

Run it::

    pip install "authfi[fastapi] @ git+https://github.com/authfi/authfi-python-sdk@main" uvicorn
    cp .env.example .env    # then fill it in
    uvicorn assistant_api:app
"""

from __future__ import annotations

import os

import httpx
from fastapi import Depends, FastAPI, Request

from authfi import AccessToken, AsyncAuthFI, Secret
from authfi.fastapi import AuthFIBearer

af = AsyncAuthFI.from_env()  # AUTHFI_ISSUER
agent = af.agent_from_env()  # the assistant: AUTHFI_AGENT_ID, AUTHFI_AGENT_KEY_FILE, AUTHFI_AGENT_KID
auth = AuthFIBearer(af.verifier(audience=[os.environ["APP_CLIENT_ID"]]))
app = FastAPI(title="Assistant")
auth.install(app)


@app.post("/assistant/refund/{order_id}")
async def refund_for_me(
    order_id: str,
    request: Request,
    person: AccessToken = Depends(auth.require()),  # noqa: B008
) -> dict[str, str]:
    """The signed-in person asks the assistant to refund one of their orders."""
    raw = request.headers["authorization"].split(" ", 1)[1]
    as_them = await agent.exchange(
        Secret(raw),
        resource=os.environ["ORDERS_API_CLIENT_ID"],
        scope=["orders:refund"],
        send_to=[os.environ["ORDERS_API_ORIGIN"]],
    )
    async with httpx.AsyncClient(base_url=os.environ["ORDERS_API_ORIGIN"], auth=as_them.httpx_auth()) as api:
        answer = await api.post(f"/orders/{order_id}/refund")
        answer.raise_for_status()
    return {"refunded": order_id, "for": str(person.principal), "by": str(as_them.actor)}
