"""An orders API that accepts AuthFI access tokens (FastAPI).

Run it::

    pip install "authfi[fastapi] @ git+https://github.com/authfi/authfi-python-sdk@main" uvicorn
    cp .env.example .env    # then fill it in
    uvicorn orders_api:app

Call it with a token minted for this API (see ../machine-to-machine).
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI

from authfi import AccessToken, AsyncAuthFI
from authfi.fastapi import AuthFIBearer

af = AsyncAuthFI.from_env()  # AUTHFI_ISSUER
auth = AuthFIBearer(
    af.verifier(
        # This API's client id. Machine and agent tokens name it when they are minted with
        # resource=<it>. Add the client ids of your own front ends to accept your people's tokens.
        audience=[os.environ["ORDERS_API_CLIENT_ID"]],
        expected_tenant_id=os.environ.get("AUTHFI_TENANT_ID"),
    ),
    resource_metadata=os.environ.get("ORDERS_API_RESOURCE_METADATA"),
)

app = FastAPI(title="Orders")
auth.install(app)


@app.get("/orders")
async def list_orders(token: AccessToken = Depends(auth.require(permissions=["orders:read"]))) -> dict[str, object]:  # noqa: B008
    """Anyone holding the ``orders:read`` permission."""
    return {"orders": [], "for": str(token.principal)}


@app.post("/orders/{order_id}/refund")
async def refund(
    order_id: str,
    token: AccessToken = Depends(auth.require(permissions=["orders:refund"])),  # noqa: B008
) -> dict[str, object]:
    """A refund needs ``orders:refund`` and, for a person, a recent strong sign-in."""
    if token.token_kind == "person":
        token.require_acr("urn:authfi:acr:mfa", "urn:authfi:acr:passkey_mfa", max_age=600)
    acting = f"{token.actor} for {token.principal}" if token.actor else str(token.principal)
    return {"refunded": order_id, "by": acting}
