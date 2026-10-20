"""A nightly job that calls the orders API with its own token (client credentials).

Run it::

    pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
    cp .env.example .env    # then fill it in
    python nightly_job.py
"""

from __future__ import annotations

import os

import httpx

from authfi import AuthFI, Secret


def main() -> None:
    """Fetch the orders as the reporting job."""
    with AuthFI.from_env() as af:  # AUTHFI_ISSUER
        credential = af.client_credentials(
            os.environ["AUTHFI_CLIENT_ID"],
            Secret(os.environ["AUTHFI_CLIENT_SECRET"]),
            resource=os.environ["ORDERS_API_CLIENT_ID"],  # the token's audience: the orders API
            send_to=[os.environ["ORDERS_API_ORIGIN"]],  # and the only place it may be sent
        )
        # The token is minted on the first request, reused, and renewed before it expires.
        with httpx.Client(base_url=os.environ["ORDERS_API_ORIGIN"], auth=credential.httpx_auth()) as api:
            response = api.get("/orders")
            response.raise_for_status()
            print(response.json())


if __name__ == "__main__":
    main()
