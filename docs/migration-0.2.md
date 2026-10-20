# Moving from 0.1 to 0.2

0.2 rebuilds the SDK on one core shared by every area. Nothing was published to PyPI, and the
changes below are the ones you meet coming from `main`'s single `authfi.py` module.

## Python 3.11 or later

0.1 declared Python 3.8. 0.2 needs 3.11: 3.8 and 3.9 are past end of life and 3.10 reaches it
this month. pip refuses the install on an older Python and names the version.

## Configure by issuer

The project is configured by its address, its issuer, and nothing else:

```python
# 0.1
auth = AuthFI(tenant="acme", api_key="sk_live_...")

# 0.2
af = AuthFI("https://acme.authfi.io")       # or AuthFI.from_env(): AUTHFI_ISSUER
```

`api_key` is no longer used: a project is named by its issuer, and each credential comes from
the client or agent you register in the console. The base domain, the API host and every endpoint
are read from the project's discovery document.

## Package layout

`authfi` is now a package. The names `main` exported are kept at the top level through the 0.x
series, with deprecation warnings where they changed; this guide lists each one as the area that
replaces it lands.
