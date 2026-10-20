import io
import re

from setuptools import setup

# ONE spelling of the version. authfi.py needs it at runtime (it goes in the
# User-Agent, which AuthFI's Cloudflare edge requires), and a copy here would be
# a second place to forget.
with io.open("authfi.py", encoding="utf-8") as fh:
    VERSION = re.search(r'^__version__ = "([^"]+)"', fh.read(), re.M).group(1)

setup(
    name="authfi",
    version=VERSION,
    description="AuthFI Python SDK — JWT validation, RBAC decorators, permission auto-sync, agent authentication",
    py_modules=["authfi"],
    python_requires=">=3.8",
    install_requires=[
        "PyJWT[crypto]>=2.4.0",
        # Ed25519 signing for agent client assertions (private_key_jwt). This is
        # NOT a new install: PyJWT[crypto] above already pulls cryptography in for
        # RS256 verification. It is declared because authfi.py imports it directly,
        # and a direct import that relies on somebody else's extra breaks silently
        # the day that extra changes.
        "cryptography>=3.4.0",
    ],
    extras_require={
        # AgentAuth.httpx_auth() and DelegatedToken.httpx_auth(). Optional: the
        # rest of the SDK, the async client included, does not use it.
        "httpx": ["httpx>=0.23"],
    },
    author="AuthFI",
    url="https://github.com/authfi/authfi-python-sdk",
    license="Apache-2.0",
    classifiers=[
        "Framework :: Flask",
        "Framework :: FastAPI",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
)
