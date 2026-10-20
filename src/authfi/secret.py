"""``Secret``: a credential that does not print, pickle or serialise by accident.

Client secrets, access, refresh and ID tokens, assertions, private keys and webhook secrets are
held in a :class:`Secret`. Its value comes out only through :meth:`Secret.reveal`, a name that
shows up in review. Everything else (``repr``, ``str``, f-strings, logging, ``json.dumps``,
``pickle``) sees ``Secret('…')`` or fails loudly.
"""

from __future__ import annotations

import hmac
from typing import Any, NoReturn, final

__all__ = ["REDACTED", "Secret"]

#: What a secret looks like wherever it is printed.
REDACTED = "Secret('…')"


@final
class Secret:
    """A secret value. Read it with :meth:`reveal`; nothing else shows it.

    - ``repr``, ``str`` and ``format`` give ``Secret('…')``;
    - ``json.dumps`` raises ``TypeError`` (it is not a JSON value);
    - ``pickle`` raises ``TypeError``: a checkpoint stores an id, never a credential;
    - ``copy`` and ``deepcopy`` return the same object, so a framework that copies its
      configuration does not fail;
    - equality compares in constant time.
    """

    __slots__ = ("_value",)

    _value: str

    def __init__(self, value: str | bytes | Secret) -> None:
        if isinstance(value, Secret):
            text = value.reveal()
        elif isinstance(value, bytes):
            text = value.decode("utf-8")
        elif isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            text = value
        else:  # a caller without a type checker
            raise TypeError(f"Secret holds a str or bytes value, not {type(value).__name__}")
        object.__setattr__(self, "_value", text)

    def reveal(self) -> str:
        """The value. Every call site is a place a credential leaves its protection: keep it short-lived."""
        return self._value

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("Secret is immutable")

    def __repr__(self) -> str:
        return REDACTED

    __str__ = __repr__

    def __format__(self, format_spec: str) -> str:
        return REDACTED

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Secret):
            return NotImplemented
        return hmac.compare_digest(self._value.encode("utf-8"), other._value.encode("utf-8"))

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    # A hash would make a secret a dictionary key, which puts it in more places than it belongs.
    __hash__ = None  # type: ignore[assignment]

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "A Secret cannot be pickled: a checkpoint, cache or queue would store the credential. "
            "Store an identifier and fetch the credential again where it is used."
        )

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        self.__reduce__()

    def __copy__(self) -> Secret:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Secret:
        return self
