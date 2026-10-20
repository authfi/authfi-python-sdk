from __future__ import annotations

import copy
import io
import json
import logging
import pickle
from collections.abc import Callable

import pytest

from authfi import Secret
from authfi._log import RedactingFilter, get_logger, redact

MARKER = "MARKER-s3cr3t-VALUE"


def test_reveal_returns_the_value() -> None:
    assert Secret(MARKER).reveal() == MARKER
    assert Secret(MARKER.encode()).reveal() == MARKER
    assert Secret(Secret(MARKER)).reveal() == MARKER


def _fstring(s: Secret) -> str:
    return f"{s}"


def _fstring_repr(s: Secret) -> str:
    return f"{s!r}"


def _percent(s: Secret) -> str:
    return "%s" % s  # noqa: UP031 - the old style is what this checks


SHOWS: list[Callable[[Secret], str]] = [repr, str, _fstring, _fstring_repr, _percent, format]


@pytest.mark.parametrize("show", SHOWS)
def test_no_ordinary_way_of_showing_it_reveals_it(show: Callable[[Secret], str]) -> None:
    text = show(Secret(MARKER))
    assert MARKER not in text
    assert text == "Secret('…')"


def test_json_and_pickle_refuse_it() -> None:
    with pytest.raises(TypeError):
        json.dumps({"s": Secret(MARKER)})
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(Secret(MARKER))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        with pytest.raises(TypeError):
            pickle.dumps([Secret(MARKER)], protocol=protocol)


def test_copies_are_the_same_object_so_framework_config_copies_do_not_fail() -> None:
    s = Secret(MARKER)
    assert copy.copy(s) is s
    assert copy.deepcopy({"k": s})["k"] is s


def test_it_has_no_dict_and_cannot_be_changed() -> None:
    s = Secret(MARKER)
    assert not hasattr(s, "__dict__")
    with pytest.raises(AttributeError):
        s._value = "x"
    assert vars(type(s)).get("__dict__") is None


def test_equality_and_no_hash() -> None:
    assert Secret("a") == Secret("a")
    assert Secret("a") != Secret("b")
    assert (Secret("a") == "a") is False
    with pytest.raises(TypeError):
        hash(Secret("a"))


def test_truthiness_is_kept_and_the_length_is_not_given_away() -> None:
    assert Secret("abc")
    assert not Secret("")
    with pytest.raises(TypeError):
        len(Secret("abc"))  # type: ignore[arg-type]


def test_a_non_string_is_refused() -> None:
    with pytest.raises(TypeError):
        Secret(12)  # type: ignore[arg-type]


def test_the_authfi_logger_redacts_jwts_and_bearer_credentials() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = get_logger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        jwt_like = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJl"
        logger.warning("sent %s and %s", jwt_like, "Bearer abc.def-ghi")
    finally:
        logger.removeHandler(handler)
    out = stream.getvalue()
    assert "eyJzdWIiOiJ4In0" not in out
    assert "abc.def-ghi" not in out
    assert "[redacted" in out


def test_a_callers_own_logger_gets_the_filter() -> None:
    mine = logging.getLogger("tests.mine")
    assert any(isinstance(f, RedactingFilter) for f in get_logger(mine).filters)


def test_redact_leaves_ordinary_text_alone() -> None:
    assert redact("token endpoint answered 400") == "token endpoint answered 400"
    assert redact("Authorization: Bearer x.y.z") == "Authorization: Bearer [redacted]"
