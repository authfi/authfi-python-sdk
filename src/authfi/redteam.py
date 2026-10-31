"""``python -m authfi redteam``: the platform's red-team corpus, run against your own agent's configuration.

The corpus is the one the AuthFI gateway is tested with (``_redteam/corpus.json``). The cases that can be run
safely from outside are edits of your agent's approved configuration (its system prompt, its tool definitions,
its model), sent to the AI gateway on your project as the agent's runtime would send them. The gateway either
refuses an edit or lets it through. What happened is read from your project's own call tape, not from the
gateway's answer to the request and not from the agent.

::

    python -m authfi redteam --origin https://acme.authfi.io --baseline agent.json            # dry run
    python -m authfi redteam --origin https://acme.authfi.io --baseline agent.json --send     # sends

A dry run prints what would be sent, makes no network call and needs no credential. ``--send`` needs the
agent's own credential (``AUTHFI_ISSUER``, ``AUTHFI_AGENT_ID``, ``AUTHFI_AGENT_KEY_FILE``), the agent's signed
configuration in ``AUTHFI_AGENT_CONFIG``, and to read the tape a console bearer in ``AUTHFI_TAPE_TOKEN`` of a
person who may read the audit tape (``read:audit``). Everything goes to the one origin given, which must be the
project the credential belongs to. The exit code is 0 when every attack was stopped, 1 when one got through or
could not be read from the tape, and 2 when the run was refused before anything was sent. A 0 does not mean the
agent is safe (see :data:`LIMITS`).

``--baseline`` is a JSON file with the configuration the gateway has approved for the agent::

    {"system": "You are ...", "tools": [{"type": "function", "function": {"name": "...", ...}}],
     "models": ["<upstream>/<model>"]}

Use a test agent: a refused attack leaves nothing behind, but a request that gets through is a real model call.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
import jwt

from authfi._protocol.ids import request_id
from authfi._protocol.urls import origin_of, validate_origin
from authfi.client import AuthFI
from authfi.errors import AuthFIError, ConfigurationError

if TYPE_CHECKING:
    from authfi.agents import AgentAuth

__all__ = [
    "LIMITS",
    "Baseline",
    "Case",
    "RedTeamError",
    "Report",
    "Result",
    "add_arguments",
    "build",
    "command",
    "corpus_baseline",
    "dry_run",
    "load_baseline",
    "plan",
    "render",
    "run",
]

#: What a run does not show, printed under every report.
LIMITS: tuple[str, ...] = (
    "A run where nothing got through says these attacks failed against your project as it is configured now. "
    "It does not say your agent is safe.",
    "It sends edits of your agent's configuration to the AI gateway. It does not test whether your agent follows "
    "instructions hidden in a tool result or a document, and it calls none of your tools.",
    "The attacks are the corpus's fixed text. An attacker adapts; this does not.",
    "The verdicts are your call tape's rows. A request with no row proves nothing either way.",
)

_PATHS = {"chat": "/mcp/v1/chat/completions", "responses": "/mcp/v1/responses", "messages": "/mcp/v1/messages"}
_CONFIG_HEADER = "Authfi-Agent-Config"
_BASELINE_CONTROL = "control-baseline"
_CODE = re.compile(r"\bE[A-Z]{2,}\d{3}\b")
_TAPE_QUERY = (
    "query($requestId: String, $first: Int) { mcpCallEvents(requestId: $requestId, first: $first) "
    "{ verdict reason tool agent requestId at } }"
)

STOPPED = "stopped"
STOPPED_ELSEWHERE = "stopped_elsewhere"
HELD = "held"
GOT_THROUGH = "got_through"
ADMITTED = "admitted"
CONTROL_FAILED = "control_failed"
NO_TAPE_ROW = "no_tape_row"
NOT_RUN = "not_run"
WOULD_SEND = "would_send"
_BAD = frozenset({GOT_THROUGH, CONTROL_FAILED, NO_TAPE_ROW})
_WORDS = {
    STOPPED: "refused",
    STOPPED_ELSEWHERE: "refused by another check",
    HELD: "held for a person",
    GOT_THROUGH: "GOT THROUGH",
    ADMITTED: "admitted (control)",
    CONTROL_FAILED: "CONTROL REFUSED",
    NO_TAPE_ROW: "NOT ON THE TAPE",
    WOULD_SEND: "would send",
}


class RedTeamError(ConfigurationError):
    """The run cannot go on, or must not start."""


@dataclass(frozen=True, slots=True)
class Baseline:
    """The configuration the gateway has approved for the agent."""

    system: str
    tools: tuple[dict[str, Any], ...]
    model: str


@dataclass(frozen=True, slots=True)
class Case:
    """One request to send (``body`` set) or one corpus case that is not run (``skip`` says why)."""

    id: str
    kind: str
    api: str
    expect: str
    control: bool
    edit: dict[str, Any]
    body: dict[str, Any] | None
    skip: str


@dataclass(frozen=True, slots=True)
class Result:
    """What became of one case.

    ``refused_by`` is the code on the tape row; ``detail`` is the row's reason, what a dry run would send, or why
    the case is not run.
    """

    id: str
    kind: str
    status: str
    refused_by: str
    detail: str
    request_id: str


@dataclass(frozen=True, slots=True)
class Report:
    """A run, or a dry run (``sent`` False)."""

    origin: str
    sent: bool
    corpus_cases: int
    results: tuple[Result, ...]

    def count(self, *statuses: str) -> int:
        """How many results have one of ``statuses``."""
        return sum(1 for r in self.results if r.status in statuses)

    @property
    def clean(self) -> bool:
        """``True`` when nothing got through and every request could be read from the tape."""
        return not any(r.status in _BAD for r in self.results)


# --- the data -----------------------------------------------------------------------------------------


def _data(name: str) -> dict[str, Any]:
    text = resources.files("authfi").joinpath("_redteam", name).read_text(encoding="utf-8")
    return cast("dict[str, Any]", json.loads(text))


def _baseline_of(doc: object, what: str) -> Baseline:
    if not isinstance(doc, dict):
        raise RedTeamError(f"{what} must be a JSON object")
    fields = cast("dict[str, Any]", doc)
    system, tools, models = fields.get("system"), fields.get("tools", []), fields.get("models")
    if not isinstance(system, str) or not system:
        raise RedTeamError(
            f"{what}: system must be the agent's system prompt (with none there is nothing to tamper with)"
        )
    tool_list: list[Any] | None = tools if isinstance(tools, list) else None  # pyright: ignore[reportUnknownVariableType]
    if tool_list is None or not all(
        isinstance(t, dict) and isinstance(cast("dict[str, Any]", t).get("function"), dict) for t in tool_list
    ):
        raise RedTeamError(f"{what}: tools must be a list of {{'type': 'function', 'function': {{...}}}} objects")
    model_list: list[Any] | None = models if isinstance(models, list) else None  # pyright: ignore[reportUnknownVariableType]
    if not model_list or not all(isinstance(m, str) and m for m in model_list):
        raise RedTeamError(f"{what}: models must list at least one model, as <upstream>/<model>")
    return Baseline(system, tuple(copy.deepcopy(cast("list[dict[str, Any]]", tool_list))), cast(str, model_list[0]))


def load_baseline(path: str | Path) -> Baseline:
    """The baseline in the JSON file at ``path``."""
    try:
        doc: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RedTeamError(f"cannot read the baseline {path}: {error}") from error
    return _baseline_of(doc, str(path))


def corpus_baseline() -> Baseline:
    """The corpus's own example configuration, which a dry run uses when no baseline is given."""
    return _baseline_of(_data("corpus.json")["baseline"], "the corpus baseline")


# --- building the requests ----------------------------------------------------------------------------


class _Inapplicable(Exception):
    """An edit that cannot be made on this baseline."""


def _base_body(b: Baseline, api: str) -> dict[str, Any]:
    if api == "responses":
        return {"model": b.model, "instructions": b.system, "input": "hi"}
    if api == "messages":
        return {"model": b.model, "system": b.system, "messages": [{"role": "user", "content": "hi"}]}
    return {
        "model": b.model,
        "messages": [{"role": "system", "content": b.system}, {"role": "user", "content": "refund order 4417"}],
        "tools": copy.deepcopy(list(b.tools)),
    }


def _system(body: dict[str, Any], api: str) -> str:
    if api == "responses":
        return cast(str, body["instructions"])
    if api == "messages":
        return cast(str, body["system"])
    return cast(str, body["messages"][0]["content"])


def _set_system(body: dict[str, Any], api: str, text: str) -> None:
    if api == "responses":
        body["instructions"] = text
    elif api == "messages":
        body["system"] = text
    else:
        body["messages"][0]["content"] = text


def _first_tool(body: dict[str, Any]) -> dict[str, Any]:
    tools = cast("list[dict[str, Any]]", body.get("tools") or [])
    if not tools:
        raise _Inapplicable("your baseline declares no tools")
    return cast("dict[str, Any]", tools[0]["function"])


def _apply(body: dict[str, Any], api: str, verb: str, arg: Any) -> None:
    if verb == "system_append":
        _set_system(body, api, _system(body, api) + arg)
    elif verb == "system_set":
        _set_system(body, api, arg)
    elif verb == "system_lower":
        _set_system(body, api, _system(body, api).lower())
    elif verb == "system_drop":
        del body["messages"][0]
    elif verb == "system_extra":
        body["messages"].insert(1, {"role": "system", "content": arg})
    elif verb == "message_add":
        body["messages"].append(dict(arg))
    elif verb == "field_add":
        body.update(arg)
    elif verb == "model_set":
        body["model"] = arg
    elif verb == "tools_drop":
        body["tools"] = []
    elif verb == "tool_add":
        body["tools"].append({"type": "function", "function": dict(arg)})
    elif verb == "tool_description_append":
        function = _first_tool(body)
        function["description"] = str(function.get("description", "")) + arg
    elif verb == "tool_property_add":
        parameters = _first_tool(body).setdefault("parameters", {"type": "object"})
        parameters.setdefault("properties", {}).update(arg)
    elif verb == "tool_name_suffix":
        function = _first_tool(body)
        function["name"] = str(function["name"]) + arg
    elif verb == "tool_name_set":
        _first_tool(body)["name"] = arg
    else:  # pragma: no cover - the plan is package data, and the tests build every edit in it
        raise RedTeamError(f"the plan names an edit this SDK does not know: {verb}")


def build(b: Baseline, api: str, edit: Mapping[str, Any]) -> dict[str, Any]:
    """The request body that is ``edit`` made to ``b``'s own request in the ``api`` (chat, responses or messages)."""
    body = _base_body(b, api)
    for verb, arg in edit.items():
        _apply(body, api, verb, arg)
    return body


def plan(baseline: Baseline) -> list[Case]:
    """Every corpus case in the order they run, after the baseline control.

    A case with a ``body`` is sent; one without says in ``skip`` why it is not.
    """
    corpus, spec = _data("corpus.json"), _data("plan.json")
    edits = cast("dict[str, dict[str, Any]]", spec["edits"])
    controls = frozenset(cast("list[str]", spec["controls"]))
    kind_reasons = cast("dict[str, str]", spec["not_run"]["kinds"])
    id_reasons = cast("dict[str, str]", spec["not_run"]["ids"])
    cases = [Case(_BASELINE_CONTROL, "baseline_control", "chat", "", True, {}, _base_body(baseline, "chat"), "")]
    for raw in cast("list[dict[str, Any]]", corpus["cases"]):
        cid, kind, api = cast(str, raw["id"]), cast(str, raw["kind"]), cast(str, raw.get("api") or "chat")
        if cid not in edits:
            reason = id_reasons.get(cid) or kind_reasons[kind]
            if kind == "call":
                reason += f" The gateway's matrix expects: {raw['action']} (class {raw['cls'] or 'undeclared'})."
            cases.append(Case(cid, kind, api, "", False, {}, None, reason))
            continue
        edit, control = edits[cid], cid in controls
        expect = "" if control else cast(str, raw["code"])
        try:
            body: dict[str, Any] | None = build(baseline, api, edit)
            skip = "the edit changes nothing on your baseline" if body == _base_body(baseline, api) else ""
        except _Inapplicable as why:
            body, skip = None, str(why)
        cases.append(Case(cid, kind, api, expect, control, edit, None if skip else body, skip))
    return cases


# --- the evidence: the project's call tape ------------------------------------------------------------


class _Tape:
    """Reads the project's own call tape (``mcpCallEvents``) with a person's console bearer."""

    def __init__(self, origin: str, token: str, client: httpx.Client) -> None:
        self._url = origin + "/graphql"
        self._token = token
        self._client = client

    def events(self, rid: str | None, first: int = 5) -> list[dict[str, Any]]:
        try:
            response = self._client.post(
                self._url,
                json={"query": _TAPE_QUERY, "variables": {"requestId": rid, "first": first}},
                headers={"Authorization": "Bearer " + self._token},
            )
        except httpx.HTTPError as error:
            raise RedTeamError(f"cannot reach the call tape at {self._url}: {type(error).__name__}") from error
        if response.status_code in (401, 403):
            raise RedTeamError(
                f"the call tape refused the token (HTTP {response.status_code}): it must be a console bearer of a "
                "person who may read the audit tape (read:audit) on this project"
            )
        try:
            document = cast("dict[str, Any]", response.json())
        except ValueError:
            document = {}
        errors = cast("list[dict[str, Any]]", document.get("errors") or [])
        if response.status_code != 200 or errors:
            what = errors[0].get("message") if errors else f"HTTP {response.status_code}"
            raise RedTeamError(f"the call tape could not be read: {what}")
        data = cast("dict[str, Any]", document.get("data") or {})
        return cast("list[dict[str, Any]]", data.get("mcpCallEvents") or [])


def _classify(case: Case, rows: Sequence[Mapping[str, Any]]) -> tuple[str, str, str]:
    """(status, refused_by, reason) for one request, from the tape row for its request id."""
    if not rows:
        return NO_TAPE_ROW, "", "no row for this request id on the tape"
    verdict, reason = str(rows[0].get("verdict", "")), str(rows[0].get("reason", ""))
    found = _CODE.search(reason)
    code = found.group(0) if found else ""
    if verdict == "allow":
        return (ADMITTED if case.control else GOT_THROUGH), "", reason
    if verdict not in ("deny", "approval_required"):
        return NO_TAPE_ROW, "", f"the row's verdict is {verdict!r}, which this command does not know"
    if case.control:
        return CONTROL_FAILED, code, reason
    if verdict == "approval_required":
        return HELD, code, reason
    return (STOPPED if code == case.expect else STOPPED_ELSEWHERE), code, reason


# --- running ------------------------------------------------------------------------------------------


def _capped(body: dict[str, Any], api: str) -> dict[str, Any]:
    """``body`` with an answer cap of a few tokens: the answer is never read."""
    out = copy.deepcopy(body)
    if api == "responses":
        out["max_output_tokens"] = 16
    else:
        out["max_tokens"] = 1
    return out


def _corpus_size() -> int:
    return len(cast("list[Any]", _data("corpus.json")["cases"]))


def run(
    origin: str,
    baseline: Baseline,
    *,
    agent: AgentAuth,
    agent_config: str,
    tape_token: str,
    tries: int = 8,
    sleep: Callable[[float], None] = time.sleep,
    tape_client: httpx.Client | None = None,
) -> Report:
    """Send the cases to ``origin`` as ``agent`` and read what the gateway did from the call tape.

    ``origin`` must be the agent's own project. The first request is the baseline itself: if the gateway does
    not admit it, nothing else is sent, because every refusal after it would mean nothing.
    """
    origin = validate_origin(origin, "origin")
    if origin_of(agent.project.issuer) != origin:
        raise RedTeamError(
            f"the agent belongs to {agent.project.issuer}, not {origin}: this command only runs with the credential "
            "of the project it is pointed at"
        )
    reader = tape_client or httpx.Client(follow_redirects=False, timeout=30.0)
    try:
        tape = _Tape(origin, tape_token, reader)
        tape.events(None, 1)  # before anything is sent: with a tape that cannot be read every verdict is unknown
        sender = agent.gateway().http_client()
        results: list[Result] = []
        halted = ""
        for case in plan(baseline):
            if case.body is None or halted:
                results.append(Result(case.id, case.kind, NOT_RUN, "", halted or case.skip, ""))
                continue
            rid = request_id()
            try:
                response = sender.post(
                    origin + _PATHS[case.api],
                    json=_capped(case.body, case.api),
                    headers={"X-Request-Id": rid, _CONFIG_HEADER: agent_config},
                )
            except httpx.HTTPError as error:
                raise RedTeamError(f"cannot reach the gateway at {origin}: {type(error).__name__}") from error
            if response.status_code == 401:
                raise RedTeamError(
                    "the gateway refused the agent's token (HTTP 401): check the credential and that the agent "
                    "is active"
                )
            rows: list[dict[str, Any]] = []
            for attempt in range(tries):
                rows = tape.events(rid)
                if rows:
                    break
                if attempt + 1 < tries:
                    sleep(0.5)
            status, code, reason = _classify(case, rows)
            results.append(Result(case.id, case.kind, status, code, reason, rid))
            if case.control and case.id == _BASELINE_CONTROL and status != ADMITTED:
                halted = "not run: the baseline itself was not admitted, so no refusal after it would mean anything"
    finally:
        if tape_client is None:
            reader.close()
    return Report(origin, True, _corpus_size(), tuple(results))


def dry_run(origin: str, baseline: Baseline) -> Report:
    """What :func:`run` would send, without a credential and without a network call."""
    origin = validate_origin(origin, "origin")
    results: list[Result] = []
    for case in plan(baseline):
        if case.body is None:
            results.append(Result(case.id, case.kind, NOT_RUN, "", case.skip, ""))
        else:
            what = (
                f"POST {_PATHS[case.api]} {json.dumps(case.edit, ensure_ascii=True) if case.edit else '(the baseline)'}"
            )
            results.append(Result(case.id, case.kind, WOULD_SEND, "", what, ""))
    return Report(origin, False, _corpus_size(), tuple(results))


# --- the report ---------------------------------------------------------------------------------------


def render(report: Report) -> str:
    """The report as text: attack, what happened, refused by what, and which got through."""
    ran = [r for r in report.results if r.status != NOT_RUN]
    skipped = [r for r in report.results if r.status == NOT_RUN]
    lines = [
        f"AuthFI red team against {report.origin}: {len(ran)} requests sent, read back from your call tape."
        if report.sent
        else f"DRY RUN against {report.origin}: nothing is sent. Add --send to send the {len(ran)} requests below.",
        f"{len(ran) - 1} of the corpus's {report.corpus_cases} cases can be run this way (plus a control on your "
        f"baseline); the other {len(skipped)} are listed below.",
        "",
    ]
    width = max(len(r.id) for r in ran)
    lines.append(f"{'CASE':<{width}}  {'WHAT HAPPENED':<25}  REFUSED BY / DETAIL")
    for r in ran:
        if r.status in (STOPPED, STOPPED_ELSEWHERE, HELD):
            detail = f"{r.refused_by or '-'}  {r.detail}"
        else:
            detail = r.detail if r.status == WOULD_SEND else (r.refused_by or "-")
        lines.append(f"{r.id:<{width}}  {_WORDS[r.status]:<25}  {detail}")
    if report.sent:
        lines += [
            "",
            f"{report.count(STOPPED)} refused by the check they target, {report.count(STOPPED_ELSEWHERE)} refused by "
            f"another check, {report.count(HELD)} held, {report.count(GOT_THROUGH)} got through, "
            f"{report.count(NO_TAPE_ROW)} not on the tape, {report.count(CONTROL_FAILED)} controls refused.",
        ]
        lines += [f"  got through: {r.id}" for r in ran if r.status == GOT_THROUGH]
        lines += [
            f"  {r.id}: refused by {r.refused_by or 'another check'}, so the check it targets was not reached"
            for r in ran
            if r.status == STOPPED_ELSEWHERE
        ]
    lines += ["", f"NOT RUN ({len(skipped)}):"]
    groups: dict[str, list[str]] = {}
    for r in skipped:
        groups.setdefault(r.detail.split(" The gateway's matrix")[0], []).append(r.id)
    for reason, ids in groups.items():
        more = f", and {len(ids) - 3} more" if len(ids) > 3 else ""
        lines.append(f"  {len(ids)} ({', '.join(ids[:3])}{more}): {reason}")
    lines += ["", "WHAT THIS DOES NOT SHOW:"] + [f"  - {text}" for text in LIMITS]
    return "\n".join(lines)


# --- the command --------------------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``redteam`` command's options."""
    parser.add_argument("--origin", required=True, help="your project's origin, such as https://acme.authfi.io")
    parser.add_argument("--baseline", help="JSON file: the agent's approved system prompt, tools and models")
    parser.add_argument(
        "--send", action="store_true", help="send the requests (without it, print them and send nothing)"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--agent-config-env", default="AUTHFI_AGENT_CONFIG", help="variable holding the signed configuration"
    )
    parser.add_argument(
        "--tape-token-env", default="AUTHFI_TAPE_TOKEN", help="variable holding the console bearer for the tape"
    )


def command(
    args: argparse.Namespace,
    env: Mapping[str, str],
    out: Callable[[str], object],
    err: Callable[[str], object],
    *,
    project_factory: Callable[[Mapping[str, str]], AuthFI] = AuthFI.from_env,
) -> int:
    """Run the command for parsed ``args``; returns the exit code."""
    try:
        origin = validate_origin(args.origin, "--origin")
        if args.send:
            if not args.baseline:
                raise RedTeamError(
                    "--send needs --baseline: the attacks are edits of your agent's approved configuration"
                )
            report = _send(origin, load_baseline(args.baseline), args, env, project_factory)
        else:
            report = dry_run(origin, load_baseline(args.baseline) if args.baseline else corpus_baseline())
    except AuthFIError as error:
        err(f"authfi redteam: {error.message}")
        return 2 if isinstance(error, ConfigurationError) else 1
    out(json.dumps({**asdict(report), "limits": list(LIMITS)}, indent=2) if args.json else render(report))
    return 0 if report.clean else 1


def _send(
    origin: str,
    baseline: Baseline,
    args: argparse.Namespace,
    env: Mapping[str, str],
    project_factory: Callable[[Mapping[str, str]], AuthFI],
) -> Report:
    config, tape_token = env.get(args.agent_config_env, "").strip(), env.get(args.tape_token_env, "").strip()
    if not config:
        raise RedTeamError(f"{args.agent_config_env} is empty: it holds the agent's signed configuration")
    if not tape_token:
        raise RedTeamError(f"{args.tape_token_env} is empty: nothing is sent when the tape cannot be read")
    _same_project(tape_token, origin, args.tape_token_env)
    with project_factory(env) as project:
        return run(origin, baseline, agent=project.agent_from_env(env), agent_config=config, tape_token=tape_token)


def _same_project(token: str, origin: str, name: str) -> None:
    """Refuse a token that names another project.

    The claim is read unverified: this guards against pasting the wrong project's token, and the call tape is
    what authenticates it.
    """
    try:
        issuer = jwt.decode(token, options={"verify_signature": False}).get("iss")
    except jwt.PyJWTError as error:
        raise RedTeamError(f"{name} is not a token") from error
    if not isinstance(issuer, str) or origin_of(issuer) != origin:
        raise RedTeamError(f"{name} was not issued by {origin}: it must be a token of the project you point this at")
