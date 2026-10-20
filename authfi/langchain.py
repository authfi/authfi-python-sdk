"""LangChain: a Claude model client whose AuthFI token tracers never see.

``pip install 'authfi[langchain]'``

LangChain's documented way to give ``ChatAnthropic`` a bearer token is
``default_headers={"Authorization": "Bearer ..."}``. LangSmith records a
model's constructor arguments in every run (``serialized.kwargs``), so with
tracing on, every model call sends the agent's token, or the token acting for
a person, to the tracing backend. ``extra_headers`` at invoke time is recorded
too. :class:`AuthFIChatAnthropic` takes the token from the agent instead, for
every call, and adds it to the request below the tracer::

    from authfi import AgentAuth
    from authfi.langchain import AuthFIChatAnthropic, metadata_only_tracing

    metadata_only_tracing()                  # once, at startup
    agent = AgentAuth.from_env()
    model = AuthFIChatAnthropic(agent=agent, model="claude-haiku-5-5")

The model talks to the project's AI gateway (``<project>/mcp``), sends the
LangChain run id of each call as ``X-Request-Id`` so a trace joins the
gateway's record, and carries the agent id in the run's metadata.
"""

from functools import cached_property

try:
    from langchain_anthropic import ChatAnthropic
    from pydantic import PrivateAttr
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.langchain needs langchain-anthropic: pip install 'authfi[langchain]'"
    ) from e

from ._tracing import langsmith_metadata_only
from .gateway import credential, model_base_url, request_id_header

# The token rides in each request's headers. The client still needs a credential
# when it is built, or it reads ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN from the
# environment and goes looking for an Anthropic profile on disk. This stands in;
# it is replaced on every request and is not a credential anywhere.
_PLACEHOLDER = "authfi-sets-the-token-per-request"


class AuthFIChatAnthropic(ChatAnthropic):
    """``ChatAnthropic`` on the AuthFI AI gateway, with the token kept out of traces.

    ``agent`` is an :class:`authfi.AgentAuth` or :class:`authfi.AsyncAgentAuth`
    (the agent's own token), an :class:`authfi.DelegatedToken` (acting for a
    person), or a function returning the current token. It is read for every
    request, so one long-lived model object keeps working across token
    refreshes. ``base_url`` defaults to the agent's project gateway. Every other
    argument is ``ChatAnthropic``'s.

    The agent is not a constructor argument LangChain records: it is held in a
    private attribute, so it is absent from ``serialized`` and from
    ``invocation_params``. The token is added to the request inside the client
    call, after the tracer has taken its copy.
    """

    _authfi_credential = PrivateAttr(default=None)

    def __init__(self, *, agent, base_url=None, **kwargs):
        cred = credential(agent)
        kwargs["base_url"] = model_base_url(cred, base_url)
        if cred.agent_id:
            kwargs["metadata"] = dict(kwargs.get("metadata") or {}, authfi_agent_id=cred.agent_id)
        super().__init__(**kwargs)
        self._authfi_credential = cred

    @cached_property
    def _client_params(self):
        params = dict(ChatAnthropic._client_params.func(self))
        params["api_key"] = None
        params["auth_token"] = _PLACEHOLDER
        return params

    def _create(self, payload):
        return super()._create(self._authorized(payload, self._authfi_credential.token()))

    async def _acreate(self, payload):
        return await super()._acreate(self._authorized(payload, await self._authfi_credential.atoken()))

    @staticmethod
    def _authorized(payload, token):
        headers = dict(payload.get("extra_headers") or {})
        headers["Authorization"] = "Bearer " + token
        return dict(payload, extra_headers=headers)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(messages, stop=stop, run_manager=run_manager, **_with_run_id(kwargs, run_manager))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager,
                                        **_with_run_id(kwargs, run_manager))

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._stream(messages, stop=stop, run_manager=run_manager, **_with_run_id(kwargs, run_manager))

    def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._astream(messages, stop=stop, run_manager=run_manager, **_with_run_id(kwargs, run_manager))


def _with_run_id(kwargs, run_manager):
    """Send this call's LangChain run id as X-Request-Id. Added here, below the tracer, like the token."""
    run_id = getattr(run_manager, "run_id", None)
    extra = request_id_header(run_id)
    if not extra:
        return kwargs
    headers = dict(kwargs.get("extra_headers") or {})
    headers.update(extra)
    return dict(kwargs, extra_headers=headers)


def metadata_only_tracing(capture_content=False, **client_kwargs):
    """Have LangSmith record runs without their content, unless ``capture_content``.

    AuthFI's default is that a trace carries metadata (names, timings, ids,
    the agent id) and not prompts, answers or tool results, which hold
    people's data. LangSmith records all of them by default. This installs a
    LangSmith client with ``hide_inputs`` and ``hide_outputs`` set as the one
    LangChain traces with, and returns it. Call it once at startup.
    ``client_kwargs`` go to ``langsmith.Client`` (``api_url``, ``api_key``...).
    """
    return langsmith_metadata_only(capture_content, **client_kwargs)
