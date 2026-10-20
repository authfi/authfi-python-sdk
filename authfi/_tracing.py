"""LangSmith set up the way AuthFI traces: metadata, not content, unless asked."""


def langsmith_metadata_only(capture_content=False, **client_kwargs):
    """Install a LangSmith client that records runs without their content, and return it.

    AuthFI's default is that a trace carries metadata (names, timings, ids, the
    agent id) and not prompts, answers or tool results, which hold people's
    data. LangSmith records all of them by default. The client installed here
    has ``hide_inputs`` and ``hide_outputs`` set unless ``capture_content``;
    every LangSmith integration that uses the global client traces with it.
    ``client_kwargs`` go to ``langsmith.Client`` (``api_url``, ``api_key``...).
    """
    import langsmith

    hide = not capture_content
    client = langsmith.Client(hide_inputs=hide, hide_outputs=hide, **client_kwargs)
    langsmith.configure(client=client)
    return client
