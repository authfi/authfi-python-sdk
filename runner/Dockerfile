# The framework test runner, as the image a bound pod runs (authfi/authfi#304, stream S5).
#
# One image per framework, because the frameworks' dependencies do not resolve together:
#   docker build --build-arg FRAMEWORK=langgraph -f runner/Dockerfile -t authfi/authfi-fw-runner-langgraph:0.1.0 .
# The context is the repository root. The SDK is installed from this checkout, so the image always runs the code at the
# commit it was built from. The framework versions are the ranges in pyproject.toml's extras (the same ones the
# framework runs use on a workstation). The image holds no key and no credential: the pod is given its token, agent id
# and approved configuration at run time, and runner/entrypoint.py refuses to start without them.
FROM python:3.12-slim AS build
ARG FRAMEWORK
# Not claude-agent-sdk: its script stops with BLOCKED on an instance (its apiKeyHelper needs an agent key file), so
# there is nothing for a pod to run until the SDK covers it. Not vercel-ai either: it is Node.
RUN case "$FRAMEWORK" in \
      langgraph)     echo "langgraph,langchain-anthropic,otel" > /extras ;; \
      openai-agents) echo "openai-agents,otel" > /extras ;; \
      google-adk)    echo "adk,otel" > /extras ;; \
      *) echo "unknown FRAMEWORK '$FRAMEWORK'" >&2; exit 2 ;; \
    esac
WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir "/src[$(cat /extras)]" httpx

FROM python:3.12-slim
ARG FRAMEWORK
COPY --from=build /opt/venv /opt/venv
COPY runner/scripts/common /app/common
COPY runner/scripts/${FRAMEWORK} /app/${FRAMEWORK}
COPY runner/entrypoint.py /app/entrypoint.py
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    FW_FRAMEWORK=${FRAMEWORK} \
    FW_OUT_DIR=/tmp/out
USER 65532:65532
WORKDIR /app
ENTRYPOINT ["python", "-B", "-I", "/app/entrypoint.py"]
