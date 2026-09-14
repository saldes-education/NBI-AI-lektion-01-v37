# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first so source edits don't invalidate this layer.
COPY pyproject.toml uv.lock ./
COPY packages/twin_core/pyproject.toml packages/twin_core/
COPY training/pyproject.toml training/
COPY serving/api/pyproject.toml serving/api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package twin-api --no-install-workspace

COPY packages/twin_core packages/twin_core
COPY serving/api serving/api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package twin-api --no-editable


FROM python:3.13-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv

# The image is pinned to one artifact; build with --build-arg ARTIFACT_DIR=artifacts/<version>.
ARG ARTIFACT_DIR=artifacts/baseline
COPY ${ARTIFACT_DIR} /app/artifacts/current

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TWIN_ARTIFACT_DIR=/app/artifacts/current
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "twin_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
