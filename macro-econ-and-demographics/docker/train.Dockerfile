# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# No extras are installed, so the notebook and tracking (mlflow) extras stay out of this image;
# twin-train logs that tracking is off and writes the same artifact.
COPY pyproject.toml uv.lock ./
COPY packages/twin_core/pyproject.toml packages/twin_core/
COPY training/pyproject.toml training/
COPY serving/api/pyproject.toml serving/api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package twin-training --no-install-workspace

COPY packages/twin_core packages/twin_core
COPY training training
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package twin-training --no-editable


FROM python:3.13-slim
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

# Mount raw data read-only at /data/raw and an output volume at /artifacts.
ENTRYPOINT ["twin-train", "--data-dir", "/data/raw", "--output-root", "/artifacts"]
