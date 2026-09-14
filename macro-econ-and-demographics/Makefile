.PHONY: install lint format test test-fast train serve notebook docker-api docker-train

install:
	uv sync --all-packages --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

train:
	uv run twin-train

serve:
	uv run uvicorn twin_api.main:app --reload

notebook:
	uv run --directory training/notebooks jupyter lab analysis.ipynb

docker-api:
	docker compose build api

docker-train:
	docker compose --profile train run --rm train
