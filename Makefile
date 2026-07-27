.PHONY: sync test lint typecheck doctor

sync:
	uv sync --locked --extra all

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

doctor:
	uv run cog-surp doctor
