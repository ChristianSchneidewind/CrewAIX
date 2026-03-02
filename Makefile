lint:
	uv run ruff check .
	uv run black --check .
	uv run mypy src

format:
	uv run ruff check . --fix
	uv run black .

.PHONY: lint format
