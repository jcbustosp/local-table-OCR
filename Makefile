.PHONY: format lint typecheck test check

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

typecheck:
	uv run pyright

test:
	uv run pytest

check: format lint typecheck test