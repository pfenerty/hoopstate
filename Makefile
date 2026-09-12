# hoopstate — single entrypoints. No flox required; uv fetches Python 3.14 if absent.
export UV_PYTHON_DOWNLOADS := automatic

PY := cd python && uv run

.PHONY: check sync lint fmt test rebuild-db

check: lint test          ## lint + tests, the one command CI and humans both run

sync:
	cd python && uv sync --group dev

lint:
	$(PY) ruff check .
	$(PY) ruff format --check .

fmt:
	$(PY) ruff format .
	$(PY) ruff check --fix .

test:
	$(PY) pytest -q

rebuild-db:               ## delete and rebuild the DuckDB file from parquet
	$(PY) python -m hoopstate.db.rebuild
