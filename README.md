# hoopstate

**Reconstructing the state the box score never records.**

A richer model derived from public NBA play-by-play data.

Raw play-by-play is a flat event log with two structural gaps that block most interesting analysis:

1. **No on-court state.** Substitutions are logged, but "who are the ten players on the floor right
   now" is never stated.
2. **No possession or sequence structure.** Asking how the previous event type affects the next play
   requires possession boundaries and adjacency features that do not exist in the source.

This project derives both, validates them against an independent oracle, and exposes the result as
versioned analysis marts.

## Layout

```
python/src/hoopstate/   ingest/ model/ derive/ validate/ db/
rust/                added at port time — see rust/README.md
docs/research/       findings from the research spikes
tests/fixtures/      golden games as small committed parquet
```

Parquet is the source of truth; DuckDB holds views plus materialized gold marts, so the database is
always rebuildable and safe to delete. Data zones live outside git.

Where those zones physically land is decided by a **storage profile** (`hoopstate.storage`), the one
module that knows which profile is active — everything else asks for a zone path and gets one. The
default `ephemeral` profile puts every zone under a single local scratch directory, so a fresh
checkout runs with zero configuration; the `local` profile splits read-mostly bulk onto a NAS and
keeps the working set (and always the DuckDB file) on fast local disk. Select with
`HOOPSTATE_PROFILE`; override individual tiers with `HOOPSTATE_HOT` / `HOOPSTATE_COLD`.

## Getting started

Only `uv` is required. If no system Python 3.14 is present, uv fetches a standalone one.

```bash
cd python
UV_PYTHON_DOWNLOADS=automatic uv sync --group dev
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

flox is optional and purely a local convenience — `flox activate` gives you the same
environment, but nothing in the build depends on it. CI and cloud sessions use the
commands above verbatim.

## Planning

All work is tracked in beads. `bd ready` shows what is unblocked; the parent epic is
`Rich NBA play-by-play data model`. Each hard problem begins with a research spike that files its
own implementation issues.
