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
docs/disk-budget.md  the local footprint budget, checked by `python -m hoopstate.footprint`
tests/fixtures/      golden games as small committed parquet
```

Parquet is the source of truth; DuckDB holds views plus materialized gold marts, so the database is
always rebuildable and safe to delete. Data zones live outside git.

The catalog follows one convention: **a DuckDB schema per zone, a view per dataset**, so
`bronze/season=2023/nbastats_2023.parquet` is queried as `SELECT * FROM bronze.nbastats WHERE season
= 2023`, and silver and gold tables as `silver.canonical_event`, `gold.<mart>`. Bronze views glob
across seasons and restore the partition key as a `season` column, so a newly ingested season needs
no rebuild. One command builds the database, and `--list` shows what it would create without
touching it:

```
python -m hoopstate.db.catalog          # rebuild from parquet
python -m hoopstate.db.catalog --list   # show the planned views
```

Where those zones physically land is decided by a **storage profile** (`hoopstate.storage`), the one
module that knows which profile is active — everything else asks for a zone path and gets one. The
default `ephemeral` profile puts every zone under a single local scratch directory, so a fresh
checkout runs with zero configuration; the `local` profile puts them under
`~/hoopstate/data`, where they survive between sessions. Select with `HOOPSTATE_PROFILE`;
override the root with `HOOPSTATE_ROOT`. That root must be on local disk — DuckDB does not
support database files on network filesystems.

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
