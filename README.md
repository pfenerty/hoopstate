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

## Getting started

```bash
flox activate
cd python && uv sync --group dev && uv run pytest
```

## Planning

All work is tracked in beads. `bd ready` shows what is unblocked; the parent epic is
`Rich NBA play-by-play data model`. Each hard problem begins with a research spike that files its
own implementation issues.
