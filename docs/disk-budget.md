# Local disk budget

`hoops-1lg.1.3`. Measured 2026-09-12 on the development machine.

This document is the budget. `python -m hoopstate.footprint` is the check that
measures against it, and it is the thing to run **before** E10 creates
`rust/crates/` — not after the disk fills.

```
cd python
uv run python -m hoopstate.footprint            # exits 1 if the budget does not fit
uv run python -m hoopstate.footprint --no-gate  # same report, always exits 0
```

The check is local-only and deliberately not part of `make check`: it reports a
property of a machine, not of the code. A cloud session gets a fresh 30 GB VM
every time and has nothing to budget.

## The budget

| consumer | budget | where the number comes from |
|---|---:|---|
| data zones (`$HOOPSTATE_ROOT`) | 4.0 GiB | 55 MB/season measured end to end, × 30 seasons, plus silver and gold |
| `rust/target/` | 5.0 GiB | arrow + polars + duckdb build artifacts routinely reach 3–5 GB |
| `~/.rustup` + `~/.cargo` | 1.5 GiB | one toolchain plus the registry and its source checkouts |
| `python/.venv` | 1.5 GiB | polars, duckdb and pyarrow wheels unpacked (363 MiB today) |
| uv cache | 1.0 GiB | wheel and source cache, shared across checkouts |
| **total** | **13.0 GiB** | |

The 55 MB/season figure is not an estimate: one full season ingested to raw and
bronze measures 55.5 MiB (25.3 raw + 30.0 bronze). Thirty seasons is therefore
~1.7 GB, and 4 GiB is that with room for silver and gold on top. **Data is not
the problem here.** Build artifacts are.

## It does not fit

The volume has **11.4 GiB free** (228 GiB total, 94% used). Against 13.0 GiB of
budget, with 382 MiB already occupied by these consumers:

```
budgeted 13.0 GiB against 11.8 GiB available (11.4 GiB free + 382.4 MiB already used)
SHORT BY 1.2 GiB
```

That shortfall is the finding, not a defect in the budget. Three things follow
from it, and E10 should not start until they are agreed:

**`[profile.dev] debug = 0` is non-optional.** `rust/README.md` already states
this. Debug symbols are the single largest line item inside a `target/` of this
shape, and leaving them on is what turns the 3 GB end of the range into the
5 GB end.

**`cargo clean` between ports, not at the end.** E10 ports module by module.
A `target/` that is never cleaned accumulates artifacts for every intermediate
dependency set, and the peak is what has to fit, not the final size.

**`raw/` and `hoopstate.duckdb` are the release valve.** Both are derived and
both come back with one command. When the volume gets tight, these go first —
which is why the check totals reclaimable bytes rather than only used ones.

| what | how it comes back | reclaimable? |
|---|---|---|
| `raw/` | re-fetchable from the source manifest | yes |
| `cache/` | keyed download cache; delete freely | yes |
| `hoopstate.duckdb` | `python -m hoopstate.db.catalog` | yes |
| `rust/target/` | `cargo clean` | yes |
| `python/.venv` | `uv sync` | yes |
| uv cache | `uv cache prune` | yes |
| `bronze/`, `silver/`, `gold/` | recomputed from `raw/` — a cost, not a reclaim | **no** |

If the shortfall has to be closed rather than managed, the honest options are a
smaller season range on this machine, or accepting that `target/` and a full
thirty-season `raw/` do not coexist and treating the check's exit code as the
thing that says which one is currently resident.

## No cache is relocated

The third acceptance criterion is satisfied by not doing it. This issue was
originally framed as moving `RUSTUP_HOME` and `CARGO_HOME` onto a NAS; `hoops-03c`
collapsed the hot/cold storage split to a single local root, so there is no tier
to move them to. **Nothing here relocates a cache onto a network filesystem**,
and nothing should: `HOOPSTATE_ROOT` must point at local disk because the DuckDB
file lives under it and DuckDB does not support database files on network
filesystems. The budget is the answer to a full disk, not relocation.

## Two things that are easy to get wrong twice

Both were verified while building the check, and both produce a plausible wrong
number rather than an error, which is what makes them worth writing down.

**Read `df`'s `Avail` column, not `Used` or `Capacity`.** On macOS `df /`
describes the sealed system snapshot (`/dev/disk3s1s1`: 12Gi used, 51%) while the
data volume `/System/Volumes/Data` reports 162Gi used, 94%. They share one APFS
container, so **`Avail` reads 11Gi on both** and only the used/capacity columns
differ. An earlier reading of this machine as "half empty" came from that
column, not from the free-space figure. The check sidesteps the question by
taking the number from `shutil.disk_usage()` on a real path.

**Measure allocated blocks with inodes deduplicated, not `sum(st_size)`.** uv
hardlinks out of its cache into the virtualenv — 233 such entries in
`python/.venv` on this machine — so summing apparent sizes counts the same bytes
under two different budget lines. Walking with `st_blocks * 512` and skipping a
repeated `(st_dev, st_ino)` reproduces `du -sh` exactly: 362.6 MiB computed
against `du`'s 363M, where naive summation gives 358.4 MiB and agrees with
nothing.
