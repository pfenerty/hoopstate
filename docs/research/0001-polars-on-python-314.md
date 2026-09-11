# polars on Python 3.14

**Status:** resolved, verified empirically on this machine
**Affects:** stack choice (E1 · Foundation)

## The apparent problem

A naive PyPI check suggests polars supports Python 3.10–3.13 and stops there:

- polars' trove classifiers list `Programming Language :: Python :: 3.10` through `3.13`, with no 3.14.
- Filtering polars' published wheels for a `cp314` tag returns nothing. So does filtering for `cp313`.

The second result is the tell. polars contains compiled Rust, so a release that publishes
*zero* CPython-tagged wheels for *any* version cannot be right.

## What is actually going on

polars split its packaging. The `polars` distribution is now a **pure-Python shim**
(`polars-1.44.2-py3-none-any.whl`, `requires_python >= 3.10`). Its `requires_dist` pulls in
`polars-runtime-32==1.44.2`, and *that* is the distribution carrying the compiled Rust.

`polars-runtime-32` ships wheels tagged **`cp310-abi3`**. `abi3` is CPython's stable ABI: a
`cp310-abi3` wheel is forward-compatible with every CPython from 3.10 onward. The tag names the
*minimum* version and never the actual one, which is precisely why grepping for `cp314` — or
`cp313` — finds nothing.

## Verification

Installed into a 3.14 venv and confirmed on disk:

```
.venv/lib/python3.14/site-packages/_polars_runtime_32/_polars_runtime.abi3.so
Tag: cp310-abi3-macosx_11_0_arm64      (polars_runtime_32-1.44.2.dist-info/WHEEL)
```

Resolved versions: CPython 3.14.7 · polars 1.44.2 · duckdb 1.5.5 · pyarrow 25.0.1.

`python/tests/test_smoke_polars_314.py` exercises the operations the pipeline actually depends on
— DataFrame construction and dtypes, `group_by`/`agg`, `shift().over()` (the primitive the entire
`event_context` table is built on), `cum_sum().over()`, a parquet round trip, and `scan_parquet`
laziness. All pass. polars → arrow → duckdb and duckdb → polars interop also verified.

## Residual risk and fallback

abi3 guarantees the *binary interface* is stable; it does not guarantee the vendor tested against
3.14. polars' classifiers stop at 3.13, so 3.14 here is **installable-and-verified-working rather
than vendor-supported**. If a subtle incompatibility surfaces later, the fallback is **Python
3.13** (`python313@python3-3.13.x` is in the flox catalog). Nothing in the design depends on any
3.14-specific feature, so that fallback is a one-line manifest change.

The smoke test is the tripwire: it stays in the default test run so a regression shows up as a
test failure rather than as corrupt derived data.
