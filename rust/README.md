# Rust track

Deliberately empty for now. The repo is laid out as a polyglot workspace from day one so that the
Rust port is an *addition* rather than a restructuring — but no crates exist until the Python model
is settled and oracle-validated.

The port is planned in beads under **E10 · Rust port**, in this order:

1. Event taxonomy → sum types and exhaustive `match`
2. Possession segmentation → a state machine over owned data
3. Lineup reconstruction → real state management; the 5-player invariant becomes a type
4. Ingestion → `arrow` / `parquet` / `polars` crates
5. Read API → `axum` over gold marts

Each step is gated on its Python counterpart being correct against the `pbpstats` oracle, and is
verified by differential test: Python is the reference, Rust must produce byte-identical parquet.

**Before adding `rust/crates/`, read [`docs/disk-budget.md`](../docs/disk-budget.md) and run `python -m hoopstate.footprint`.** Local free space is tight and a
`target/` directory with these dependencies is the single largest consumer. `[profile.dev] debug = 0`
is not optional here.
