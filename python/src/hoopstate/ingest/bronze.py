"""Bronze conversion for play-by-play sources (``hoops-1lg.2.2``).

The bulk loader (:mod:`hoopstate.ingest.bulk_loader`) lands a *lossless string
capture* of each dataset at ``bronze/season=<year>/<name>.parquet`` — every
column read as text, 1:1 with the source CSV, typing deliberately deferred.
This module is that deferred step: it reads the string capture, applies an
explicit, hand-authored per-source schema, and rewrites the **same** bronze
path with typed columns. The typed table is what persists in bronze; the string
form is a transient staging state within the pipeline.

Why the schema is hand-authored rather than inferred:

* **Reproducibility.** Inference depends on which rows a reader happens to see.
  An explicit schema makes the bronze contract a property of the code, not of
  the data sample, so two runs — and two seasons — always agree.
* **Faithful nulls and sentinels.** stats.nba.com encodes "no player" as id
  ``0`` and a tied margin as the literal ``"TIE"``. Columns are typed to
  preserve those exactly: id columns stay integers with ``0`` intact, and
  ``SCOREMARGIN`` stays a string so ``"TIE"`` is not silently dropped.
* **Dirty-data surfacing.** Casts are strict, so a value that does not fit its
  declared type raises here instead of silently becoming null downstream.

Two sources are converted, and both matter for different reasons:

``nbastats``
    The primary event feed from stats.nba.com — the widest, most descriptive
    play-by-play, and the backbone of the canonical model.

``datanba``
    A parallel feed whose ``oftid`` column carries an explicit **offense team
    id** on every event. That single column is the independent cross-check that
    catches the out-of-order events stats.nba.com is known to contain, which is
    why this source is converted alongside the primary one rather than later.

Both feeds cover the same games and join on ``(GAME_ID, EVENTNUM/evt)``; typing
both id columns as integers here keeps that join key consistent across sources.

Every path resolves through :func:`hoopstate.ingest.bulk_loader.bronze_parquet_path`,
so this module, like the loader, never learns which storage profile is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hoopstate.ingest.bulk_loader import bronze_parquet_path, parse_dataset_name
from hoopstate.storage import StorageProfile, resolve_profile

__all__ = [
    "SOURCE_SCHEMAS",
    "BronzeResult",
    "apply_schema",
    "convert_dataset",
    "schema_for_dataset",
]


# --- explicit bronze schemas ------------------------------------------------
#
# Column order mirrors the source CSV header exactly (1:1 with source). Types
# were chosen from a full-season scan of 2023-24: integer columns verified to
# hold only ``^-?\d+$`` in every non-null row, string columns kept where the
# source carries text, sentinels ("TIE"), or formatting ("0 - 2", "12:00") that
# a numeric cast would destroy. Ids stay Int64 (values reach ~1.6e9, past Int32)
# so the same type covers team, player, and game ids across every season.

_NBASTATS_SCHEMA: dict[str, pl.DataType] = {
    "GAME_ID": pl.Int64,
    "EVENTNUM": pl.Int64,
    "EVENTMSGTYPE": pl.Int64,
    "EVENTMSGACTIONTYPE": pl.Int64,
    "PERIOD": pl.Int64,
    "WCTIMESTRING": pl.String,  # wall-clock "7:11 PM"
    "PCTIMESTRING": pl.String,  # game clock "12:00"
    "HOMEDESCRIPTION": pl.String,
    "NEUTRALDESCRIPTION": pl.String,
    "VISITORDESCRIPTION": pl.String,
    "SCORE": pl.String,  # "0 - 2"; blank until the first made basket
    "SCOREMARGIN": pl.String,  # signed int as text, plus the literal "TIE"
    "PERSON1TYPE": pl.Int64,
    "PLAYER1_ID": pl.Int64,  # 0 == no player
    "PLAYER1_NAME": pl.String,
    "PLAYER1_TEAM_ID": pl.Int64,
    "PLAYER1_TEAM_CITY": pl.String,
    "PLAYER1_TEAM_NICKNAME": pl.String,
    "PLAYER1_TEAM_ABBREVIATION": pl.String,
    "PERSON2TYPE": pl.Int64,
    "PLAYER2_ID": pl.Int64,
    "PLAYER2_NAME": pl.String,
    "PLAYER2_TEAM_ID": pl.Int64,
    "PLAYER2_TEAM_CITY": pl.String,
    "PLAYER2_TEAM_NICKNAME": pl.String,
    "PLAYER2_TEAM_ABBREVIATION": pl.String,
    "PERSON3TYPE": pl.Int64,
    "PLAYER3_ID": pl.Int64,
    "PLAYER3_NAME": pl.String,
    "PLAYER3_TEAM_ID": pl.Int64,
    "PLAYER3_TEAM_CITY": pl.String,
    "PLAYER3_TEAM_NICKNAME": pl.String,
    "PLAYER3_TEAM_ABBREVIATION": pl.String,
    "VIDEO_AVAILABLE_FLAG": pl.Int64,  # 0/1; kept integer, 1:1 with source
}

_DATANBA_SCHEMA: dict[str, pl.DataType] = {
    "evt": pl.Int64,  # event number, joins to nbastats EVENTNUM
    "wallclk": pl.String,  # ISO-8601 with mixed sub-second precision
    "cl": pl.String,  # game clock "12:00"
    "de": pl.String,  # description
    "locX": pl.Int64,
    "locY": pl.Int64,
    "opt1": pl.Int64,
    "opt2": pl.Int64,
    "opt3": pl.Int64,
    "opt4": pl.Int64,
    "mtype": pl.Int64,
    "etype": pl.Int64,
    "opid": pl.Int64,  # secondary player id; null on most events
    "tid": pl.Int64,
    "pid": pl.Int64,
    "hs": pl.Int64,  # home score
    "vs": pl.Int64,  # visitor score
    "epid": pl.Int64,  # tertiary player id; null on most events
    "oftid": pl.Int64,  # OFFENSE team id — the cross-check against nbastats
    "ord": pl.Int64,
    "pts": pl.Int64,
    "PERIOD": pl.Int64,
    "GAME_ID": pl.Int64,
}

# Keyed by the ``source`` component of a dataset name (see
# :func:`hoopstate.ingest.bulk_loader.parse_dataset_name`), so ``nbastats_2023``
# and a hypothetical ``nbastats_po_2023`` both resolve to the same schema.
SOURCE_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "nbastats": _NBASTATS_SCHEMA,
    "datanba": _DATANBA_SCHEMA,
}


@dataclass(frozen=True)
class BronzeResult:
    """The outcome of converting one dataset to typed bronze.

    ``schema`` is the realized column -> dtype mapping (dtype as its polars
    string name), so a caller can log or assert the exact types written without
    re-opening the parquet.
    """

    name: str
    source: str
    parquet_path: Path
    rows: int
    schema: dict[str, str]


def schema_for_dataset(name: str) -> tuple[str, dict[str, pl.DataType]]:
    """Return the ``(source, schema)`` for a dataset name.

    Raises :class:`KeyError` for a source with no registered schema, so an
    attempt to bronze-convert a dataset this module does not yet understand
    fails loudly rather than writing an untyped table.
    """
    parsed = parse_dataset_name(name)
    schema = SOURCE_SCHEMAS.get(parsed.source)
    if schema is None:
        known = ", ".join(sorted(SOURCE_SCHEMAS))
        raise KeyError(f"no bronze schema for source {parsed.source!r} (known: {known})")
    return parsed.source, schema


def apply_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Cast ``frame`` to ``schema``, validating the column set exactly.

    The frame's columns must match the schema's keys as a set — a missing or
    unexpected column means the parquet is not the source this schema describes,
    and is an error rather than something to paper over. Columns are returned in
    schema order (which mirrors the source CSV). Casts are strict, so a value
    that does not fit its declared type raises. Nulls are preserved: an empty
    source field is already null in the string capture and stays null.
    """
    got = set(frame.columns)
    want = set(schema)
    if got != want:
        missing = sorted(want - got)
        unexpected = sorted(got - want)
        raise ValueError(f"column mismatch: missing={missing} unexpected={unexpected}")
    return frame.select(pl.col(col).cast(dtype, strict=True) for col, dtype in schema.items())


def convert_dataset(name: str, *, profile: StorageProfile | None = None) -> BronzeResult:
    """Convert one dataset's string capture in bronze to a typed table in place.

    Reads the lossless string parquet the bulk loader wrote at the dataset's
    bronze path, applies the source's explicit schema, and atomically rewrites
    the same path with typed columns (write-temp-then-rename, so an interrupted
    write never leaves a corrupt parquet behind). Row count is invariant — the
    conversion is 1:1 — and returned for the caller to assert against the source.

    Idempotent: running it again re-reads the now-typed parquet and re-applies
    the schema, a no-op cast.
    """
    if profile is None:
        profile = resolve_profile()
    source, schema = schema_for_dataset(name)
    path = bronze_parquet_path(profile, name)
    if not path.exists():
        raise FileNotFoundError(f"{name}: no bronze parquet at {path}; run the bulk loader first")

    typed = apply_schema(pl.read_parquet(path), schema)

    tmp = path.with_name(path.name + ".tmp")
    try:
        typed.write_parquet(tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return BronzeResult(
        name=name,
        source=source,
        parquet_path=path,
        rows=typed.height,
        schema={col: str(dtype) for col, dtype in typed.schema.items()},
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hoopstate.ingest.bronze <name> [<name> ...]``.

    By default it (bulk-)loads each dataset first — a no-op if already cached —
    then converts it to typed bronze, so the command works from a cold checkout.
    ``--skip-load`` converts an existing string capture without touching the
    network; ``--force`` re-downloads and re-converts.
    """
    import argparse

    from hoopstate.ingest.bulk_loader import (
        load_dataset,
        read_manifest,
        requests_byte_source,
    )

    parser = argparse.ArgumentParser(
        description="Convert bulk-loaded play-by-play datasets to typed bronze parquet."
    )
    parser.add_argument(
        "names", nargs="+", help="Dataset names from the manifest, e.g. nbastats_2023."
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Convert an already-loaded bronze parquet without (re)fetching it.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download and re-convert even if cached."
    )
    args = parser.parse_args(argv)

    profile = resolve_profile()

    if not args.skip_load:
        byte_source = requests_byte_source()
        manifest = read_manifest(byte_source)
        for name in args.names:
            load_dataset(
                name,
                byte_source=byte_source,
                profile=profile,
                manifest=manifest,
                force=args.force,
            )

    for name in args.names:
        result = convert_dataset(name, profile=profile)
        print(f"{name}: {result.rows} rows typed ({result.source}) -> {result.parquet_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
