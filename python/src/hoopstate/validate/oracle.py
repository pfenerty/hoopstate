"""Quarantined ingest of the pbpstats validation oracle (``hoops-1lg.2.4``).

pbpstats.com publishes possession-level data derived by a mature reference
implementation: where each possession started and ended, and *how* it started.
That makes it this project's answer key — the thing E7's comparison harness
(``hoops-1lg.7.1``) grades derived possessions against, and later the contract
the E10 Rust port must satisfy.

An answer key is only worth something if it never contaminates the work being
graded. If a possession-derivation module could read ``STARTTYPE``, it could
copy answers instead of deriving them, and the harness would end up measuring
pbpstats against itself and reporting a match rate that means nothing. So the
oracle is quarantined: it lands in its own zone, this package is the only one
allowed to name it, and ``tests/test_oracle_quarantine.py`` enforces that
mechanically rather than by convention.

The quarantine is physical as well as static. The downloaded archive, its
checksum sidecar and the typed parquet all live under
:attr:`~hoopstate.storage.Zone.ORACLE`, so nothing belonging to the answer key
is stored among the core-model sources and no glob over the raw zone can sweep
it in. Only the zone-blind primitives are reused from
:mod:`hoopstate.ingest.bulk_loader` — they take a directory and a schema and
learn nothing about the oracle, so the dependency runs ``validate`` ->
``ingest`` and never the reverse.

What the source actually contains
---------------------------------

One row per possession, both teams, 19 columns. The 2023 season carries 478,625
possessions across 1,228 games — a median of 388 per game.

* ``GAMEID`` is the **leading-zero-stripped** 8-character form (``"22300001"``),
  which is exactly the form bronze stores (see :mod:`hoopstate.ingest.bronze`).
  The oracle therefore joins to the event feed on ``GAMEID`` with no
  re-padding. This was worth checking rather than assuming: the canonical NBA
  id is 10 characters, and the video links in ``URL`` do carry the padded form.
* ``STARTTIME`` and ``ENDTIME`` are clock **strings** (``"MM:SS"``, e.g.
  ``"00:07"``), counting time remaining in the period, not elapsed. They stay
  :class:`polars.String` under the same rule that keeps ``PCTIMESTRING`` one;
  converting them to a duration is a silver-zone concern.
* ``STARTTYPE`` carries 17 distinct values in 2023, every one prefixed
  ``"Off "``: ``Off At Rim Make/Miss/Block``, ``Off Arc 3 Make/Miss``,
  ``Off Corner 3 Make/Miss``, ``Off Short Mid-Range Make/Miss``,
  ``Off Long Mid-Range Make/Miss``, ``Off FT Make/Miss``, ``Off Dead Ball``,
  ``Off Steal``, ``Off Timeout``, ``Off Block``. E7 needs this vocabulary to
  design the start-type comparison, so it is recorded here rather than
  rediscovered.
* ``OPPONENT`` is the **defending** team's tricode. There is no column naming
  the team in possession, so recovering it means knowing the game's two teams.
* ``EVENTS`` is a newline-joined list of every action in the possession, so the
  CSV contains quoted multi-line fields — 477,607 of the 478,625 rows have one.
  Anything that processes the raw CSV line-by-line will be wrong.
* Two games are **missing** from the 2023 file: ``22301177`` and ``22301195``.
  The id space runs ``22300001``-``22301230`` without other gaps. E7 must treat
  an absent game as absent rather than as a mismatch.

Typing follows the same rules as bronze: identifiers and dates stay
:class:`polars.String` (``GAMEDATE`` is ISO ``"2023-10-24"`` here, and parsing
it belongs to silver), counts and score differentials are
:class:`polars.Int64`, and free text and clock strings stay strings. Casts are
strict and the column set must match exactly, so upstream schema drift fails
the conversion loudly.

Usage::

    python -m hoopstate.validate.oracle              # defaults to pbpstats_2023
    python -m hoopstate.validate.oracle pbpstats_2022 pbpstats_2023
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hoopstate.ingest.bulk_loader import (
    ByteSource,
    ensure_archive,
    parse_dataset_name,
    stream_archive_to_typed_parquet,
)
from hoopstate.storage import StorageProfile, Zone, resolve_profile

__all__ = [
    "DEFAULT_ORACLE_DATASET",
    "ORACLE_SCHEMA",
    "ORACLE_SOURCE",
    "OracleResult",
    "ingest_oracle",
    "oracle_archive_dir",
    "oracle_parquet_path",
]

# The manifest source name for the oracle, and the season the validation work
# is built around. Parameterized rather than hardcoded: E7 may want other
# seasons, and the playoff companion (``pbpstats_po_2023``) parses the same way.
ORACLE_SOURCE = "pbpstats"
DEFAULT_ORACLE_DATASET = "pbpstats_2023"

_STR = pl.String()
_INT = pl.Int64()

# Explicit schema, in the source CSV's column order. Every dtype is declared;
# none is inferred. See the module docstring for the per-column rationale.
ORACLE_SCHEMA: dict[str, pl.DataType] = {
    "ENDTIME": _STR,  # clock string "MM:SS" remaining in the period
    "EVENTS": _STR,  # newline-joined description of every action
    "FG2A": _INT,
    "FG2M": _INT,
    "FG3A": _INT,
    "FG3M": _INT,
    "GAMEDATE": _STR,  # ISO date, e.g. "2023-10-24"
    "GAMEID": _STR,  # 8-char, zero-stripped — joins to bronze as-is
    "NONSHOOTINGFOULSTHATRESULTEDINFTS": _INT,
    "OFFENSIVEREBOUNDS": _INT,
    "OPPONENT": _STR,  # the *defending* team's tricode
    "PERIOD": _INT,  # 1-6 observed; overtime periods continue the count
    "SHOOTINGFOULSDRAWN": _INT,
    "STARTSCOREDIFFERENTIAL": _INT,  # signed, from the offense's perspective
    "STARTTIME": _STR,  # clock string "MM:SS" remaining in the period
    "STARTTYPE": _STR,  # 17-value vocabulary; see the module docstring
    "TURNOVERS": _INT,
    "DESCRIPTION": _STR,  # the possession's final action
    "URL": _STR,  # video link; empty or null for many possessions
}


def oracle_archive_dir(profile: StorageProfile) -> Path:
    """Where oracle archives are cached — inside the oracle zone, not raw.

    Keeping the archive here is what makes the quarantine physical: the whole
    answer key, source bytes included, sits under one directory.
    """
    return profile.zone(Zone.ORACLE) / "raw"


def oracle_parquet_path(name: str, *, profile: StorageProfile) -> Path:
    """The canonical oracle parquet path for dataset ``name``.

    Flat rather than season-partitioned: the dataset name already carries the
    season, and the oracle is small enough that partitioning would buy nothing.
    """
    return profile.zone(Zone.ORACLE) / f"{name}.parquet"


@dataclass(frozen=True)
class OracleResult:
    """The outcome of ingesting one oracle dataset."""

    name: str
    parquet_path: Path
    rows: int


def ingest_oracle(
    name: str = DEFAULT_ORACLE_DATASET,
    *,
    byte_source: ByteSource,
    profile: StorageProfile | None = None,
    manifest: dict[str, str] | None = None,
    force: bool = False,
) -> OracleResult:
    """Ingest one pbpstats dataset into the quarantined oracle zone.

    Downloads and caches the archive inside the oracle zone, then writes typed
    parquet beside it using :data:`ORACLE_SCHEMA`. ``force`` re-downloads and
    re-writes.

    A dataset whose source is not pbpstats is refused: this function is the
    oracle's front door, and pointing it at a core-model source would put
    ordinary data in the zone the guard treats as untouchable.
    """
    if profile is None:
        profile = resolve_profile()

    parsed = parse_dataset_name(name)
    if parsed.source != ORACLE_SOURCE:
        raise ValueError(
            f"{name!r} is not an oracle dataset (source {parsed.source!r}, "
            f"expected {ORACLE_SOURCE!r}); only the oracle belongs in the oracle zone"
        )

    ref = ensure_archive(
        name,
        byte_source=byte_source,
        profile=profile,
        manifest=manifest,
        force=force,
        archive_dir=oracle_archive_dir(profile),
    )
    parquet_path = oracle_parquet_path(name, profile=profile)
    rows = stream_archive_to_typed_parquet(ref.archive_path, parquet_path, ORACLE_SCHEMA)
    return OracleResult(name=name, parquet_path=parquet_path, rows=rows)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hoopstate.validate.oracle [<name> ...]``."""
    import argparse

    from hoopstate.ingest.bulk_loader import read_manifest, requests_byte_source

    parser = argparse.ArgumentParser(
        description="Ingest the pbpstats validation oracle into the quarantined oracle zone."
    )
    parser.add_argument(
        "names",
        nargs="*",
        default=[DEFAULT_ORACLE_DATASET],
        help=f"Oracle dataset names (default: {DEFAULT_ORACLE_DATASET}).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download and re-convert even if cached."
    )
    args = parser.parse_args(argv)

    byte_source = requests_byte_source()
    profile = resolve_profile()
    manifest = read_manifest(byte_source)
    for name in args.names:
        result = ingest_oracle(
            name, byte_source=byte_source, profile=profile, manifest=manifest, force=args.force
        )
        print(f"{name}: {result.rows} rows -> {result.parquet_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
