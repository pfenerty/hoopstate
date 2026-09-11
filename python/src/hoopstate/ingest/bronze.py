"""Bronze conversion for the bulk-loaded source datasets.

Play-by-play sources arrived with ``hoops-1lg.2.2``; shotdetail and matchups with
``hoops-1lg.2.3``.

The bulk loader (:mod:`hoopstate.ingest.bulk_loader`) lands each source as a
lossless, all-strings parquet — a faithful 1:1 capture that infers nothing. This
module is the second half of the bronze zone's contract: it re-reads the raw
archive and writes *typed* parquet, casting every column to an **explicit**
dtype declared here, never one inferred from the data. Bronze is thus typed and
1:1 with source, and fully rebuildable from the immutable raw archive.

Four sources are converted, and they matter for different reasons:

``nbastats``
    The primary event feed, from stats.nba.com's ``playbyplayv2`` endpoint. One
    row per logged event, with up to three involved players and free-text
    descriptions split across home/neutral/visitor columns.

``datanba``
    A parallel feed from data.nba.com. Its value here is the explicit offense
    **team** id (``oftid``) on every event: an independent observation of who had
    the ball, which is the cross-check that catches the out-of-order events
    stats.nba.com is known to contain. It also carries shot coordinates
    (``locX``/``locY``) and an absolute ordering key (``ord``).

``shotdetail``
    One row per field-goal attempt, from stats.nba.com's ``shotchartdetail`` endpoint,
    carrying shot coordinates (``LOC_X``/``LOC_Y``), distance, and the zone labels
    (basic/area/range) the league's own shot charts are drawn from. It joins to the
    event feed on ``(GAME_ID, GAME_EVENT_ID)`` -> ``(GAME_ID, EVENTNUM)``. Available
    from season 1996 onward.

``matchups``
    One row per (offensive player, defensive player) pair **per game**, with total
    matchup minutes, partial possessions, and a matchup box-score line. Available from
    season 2017 onward.

    **This source is a game-level aggregate and has no period column.** The backlog
    originally described it as per-possession, which is wrong: it cannot say who was on
    the floor at the start of a given period. For E4's period-starter problem it is a
    game-level roster and a minutes/partial-possession weighting prior, not a direct
    on-court observation. See ``hoops-1lg.4.1``.

Every source joins to the rest of the model on ``GAME_ID``. The cross-source facts that
justify those joins are not asserted here in prose — they are measured by the named
reports in :mod:`hoopstate.ingest.reports`, which can be re-run for any season.

Typing decisions, applied uniformly across both schemas
--------------------------------------------------------

* ``GAME_ID`` stays :class:`polars.String`. It is an identifier and a join key,
  not a quantity. This collection stores it with the canonical NBA 10-character
  form's leading zeros stripped (``"22300875"``, not ``"0022300875"``); both
  sources strip it identically, so they still join. Re-padding is a
  canonicalization concern for the silver zone, not something bronze invents.
* Integer columns — event/message codes, periods, player and team ids, scores,
  coordinates, ordering — are :class:`polars.Int64`. Int64 (rather than a
  narrower width) is deliberate: it clears NBA's 10-digit franchise ids with
  headroom and removes any per-column overflow reasoning, at negligible cost in
  parquet. Nullable throughout — an absent second/third player or an unassigned
  team id is null, not zero.
* Rates, percentages and fractional durations are :class:`polars.Float64`. A column
  that is semantically a rate is declared as one **even when every observed value
  happens to parse as an integer** — ``help_field_goals_percentage`` is the string
  ``"0"`` in every 2023 row, and declaring it ``Int64`` on that evidence would be
  exactly the infer-from-data mistake this module exists to prevent.
* Free text, clock strings, wall-clock timestamps, and the composite ``SCORE``
  string stay :class:`polars.String`. ``SCOREMARGIN`` also stays a string
  because it carries the sentinel ``"TIE"`` alongside signed integers.
  ``matchup_minutes`` (``"0:20"``) follows the same rule as ``PCTIMESTRING``; its
  ``Float64`` companion ``matchup_minutes_sort`` carries the sortable duration.
* Identifiers that merely look numeric stay :class:`polars.String`. Jersey numbers are
  the case that bites: the league issues ``"00"``, which an ``Int64`` cast would
  silently collapse into ``0``. ``GAME_DATE`` likewise stays a string — it is a
  ``YYYYMMDD`` date, not a quantity, and parsing it is a silver-zone concern.

Casts are strict (see :func:`~hoopstate.ingest.bulk_loader.stream_archive_to_typed_parquet`):
a value that does not fit its declared type fails the conversion loudly rather
than being silently nulled, and an unexpected or missing column is treated as
upstream schema drift and raises. Because every row is cast and kept, the typed
row count equals the source CSV's.
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
    "BRONZE_SCHEMAS",
    "BronzeResult",
    "bronze_parquet_path",
    "convert_dataset",
    "schema_for_source",
]

# Explicit bronze schemas, keyed by source (the ``source`` field of a parsed
# dataset name, e.g. ``nbastats_2023`` -> ``"nbastats"``). Column order matches
# the source CSV; it is also the order the typed parquet is written in. See the
# module docstring for the typing rationale.
_STR = pl.String()
_INT = pl.Int64()
_FLOAT = pl.Float64()

_NBASTATS_SCHEMA: dict[str, pl.DataType] = {
    "GAME_ID": _STR,
    "EVENTNUM": _INT,
    "EVENTMSGTYPE": _INT,
    "EVENTMSGACTIONTYPE": _INT,
    "PERIOD": _INT,
    "WCTIMESTRING": _STR,  # wall-clock, e.g. "6:05 PM"
    "PCTIMESTRING": _STR,  # game clock remaining in the period, e.g. "0:35"
    "HOMEDESCRIPTION": _STR,
    "NEUTRALDESCRIPTION": _STR,
    "VISITORDESCRIPTION": _STR,
    "SCORE": _STR,  # composite "home - visitor", e.g. "128 - 114"
    "SCOREMARGIN": _STR,  # signed int or the sentinel "TIE"
    "PERSON1TYPE": _INT,
    "PLAYER1_ID": _INT,
    "PLAYER1_NAME": _STR,
    "PLAYER1_TEAM_ID": _INT,
    "PLAYER1_TEAM_CITY": _STR,
    "PLAYER1_TEAM_NICKNAME": _STR,
    "PLAYER1_TEAM_ABBREVIATION": _STR,
    "PERSON2TYPE": _INT,
    "PLAYER2_ID": _INT,
    "PLAYER2_NAME": _STR,
    "PLAYER2_TEAM_ID": _INT,
    "PLAYER2_TEAM_CITY": _STR,
    "PLAYER2_TEAM_NICKNAME": _STR,
    "PLAYER2_TEAM_ABBREVIATION": _STR,
    "PERSON3TYPE": _INT,
    "PLAYER3_ID": _INT,
    "PLAYER3_NAME": _STR,
    "PLAYER3_TEAM_ID": _INT,
    "PLAYER3_TEAM_CITY": _STR,
    "PLAYER3_TEAM_NICKNAME": _STR,
    "PLAYER3_TEAM_ABBREVIATION": _STR,
    "VIDEO_AVAILABLE_FLAG": _INT,
}

_DATANBA_SCHEMA: dict[str, pl.DataType] = {
    "evt": _INT,  # event number within the game
    "wallclk": _STR,  # ISO 8601 UTC timestamp, e.g. "2023-12-12T01:22:01.900Z"
    "cl": _STR,  # game clock remaining, e.g. "00:50.5"
    "de": _STR,  # free-text event description
    "locX": _INT,  # shot location x (signed; court-relative)
    "locY": _INT,  # shot location y (signed; court-relative)
    "opt1": _INT,
    "opt2": _INT,
    "opt3": _INT,
    "opt4": _INT,
    "mtype": _INT,  # message/action sub-type
    "etype": _INT,  # event type
    "opid": _INT,  # secondary/assist player id (nullable)
    "tid": _INT,  # team id of the acting player
    "pid": _INT,  # acting player id
    "hs": _INT,  # home score after the event
    "vs": _INT,  # visitor score after the event
    "epid": _INT,  # tertiary player id, e.g. blocked/stolen-from (nullable)
    "oftid": _INT,  # offense team id — the cross-check for event ordering
    "ord": _INT,  # absolute ordering key within the game
    "pts": _INT,  # points scored on the event
    "PERIOD": _INT,
    "GAME_ID": _STR,
}

# shotdetail: one row per field-goal attempt, from stats.nba.com's ``shotchartdetail``
# endpoint. ``GAME_DATE`` stays a string: it is a ``YYYYMMDD`` date, not a quantity, and
# parsing it is a silver-zone concern — the same rule that keeps ``WCTIMESTRING`` a string.
_SHOTDETAIL_SCHEMA: dict[str, pl.DataType] = {
    "GRID_TYPE": _STR,  # constant "Shot Chart Detail" in every row observed
    "GAME_ID": _STR,
    "GAME_EVENT_ID": _INT,  # joins to nbastats EVENTNUM
    "PLAYER_ID": _INT,
    "PLAYER_NAME": _STR,
    "TEAM_ID": _INT,
    "TEAM_NAME": _STR,
    "PERIOD": _INT,
    "MINUTES_REMAINING": _INT,
    "SECONDS_REMAINING": _INT,
    "EVENT_TYPE": _STR,  # "Made Shot" / "Missed Shot"
    "ACTION_TYPE": _STR,  # shot descriptor, e.g. "Floating Jump shot"
    "SHOT_TYPE": _STR,  # "2PT Field Goal" / "3PT Field Goal"
    "SHOT_ZONE_BASIC": _STR,
    "SHOT_ZONE_AREA": _STR,
    "SHOT_ZONE_RANGE": _STR,
    "SHOT_DISTANCE": _INT,  # feet
    "LOC_X": _INT,  # court-relative, tenths of a foot, signed
    "LOC_Y": _INT,
    "SHOT_ATTEMPTED_FLAG": _INT,
    "SHOT_MADE_FLAG": _INT,
    "GAME_DATE": _STR,  # YYYYMMDD, e.g. "20231114"
    "HTM": _STR,  # home team tricode
    "VTM": _STR,  # visitor team tricode
}

# matchups: one row per (offensive player, defensive player) pair **per game**. Note the
# lowercase snake_case column names — this source does not share the upstream convention of
# the others. There is deliberately no period column here, because the source has none; see
# the module docstring.
_MATCHUPS_SCHEMA: dict[str, pl.DataType] = {
    "game_id": _STR,
    "away_team_id": _INT,
    "home_team_id": _INT,
    "team_id": _INT,  # the offensive player's team
    "team_name": _STR,
    "team_city": _STR,
    "team_tricode": _STR,
    "team_slug": _STR,
    "person_id": _INT,  # offensive player
    "first_name": _STR,
    "family_name": _STR,
    "name_i": _STR,  # abbreviated form, e.g. "B. Mathurin"
    "player_slug": _STR,
    "position": _STR,
    "comment": _STR,  # null in every 2023 row; declared, not dropped — bronze is 1:1
    "jersey_num": _STR,  # identifier, not a quantity — see the module docstring
    "matchups_person_id": _INT,  # defensive player
    "matchups_first_name": _STR,
    "matchups_family_name": _STR,
    "matchups_name_i": _STR,
    "matchups_player_slug": _STR,
    "matchups_jersey_num": _STR,
    "matchup_minutes": _STR,  # clock string, e.g. "0:20"
    "matchup_minutes_sort": _FLOAT,  # the same duration in seconds
    "partial_possessions": _FLOAT,
    "percentage_defender_total_time": _FLOAT,
    "percentage_offensive_total_time": _FLOAT,
    "percentage_total_time_both_on": _FLOAT,
    "switches_on": _INT,
    "player_points": _INT,
    "team_points": _INT,
    "matchup_assists": _INT,
    "matchup_potential_assists": _INT,
    "matchup_turnovers": _INT,
    "matchup_blocks": _INT,
    "matchup_field_goals_made": _INT,
    "matchup_field_goals_attempted": _INT,
    "matchup_field_goals_percentage": _FLOAT,
    "matchup_three_pointers_made": _INT,
    "matchup_three_pointers_attempted": _INT,
    "matchup_three_pointers_percentage": _FLOAT,
    "help_blocks": _INT,
    "help_field_goals_made": _INT,
    "help_field_goals_attempted": _INT,
    "help_field_goals_percentage": _FLOAT,  # all-zero in 2023; a rate regardless
    "matchup_free_throws_made": _INT,
    "matchup_free_throws_attempted": _INT,
    "shooting_fouls": _INT,
}

BRONZE_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "nbastats": _NBASTATS_SCHEMA,
    "datanba": _DATANBA_SCHEMA,
    "shotdetail": _SHOTDETAIL_SCHEMA,
    "matchups": _MATCHUPS_SCHEMA,
}


def schema_for_source(source: str) -> dict[str, pl.DataType]:
    """Return the explicit bronze schema for ``source``.

    Raises :class:`KeyError` for a source without a declared schema — the bronze
    conversion is per-source and deliberately refuses to guess.
    """
    try:
        return BRONZE_SCHEMAS[source]
    except KeyError:
        known = ", ".join(sorted(BRONZE_SCHEMAS))
        raise KeyError(f"no bronze schema for source {source!r}; known sources: {known}") from None


def bronze_parquet_path(name: str, *, profile: StorageProfile) -> Path:
    """Return the canonical bronze parquet path for dataset ``name``.

    The season partition is taken from the name itself, so this is the one place
    that knows where a converted dataset lands. :mod:`hoopstate.ingest.reports`
    reads through it rather than re-deriving the layout.
    """
    parsed = parse_dataset_name(name)
    return profile.zone(Zone.BRONZE, season=parsed.season) / f"{name}.parquet"


@dataclass(frozen=True)
class BronzeResult:
    """The outcome of converting one dataset to typed bronze parquet."""

    name: str
    source: str
    parquet_path: Path
    rows: int


def convert_dataset(
    name: str,
    *,
    byte_source: ByteSource,
    profile: StorageProfile | None = None,
    manifest: dict[str, str] | None = None,
    force: bool = False,
) -> BronzeResult:
    """Convert one manifest dataset to typed bronze parquet.

    Obtains the raw archive (downloading and caching it if needed via
    :func:`~hoopstate.ingest.bulk_loader.ensure_archive`), then writes typed
    parquet to the canonical, season-partitioned bronze path for ``name`` using
    the source's explicit schema. The written table supersedes any lossless
    string capture the bulk loader may have left at the same path.

    ``force`` re-downloads the archive and re-writes the parquet.
    """
    if profile is None:
        profile = resolve_profile()

    parsed = parse_dataset_name(name)
    schema = schema_for_source(parsed.source)

    ref = ensure_archive(
        name, byte_source=byte_source, profile=profile, manifest=manifest, force=force
    )
    parquet_path = bronze_parquet_path(name, profile=profile)
    rows = stream_archive_to_typed_parquet(ref.archive_path, parquet_path, schema)
    return BronzeResult(name=name, source=parsed.source, parquet_path=parquet_path, rows=rows)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hoopstate.ingest.bronze <name> [<name> ...]``."""
    import argparse

    from hoopstate.ingest.bulk_loader import read_manifest, requests_byte_source

    parser = argparse.ArgumentParser(
        description="Convert play-by-play sources to typed bronze parquet."
    )
    parser.add_argument(
        "names", nargs="+", help="Dataset names from the manifest, e.g. nbastats_2023."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download and re-convert even if cached."
    )
    args = parser.parse_args(argv)

    byte_source = requests_byte_source()
    profile = resolve_profile()
    manifest = read_manifest(byte_source)
    for name in args.names:
        result = convert_dataset(
            name, byte_source=byte_source, profile=profile, manifest=manifest, force=args.force
        )
        print(f"{name}: {result.rows} rows ({result.source}) -> {result.parquet_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
