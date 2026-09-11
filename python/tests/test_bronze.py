"""Tests for the bronze conversion of the bulk-loaded sources.

Covers ``hoops-1lg.2.2`` (nbastats, datanba) and ``hoops-1lg.2.3``
(shotdetail, matchups).

Like the bulk-loader tests these run entirely offline: fixture ``tar.xz``
archives are built in-process and served through an in-memory
:class:`~hoopstate.ingest.bulk_loader.ByteSource`.

Acceptance criteria covered:

* Every declared dataset lands as typed parquet in bronze, season
  partitioned.
* Column types are explicit (Int64 / Float64 / String per the declared schema),
  not inferred — including ``GAME_ID`` kept as a string, ``SCOREMARGIN`` kept a
  string so its ``"TIE"`` sentinel survives, jersey numbers kept strings so
  ``"00"`` does not collapse to ``0``, and percentage columns kept Float64 even
  where a season happens to contain only integral values.
* Row counts match the source CSV.
* Schema drift (an unexpected or missing column) fails loudly.
* A source without a declared schema is refused rather than guessed at.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pytest

from hoopstate.ingest import bronze, bulk_loader
from hoopstate.ingest.bronze import (
    BRONZE_SCHEMAS,
    bronze_parquet_path,
    convert_dataset,
    schema_for_source,
)
from hoopstate.storage import ENV_COLD, ENV_HOT, ENV_PROFILE, Zone, resolve_profile

# --- fixtures ---------------------------------------------------------------


def _make_archive(csv_name: str, csv_text: str) -> bytes:
    csv_bytes = csv_text.encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        info = tarfile.TarInfo(name=csv_name)
        info.size = len(csv_bytes)
        tar.addfile(info, io.BytesIO(csv_bytes))
    return buf.getvalue()


def _byte_source_from(mapping: dict[str, bytes]) -> bulk_loader.ByteSource:
    @contextmanager
    def _open(url: str) -> Iterator[Iterator[bytes]]:
        if url not in mapping:
            raise KeyError(f"no fixture for {url}")
        payload = mapping[url]
        yield iter(payload[i : i + 7] for i in range(0, len(payload), 7))

    return _open


def _csv_from(columns: list[str], rows: list[list[str]]) -> str:
    lines = [",".join(columns)]
    lines.extend(",".join(r) for r in rows)
    return "\n".join(lines) + "\n"


_DEFAULT_BY_DTYPE = {pl.String(): "x", pl.Int64(): "1", pl.Float64(): "1.5"}


def _row_for(source: str, overrides: dict[str, str]) -> list[str]:
    """Build one CSV row for ``source``, filling uninteresting columns by dtype.

    shotdetail has 24 columns and matchups 48; spelling every value out by hand
    would bury the handful that each test actually asserts on.
    """
    schema = BRONZE_SCHEMAS[source]
    unknown = set(overrides) - set(schema)
    assert not unknown, f"overrides not in {source} schema: {sorted(unknown)}"
    return [overrides.get(col, _DEFAULT_BY_DTYPE[dtype]) for col, dtype in schema.items()]


# A minimal nbastats CSV: exercises GAME_ID-as-string, a nullable second player,
# the "TIE" SCOREMARGIN sentinel, and a null (empty) description column.
NBASTATS_COLUMNS = list(BRONZE_SCHEMAS["nbastats"])
NBASTATS_ROWS = [
    # GAME_ID, EVENTNUM, MSGTYPE, ACTIONTYPE, PERIOD, WC, PC, HOME, NEUTRAL,
    # VISITOR, SCORE, SCOREMARGIN, P1TYPE, P1_ID, P1_NAME, P1_TEAM_ID, ...
    [
        "0022300001",
        "2",
        "12",
        "0",
        "1",
        "7:31 PM",
        "12:00",
        "",
        "Start of 1st Period",
        "",
        "",
        "",
        "0",
        "0",
        "",
        "",
        "",
        "",
        "",
        "0",
        "0",
        "",
        "",
        "",
        "",
        "",
        "0",
        "0",
        "",
        "",
        "",
        "",
        "",
        "1",
    ],
    [
        "0022300001",
        "4",
        "1",
        "5",
        "1",
        "7:32 PM",
        "11:40",
        "Smith 2' Layup (2 PTS)",
        "",
        "",
        "2 - 0",
        "TIE",
        "4",
        "203999",
        "Nikola Jokic",
        "1610612743",
        "Denver",
        "Nuggets",
        "DEN",
        "5",
        "201142",
        "Kevin Durant",
        "1610612756",
        "Phoenix",
        "Suns",
        "PHX",
        "0",
        "0",
        "",
        "",
        "",
        "",
        "",
        "1",
    ],
]

# A minimal datanba CSV: exercises negative shot coords, nullable opid/epid, and
# the offense team id cross-check column.
DATANBA_COLUMNS = list(BRONZE_SCHEMAS["datanba"])
DATANBA_ROWS = [
    # evt, wallclk, cl, de, locX, locY, opt1..4, mtype, etype, opid, tid, pid,
    # hs, vs, epid, oftid, ord, pts, PERIOD, GAME_ID
    [
        "1",
        "2023-10-25T00:10:00.100Z",
        "12:00",
        "[DEN] Start",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "",
        "1610612743",
        "203999",
        "0",
        "0",
        "",
        "1610612743",
        "100",
        "0",
        "1",
        "0022300001",
    ],
    [
        "2",
        "2023-10-25T00:10:20Z",
        "11:40",
        "[DEN] Jokic layup",
        "-12",
        "43",
        "1",
        "0",
        "0",
        "0",
        "1",
        "1",
        "",
        "1610612743",
        "203999",
        "2",
        "0",
        "201142",
        "1610612743",
        "200",
        "2",
        "1",
        "0022300001",
    ],
]


@pytest.fixture
def ephemeral_profile(tmp_path: Path):
    return resolve_profile(
        env={ENV_PROFILE: "ephemeral", ENV_HOT: str(tmp_path), ENV_COLD: str(tmp_path)}
    )


def _dataset(name: str, columns: list[str], rows: list[list[str]]):
    url = f"https://example.com/{name}.tar.xz"
    archive = _make_archive(f"{name}.csv", _csv_from(columns, rows))
    return name, {url: archive}, {name: url}


# --- pure helpers -----------------------------------------------------------


def test_schema_for_source_known() -> None:
    assert schema_for_source("nbastats") is BRONZE_SCHEMAS["nbastats"]
    assert schema_for_source("datanba") is BRONZE_SCHEMAS["datanba"]


def test_schema_for_source_unknown_refuses() -> None:
    with pytest.raises(KeyError, match="no bronze schema for source 'boxscore'"):
        schema_for_source("boxscore")


def test_game_id_is_string_in_every_schema() -> None:
    # GAME_ID is the cross-source join key and must never become an integer.
    assert BRONZE_SCHEMAS["nbastats"]["GAME_ID"] == pl.String()
    assert BRONZE_SCHEMAS["datanba"]["GAME_ID"] == pl.String()
    assert BRONZE_SCHEMAS["shotdetail"]["GAME_ID"] == pl.String()
    assert BRONZE_SCHEMAS["matchups"]["game_id"] == pl.String()


def test_jersey_numbers_are_strings() -> None:
    # Every 2023 value parses as an integer, but the league issues "00" jerseys
    # and an Int64 cast would silently collapse that into 0.
    assert BRONZE_SCHEMAS["matchups"]["jersey_num"] == pl.String()
    assert BRONZE_SCHEMAS["matchups"]["matchups_jersey_num"] == pl.String()


def test_percentage_columns_are_float() -> None:
    # Declared from what the column means, not from what one season contains:
    # help_field_goals_percentage is the string "0" in every 2023 row.
    pct = {c: t for c, t in BRONZE_SCHEMAS["matchups"].items() if "percentage" in c}
    assert pct, "expected percentage columns in the matchups schema"
    assert set(pct.values()) == {pl.Float64()}


def test_bronze_parquet_path_is_season_partitioned(ephemeral_profile) -> None:
    path = bronze_parquet_path("shotdetail_2023", profile=ephemeral_profile)
    assert path == ephemeral_profile.zone(Zone.BRONZE, season=2023) / "shotdetail_2023.parquet"


# --- convert_dataset --------------------------------------------------------


def test_convert_nbastats_types_and_rows(ephemeral_profile) -> None:
    name, byte_map, manifest = _dataset("nbastats_2023", NBASTATS_COLUMNS, NBASTATS_ROWS)
    source = _byte_source_from(byte_map)

    result = convert_dataset(name, byte_source=source, profile=ephemeral_profile, manifest=manifest)

    assert result.source == "nbastats"
    assert result.rows == len(NBASTATS_ROWS)
    # Landed under bronze, season-partitioned.
    assert result.parquet_path.is_relative_to(ephemeral_profile.zone(Zone.BRONZE))
    assert "season=2023" in str(result.parquet_path)

    frame = pl.read_parquet(result.parquet_path)
    assert frame.height == len(NBASTATS_ROWS)
    dtypes = dict(zip(frame.columns, frame.dtypes, strict=True))
    # Explicit types, not inferred.
    assert dtypes["GAME_ID"] == pl.String
    assert dtypes["EVENTNUM"] == pl.Int64
    assert dtypes["PLAYER1_TEAM_ID"] == pl.Int64
    assert dtypes["SCOREMARGIN"] == pl.String
    # The "TIE" sentinel survived because SCOREMARGIN is a string.
    assert "TIE" in frame["SCOREMARGIN"].to_list()
    # An empty description field is null, and a real one is preserved.
    assert frame["HOMEDESCRIPTION"].to_list() == [None, "Smith 2' Layup (2 PTS)"]
    # An absent third player is a null id, not zero-as-string.
    assert frame["PLAYER1_ID"].to_list() == [0, 203999]


def test_convert_datanba_types_and_offense_team_id(ephemeral_profile) -> None:
    name, byte_map, manifest = _dataset("datanba_2023", DATANBA_COLUMNS, DATANBA_ROWS)
    source = _byte_source_from(byte_map)

    result = convert_dataset(name, byte_source=source, profile=ephemeral_profile, manifest=manifest)

    assert result.source == "datanba"
    assert result.rows == len(DATANBA_ROWS)

    frame = pl.read_parquet(result.parquet_path)
    dtypes = dict(zip(frame.columns, frame.dtypes, strict=True))
    assert dtypes["GAME_ID"] == pl.String
    assert dtypes["oftid"] == pl.Int64  # the offense-team cross-check
    assert dtypes["locX"] == pl.Int64
    # Signed shot coordinates round-trip.
    assert frame["locX"].to_list() == [0, -12]
    # oftid is populated on every row — it is the reason this source is ingested.
    assert frame["oftid"].null_count() == 0
    # Nullable secondary player ids: empty -> null.
    assert frame["opid"].to_list() == [None, None]
    assert frame["epid"].to_list() == [None, 201142]


def test_convert_shotdetail_types_and_rows(ephemeral_profile) -> None:
    columns = list(BRONZE_SCHEMAS["shotdetail"])
    rows = [
        _row_for(
            "shotdetail",
            {
                "GAME_ID": "0022300001",
                "GAME_EVENT_ID": "4",
                "PERIOD": "1",
                "LOC_X": "-138",
                "LOC_Y": "83",
                "SHOT_MADE_FLAG": "1",
                "GAME_DATE": "20231024",
                "ACTION_TYPE": "Jump Shot",
            },
        ),
        _row_for(
            "shotdetail",
            {
                "GAME_ID": "0022300001",
                "GAME_EVENT_ID": "7",
                "PERIOD": "2",
                # Negative coordinates are ordinary: LOC_X is signed about the
                # basket, so Int64 must not be confused for an unsigned count.
                "LOC_X": "-7",
                "LOC_Y": "-11",
                "SHOT_MADE_FLAG": "0",
                "GAME_DATE": "20231024",
                "ACTION_TYPE": "Driving Layup Shot",
            },
        ),
    ]
    name, byte_map, manifest = _dataset("shotdetail_2023", columns, rows)

    result = convert_dataset(
        name, byte_source=_byte_source_from(byte_map), profile=ephemeral_profile, manifest=manifest
    )

    assert result.source == "shotdetail"
    assert result.rows == len(rows)
    assert result.parquet_path == bronze_parquet_path(name, profile=ephemeral_profile)

    frame = pl.read_parquet(result.parquet_path)
    assert frame.height == len(rows)
    assert frame.columns == columns
    dtypes = dict(zip(frame.columns, frame.dtypes, strict=True))
    assert dtypes["GAME_ID"] == pl.String
    assert dtypes["GAME_EVENT_ID"] == pl.Int64
    assert dtypes["LOC_X"] == pl.Int64
    # GAME_DATE is a YYYYMMDD identifier, not a quantity; parsing is silver's job.
    assert dtypes["GAME_DATE"] == pl.String
    assert frame["GAME_DATE"].to_list() == ["20231024", "20231024"]
    assert frame["LOC_X"].to_list() == [-138, -7]
    assert frame["ACTION_TYPE"].to_list() == ["Jump Shot", "Driving Layup Shot"]


def test_convert_matchups_types_and_rows(ephemeral_profile) -> None:
    columns = list(BRONZE_SCHEMAS["matchups"])
    rows = [
        _row_for(
            "matchups",
            {
                "game_id": "0022300001",
                "person_id": "203999",
                "matchups_person_id": "1629029",
                # The league issues "00": an Int64 cast would collapse it to 0.
                "jersey_num": "00",
                "matchups_jersey_num": "77",
                # A clock string, typed like PCTIMESTRING rather than a number.
                "matchup_minutes": "6:20",
                "matchup_minutes_sort": "380.0",
                "partial_possessions": "13.4",
                # Integral in the source text but semantically a rate.
                "help_field_goals_percentage": "0",
                "comment": "",
            },
        ),
        _row_for(
            "matchups",
            {
                "game_id": "0022300001",
                "person_id": "1629029",
                "matchups_person_id": "203999",
                "jersey_num": "7",
                "matchups_jersey_num": "00",
                "matchup_minutes": "0:20",
                "matchup_minutes_sort": "20.0",
                "partial_possessions": "0.5",
                "help_field_goals_percentage": "0.5",
                "comment": "",
            },
        ),
    ]
    name, byte_map, manifest = _dataset("matchups_2023", columns, rows)

    result = convert_dataset(
        name, byte_source=_byte_source_from(byte_map), profile=ephemeral_profile, manifest=manifest
    )

    assert result.source == "matchups"
    assert result.rows == len(rows)

    frame = pl.read_parquet(result.parquet_path)
    assert frame.columns == columns
    dtypes = dict(zip(frame.columns, frame.dtypes, strict=True))
    assert dtypes["game_id"] == pl.String
    assert dtypes["person_id"] == pl.Int64
    assert dtypes["jersey_num"] == pl.String
    assert dtypes["matchups_jersey_num"] == pl.String
    assert dtypes["matchup_minutes"] == pl.String
    assert dtypes["matchup_minutes_sort"] == pl.Float64
    assert dtypes["partial_possessions"] == pl.Float64
    assert dtypes["help_field_goals_percentage"] == pl.Float64
    # "00" survived as written.
    assert frame["jersey_num"].to_list() == ["00", "7"]
    assert frame["matchups_jersey_num"].to_list() == ["77", "00"]
    assert frame["matchup_minutes"].to_list() == ["6:20", "0:20"]
    assert frame["partial_possessions"].to_list() == [13.4, 0.5]
    # A fully-empty column is declared and kept null, not dropped: bronze is 1:1.
    assert frame["comment"].to_list() == [None, None]


def test_matchups_has_no_period_column() -> None:
    """matchups is a game-level aggregate, not a period-level observation.

    ``hoops-1lg.4.1`` planned period-starter resolution around a period column
    here. There is none, and this test pins that so the correction is not
    quietly undone by a future schema edit.
    """
    lowered = {c.lower() for c in BRONZE_SCHEMAS["matchups"]}
    assert "period" not in lowered
    assert not any(c.startswith("period") for c in lowered)


def test_convert_writes_typed_over_lossless_capture(ephemeral_profile) -> None:
    # A prior bulk-loader run leaves an all-strings parquet at the canonical
    # bronze path; the conversion supersedes it with the typed table.
    name, byte_map, manifest = _dataset("nbastats_2023", NBASTATS_COLUMNS, NBASTATS_ROWS)
    source = _byte_source_from(byte_map)

    loaded = bulk_loader.load_dataset(
        name, byte_source=source, profile=ephemeral_profile, manifest=manifest
    )
    assert set(pl.read_parquet(loaded.parquet_path).dtypes) == {pl.String}

    converted = convert_dataset(
        name, byte_source=source, profile=ephemeral_profile, manifest=manifest
    )
    assert converted.parquet_path == loaded.parquet_path
    assert pl.read_parquet(converted.parquet_path)["EVENTNUM"].dtype == pl.Int64


def test_convert_is_idempotent(ephemeral_profile) -> None:
    name, byte_map, manifest = _dataset("nbastats_2023", NBASTATS_COLUMNS, NBASTATS_ROWS)
    source = _byte_source_from(byte_map)
    first = convert_dataset(name, byte_source=source, profile=ephemeral_profile, manifest=manifest)
    second = convert_dataset(name, byte_source=source, profile=ephemeral_profile, manifest=manifest)
    assert first.rows == second.rows
    assert pl.read_parquet(second.parquet_path).equals(pl.read_parquet(first.parquet_path))


def test_convert_refuses_source_without_schema(ephemeral_profile) -> None:
    name, byte_map, manifest = _dataset("boxscore_2023", ["GAME_ID", "x"], [["1", "2"]])
    source = _byte_source_from(byte_map)
    with pytest.raises(KeyError, match="no bronze schema for source 'boxscore'"):
        convert_dataset(name, byte_source=source, profile=ephemeral_profile, manifest=manifest)


# --- schema drift detection (via the typed streaming primitive) -------------


def test_typed_conversion_rejects_unexpected_column(tmp_path: Path) -> None:
    archive = tmp_path / "d.tar.xz"
    archive.write_bytes(_make_archive("d.csv", "GAME_ID,EXTRA\n1,2\n"))
    with pytest.raises(ValueError, match="unexpected=\\['EXTRA'\\]"):
        bulk_loader.stream_archive_to_typed_parquet(
            archive, tmp_path / "out.parquet", {"GAME_ID": pl.String()}
        )


def test_typed_conversion_rejects_missing_column(tmp_path: Path) -> None:
    archive = tmp_path / "d.tar.xz"
    archive.write_bytes(_make_archive("d.csv", "GAME_ID\n1\n"))
    with pytest.raises(ValueError, match="missing=\\['EVENTNUM'\\]"):
        bulk_loader.stream_archive_to_typed_parquet(
            archive,
            tmp_path / "out.parquet",
            {"GAME_ID": pl.String(), "EVENTNUM": pl.Int64()},
        )


def test_typed_conversion_strict_cast_raises_on_bad_value(tmp_path: Path) -> None:
    # A non-integer in an Int64 column fails loudly instead of nulling.
    archive = tmp_path / "d.tar.xz"
    archive.write_bytes(_make_archive("d.csv", "EVENTNUM\nnot-a-number\n"))
    parquet = tmp_path / "out.parquet"
    with pytest.raises(Exception):  # noqa: B017 - polars raises its own error type
        bulk_loader.stream_archive_to_typed_parquet(archive, parquet, {"EVENTNUM": pl.Int64()})
    assert not parquet.exists()


def test_bronze_module_public_api() -> None:
    assert set(bronze.__all__) == {
        "BRONZE_SCHEMAS",
        "BronzeResult",
        "bronze_parquet_path",
        "convert_dataset",
        "schema_for_source",
    }
