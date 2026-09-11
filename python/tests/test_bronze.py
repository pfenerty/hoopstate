"""Tests for the play-by-play bronze conversion (``hoops-1lg.2.2``).

Like the bulk-loader tests these run entirely offline: fixture ``tar.xz``
archives are built in-process and served through an in-memory
:class:`~hoopstate.ingest.bulk_loader.ByteSource`.

Acceptance criteria covered:

* nbastats and datanba datasets land as typed parquet in bronze, season
  partitioned.
* Column types are explicit (Int64 / String per the declared schema), not
  inferred — including ``GAME_ID`` kept as a string and ``SCOREMARGIN`` kept a
  string so its ``"TIE"`` sentinel survives.
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
    with pytest.raises(KeyError, match="no bronze schema for source 'shotdetail'"):
        schema_for_source("shotdetail")


def test_game_id_is_string_in_both_schemas() -> None:
    # GAME_ID is the join key and must never be coerced to an integer.
    assert BRONZE_SCHEMAS["nbastats"]["GAME_ID"] == pl.String()
    assert BRONZE_SCHEMAS["datanba"]["GAME_ID"] == pl.String()


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
    name, byte_map, manifest = _dataset("shotdetail_2023", ["GAME_ID", "x"], [["1", "2"]])
    source = _byte_source_from(byte_map)
    with pytest.raises(KeyError, match="no bronze schema for source 'shotdetail'"):
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
        "convert_dataset",
        "schema_for_source",
    }
