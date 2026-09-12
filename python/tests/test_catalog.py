"""Tests for the DuckDB catalog and rebuild helper (``hoops-1lg.1.4``).

These assert the acceptance criteria directly:

* A view convention per zone — one DuckDB schema per zone, one view per dataset.
* One command rebuilds the database from parquet with nothing lost.
* Deleting the database file and rebuilding produces identical query results.

Plus the two mechanical traps the convention is shaped around, kept as
regressions because both fail silently or only under a DuckDB upgrade: a glob
that matches nothing must not be turned into a view, and ``min``/``max`` over
the season column must work.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from hoopstate.db.catalog import (
    CATALOG_ZONES,
    connect,
    main,
    planned_views,
    rebuild,
)
from hoopstate.ingest.bronze import BRONZE_SCHEMAS
from hoopstate.storage import ENV_PROFILE, ENV_ROOT, Profile, StorageProfile, Zone


@pytest.fixture
def profile(tmp_path: Path) -> StorageProfile:
    return StorageProfile(name=Profile.EPHEMERAL, root=tmp_path)


def write_bronze(profile: StorageProfile, source: str, season: int, rows: int = 3) -> Path:
    """Write a small typed parquet where a bronze dataset would land."""
    directory = profile.zone(Zone.BRONZE, season=season)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{source}_{season}.parquet"
    pl.DataFrame(
        {
            "GAME_ID": [f"{season}0000{i}" for i in range(rows)],
            "EVENTNUM": list(range(rows)),
        }
    ).write_parquet(path)
    return path


def write_flat(profile: StorageProfile, zone: Zone, name: str, rows: int = 2) -> Path:
    """Write ``<zone>/<name>.parquet`` — the silver/gold convention."""
    directory = profile.zone(zone)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    pl.DataFrame({"event_id": list(range(rows))}).write_parquet(path)
    return path


@pytest.fixture
def seeded(profile: StorageProfile) -> StorageProfile:
    """A root with two bronze sources (one across two seasons) and one silver table."""
    write_bronze(profile, "nbastats", 2022, rows=2)
    write_bronze(profile, "nbastats", 2023, rows=3)
    write_bronze(profile, "datanba", 2023, rows=4)
    write_flat(profile, Zone.SILVER, "canonical_event", rows=5)
    return profile


def snapshot(path: Path) -> dict[str, object]:
    """Everything about the catalog that a rebuild must reproduce exactly."""
    connection = duckdb.connect(str(path), read_only=True)
    try:
        schemas = connection.execute(
            "SELECT schema_name FROM duckdb_schemas() WHERE NOT internal ORDER BY 1"
        ).fetchall()
        views = connection.execute(
            "SELECT schema_name, view_name, sql FROM duckdb_views() "
            "WHERE NOT internal ORDER BY 1, 2"
        ).fetchall()
        columns: dict[str, object] = {}
        contents: dict[str, object] = {}
        for schema, view, _sql in views:
            qualified = f'"{schema}"."{view}"'
            columns[f"{schema}.{view}"] = connection.execute(f"DESCRIBE {qualified}").fetchall()
            rows = connection.execute(f"SELECT * FROM {qualified}").fetchall()
            contents[f"{schema}.{view}"] = sorted(repr(row) for row in rows)
    finally:
        connection.close()
    return {"schemas": schemas, "views": views, "columns": columns, "contents": contents}


# --- the acceptance criteria -------------------------------------------------


def test_deleting_the_database_and_rebuilding_is_identical(seeded: StorageProfile) -> None:
    """The acceptance criterion, stated as directly as it can be.

    Not just the same row counts: the same schemas, the same view definitions,
    the same column types, and the same rows.
    """
    first = rebuild(seeded)
    before = snapshot(first.database)

    first.database.unlink()
    assert not first.database.exists()

    second = rebuild(seeded)
    assert snapshot(second.database) == before


def test_rebuild_publishes_a_schema_per_zone_and_a_view_per_dataset(
    seeded: StorageProfile,
) -> None:
    result = rebuild(seeded)
    assert {spec.qualified_name for spec in result.created} == {
        "bronze.nbastats",
        "bronze.datanba",
        "silver.canonical_event",
    }

    state = snapshot(result.database)
    assert [row[0] for row in state["schemas"]] == sorted(zone.value for zone in CATALOG_ZONES)  # type: ignore[index]


def test_no_data_is_lost_through_the_views(seeded: StorageProfile) -> None:
    """Every parquet row reaches the view it belongs to."""
    result = rebuild(seeded)
    connection = connect(seeded)
    try:
        assert connection.execute("SELECT count(*) FROM bronze.nbastats").fetchone() == (5,)
        assert connection.execute("SELECT count(*) FROM bronze.datanba").fetchone() == (4,)
        assert connection.execute("SELECT count(*) FROM silver.canonical_event").fetchone() == (5,)
    finally:
        connection.close()
    assert result.database.exists()


# --- the season column -------------------------------------------------------


def test_the_season_partition_key_comes_back_as_a_column(seeded: StorageProfile) -> None:
    rebuild(seeded)
    connection = connect(seeded)
    try:
        rows = connection.execute(
            "SELECT season, count(*) FROM bronze.nbastats GROUP BY 1 ORDER BY 1"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [(2022, 2), (2023, 3)]


def test_season_is_a_bigint_and_leads_the_projection(seeded: StorageProfile) -> None:
    rebuild(seeded)
    connection = connect(seeded)
    try:
        described = connection.execute("DESCRIBE bronze.nbastats").fetchall()
    finally:
        connection.close()
    assert described[0][1] == "BIGINT"
    # Exactly the parquet's own columns, with season in front. Nothing the read
    # added leaks in: not ``filename``, and not a second season column from
    # auto-detected hive partitioning, which DuckDB would silently admit under
    # a disambiguated name rather than reject.
    assert [row[0] for row in described] == ["season", "GAME_ID", "EVENTNUM"]


def test_min_and_max_over_season_do_not_trip_the_optimizer(seeded: StorageProfile) -> None:
    """Regression for the DuckDB 1.5.5 hive-column bug.

    ``min``/``max`` over a hive-partition column raises ``InternalException``
    from statistics propagation, so the season column is derived from the
    filename instead. Nothing else in the suite would notice a regression here:
    counts and GROUP BY work fine either way.
    """
    rebuild(seeded)
    connection = connect(seeded)
    try:
        assert connection.execute(
            "SELECT min(season), max(season) FROM bronze.nbastats"
        ).fetchone() == (2022, 2023)
    finally:
        connection.close()


def test_a_new_season_is_visible_without_a_rebuild(seeded: StorageProfile) -> None:
    """The views glob across seasons, which is the point of keeping the glob."""
    rebuild(seeded)
    write_bronze(seeded, "nbastats", 2024, rows=7)
    connection = connect(seeded)
    try:
        assert connection.execute("SELECT count(*) FROM bronze.nbastats").fetchone() == (12,)
    finally:
        connection.close()


# --- absence is not an error -------------------------------------------------


def test_a_declared_source_with_no_parquet_is_skipped_not_omitted(
    seeded: StorageProfile,
) -> None:
    result = rebuild(seeded)
    skipped = {spec.name for spec in result.skipped}
    assert skipped == set(BRONZE_SCHEMAS) - {"nbastats", "datanba"}
    assert skipped, "the fixture must leave at least one declared source un-ingested"


def test_rebuild_on_an_empty_root_succeeds(profile: StorageProfile) -> None:
    """A glob matching nothing errors at CREATE VIEW, so discovery must come first."""
    result = rebuild(profile)
    assert result.created == ()
    assert {spec.name for spec in result.skipped} == set(BRONZE_SCHEMAS)

    state = snapshot(result.database)
    assert [row[0] for row in state["schemas"]] == sorted(zone.value for zone in CATALOG_ZONES)  # type: ignore[index]
    assert state["views"] == []


def test_planned_views_touches_nothing(profile: StorageProfile) -> None:
    specs = planned_views(profile)
    assert specs
    assert not profile.duckdb_path().exists()
    assert not any(spec.present for spec in specs)


# --- the database is disposable ----------------------------------------------


def test_rebuild_replaces_rather_than_accumulates(seeded: StorageProfile) -> None:
    """State that is not derivable from parquet must not survive a rebuild."""
    rebuild(seeded)
    connection = connect(seeded, read_only=False)
    try:
        connection.execute("CREATE TABLE bronze.hand_written AS SELECT 1 AS x")
    finally:
        connection.close()

    rebuild(seeded)
    connection = connect(seeded)
    try:
        remaining = connection.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE NOT internal"
        ).fetchone()
    finally:
        connection.close()
    assert remaining == (0,)


def test_connect_on_a_missing_database_says_how_to_build_it(profile: StorageProfile) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        connect(profile)
    assert "hoopstate.db.catalog" in str(excinfo.value)


# --- the CLI -----------------------------------------------------------------


def test_cli_list_reports_present_and_skipped_without_building(
    seeded: StorageProfile,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ENV_PROFILE, "ephemeral")
    monkeypatch.setenv(ENV_ROOT, str(seeded.root))

    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "bronze.nbastats" in out
    assert "silver.canonical_event" in out
    assert "skipped" in out
    assert not seeded.duckdb_path().exists()


def test_cli_default_action_rebuilds(
    seeded: StorageProfile,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ENV_PROFILE, "ephemeral")
    monkeypatch.setenv(ENV_ROOT, str(seeded.root))

    assert main([]) == 0
    assert "bronze.nbastats" in capsys.readouterr().out
    assert seeded.duckdb_path().exists()
