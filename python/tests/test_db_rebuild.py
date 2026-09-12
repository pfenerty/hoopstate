"""Tests for the DuckDB rebuild helper (``hoops-1lg.1.4``).

These assert the acceptance criteria directly:

* A documented convention for relations over each zone (bronze union views,
  silver views, gold materialized tables; raw/cache/oracle excluded).
* One command rebuilds the database from parquet with no data loss.
* Deleting the DuckDB file and rebuilding produces identical query results.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from hoopstate.db.rebuild import (
    Relation,
    plan_relations,
    rebuild_database,
)
from hoopstate.storage import (
    ENV_COLD,
    ENV_HOT,
    ENV_PROFILE,
    StorageProfile,
    Zone,
    resolve_profile,
)


@pytest.fixture
def profile(tmp_path: Path) -> StorageProfile:
    """An ephemeral profile rooted at a throwaway directory.

    Both tiers are pinned to ``tmp_path`` so the split collapses (as it does for
    a real ephemeral profile) and nothing leaks into the shared scratch root.
    """
    return resolve_profile(
        env={ENV_PROFILE: "ephemeral", ENV_HOT: str(tmp_path), ENV_COLD: str(tmp_path)}
    )


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _seed_bronze(profile: StorageProfile) -> None:
    """Two seasons of one source plus a playoff file, and a second source."""
    reg_2023 = profile.zone(Zone.BRONZE, season=2023) / "nbastats_2023.parquet"
    reg_2024 = profile.zone(Zone.BRONZE, season=2024) / "nbastats_2024.parquet"
    po_2023 = profile.zone(Zone.BRONZE, season=2023) / "nbastats_po_2023.parquet"
    datanba = profile.zone(Zone.BRONZE, season=2023) / "datanba_2023.parquet"
    _write_parquet(reg_2023, pl.DataFrame({"GAME_ID": ["1", "2"], "EVENTNUM": [1, 2]}))
    _write_parquet(reg_2024, pl.DataFrame({"GAME_ID": ["3"], "EVENTNUM": [1]}))
    _write_parquet(po_2023, pl.DataFrame({"GAME_ID": ["9"], "EVENTNUM": [1]}))
    _write_parquet(datanba, pl.DataFrame({"GAME_ID": ["1"], "evt": [1], "oftid": [10]}))


def test_plan_groups_bronze_by_source(profile: StorageProfile) -> None:
    _seed_bronze(profile)
    relations = {r.name: r for r in plan_relations(profile)}
    assert set(relations) == {"bronze_nbastats", "bronze_datanba"}
    # nbastats spans two regular-season files plus one playoff file.
    assert len(relations["bronze_nbastats"].sources) == 3
    assert all(not r.materialized for r in relations.values())


def test_bronze_view_unions_seasons_and_adds_partition_columns(profile: StorageProfile) -> None:
    _seed_bronze(profile)
    result = rebuild_database(profile=profile)
    con = duckdb.connect(str(result.db_path), read_only=True)
    try:
        # All three nbastats files unioned: 2 + 1 + 1 = 4 rows.
        assert con.execute("SELECT count(*) FROM bronze_nbastats").fetchone()[0] == 4
        # season and playoffs are surfaced from the file name, not the columns.
        rows = con.execute(
            "SELECT season, playoffs, count(*) FROM bronze_nbastats "
            "GROUP BY season, playoffs ORDER BY season, playoffs"
        ).fetchall()
        assert rows == [(2023, False, 2), (2023, True, 1), (2024, False, 1)]
    finally:
        con.close()


def test_no_data_loss_row_counts_match_parquet(profile: StorageProfile) -> None:
    _seed_bronze(profile)
    result = rebuild_database(profile=profile)
    con = duckdb.connect(str(result.db_path), read_only=True)
    try:
        view_total = con.execute("SELECT count(*) FROM bronze_nbastats").fetchone()[0]
    finally:
        con.close()
    parquet_total = sum(
        pl.read_parquet(p).height
        for r in result.relations
        if r.name == "bronze_nbastats"
        for p in r.sources
    )
    assert view_total == parquet_total


def test_silver_is_a_view_gold_is_materialized(profile: StorageProfile) -> None:
    _write_parquet(
        profile.zone(Zone.SILVER) / "possession.parquet",
        pl.DataFrame({"possession_id": [1, 2, 3]}),
    )
    _write_parquet(
        profile.zone(Zone.GOLD) / "team_ratings.parquet",
        pl.DataFrame({"team_id": [10], "rating": [1.5]}),
    )
    result = rebuild_database(profile=profile)
    kinds = {r.name: r.materialized for r in result.relations}
    assert kinds["silver_possession"] is False
    assert kinds["gold_team_ratings"] is True

    con = duckdb.connect(str(result.db_path), read_only=True)
    try:
        catalog = dict(
            con.execute(
                "SELECT table_name, table_type FROM information_schema.tables "
                "WHERE table_name IN ('silver_possession', 'gold_team_ratings')"
            ).fetchall()
        )
    finally:
        con.close()
    assert catalog["silver_possession"] == "VIEW"
    assert catalog["gold_team_ratings"] == "BASE TABLE"


def test_oracle_and_archive_zones_are_never_surfaced(profile: StorageProfile) -> None:
    _seed_bronze(profile)
    # Oracle parquet exists on disk but must not become a relation (quarantined).
    _write_parquet(
        profile.zone(Zone.ORACLE) / "pbpstats_2023.parquet",
        pl.DataFrame({"x": [1]}),
    )
    # Raw/cache hold archives, not tables.
    (profile.zone(Zone.RAW)).mkdir(parents=True, exist_ok=True)
    result = rebuild_database(profile=profile)
    names = {r.name for r in result.relations}
    assert not any(n.startswith(("oracle", "raw", "cache")) for n in names)


def test_delete_and_rebuild_produces_identical_results(profile: StorageProfile) -> None:
    """The core invariant: the DuckDB file is disposable."""
    _seed_bronze(profile)
    _write_parquet(
        profile.zone(Zone.GOLD) / "team_ratings.parquet",
        pl.DataFrame({"team_id": [10, 20], "rating": [1.5, -0.5]}),
    )

    def snapshot() -> dict[str, list[tuple]]:
        result = rebuild_database(profile=profile)
        con = duckdb.connect(str(result.db_path), read_only=True)
        try:
            return {
                "bronze": con.execute(
                    "SELECT * FROM bronze_nbastats ORDER BY GAME_ID, season, playoffs"
                ).fetchall(),
                "gold": con.execute("SELECT * FROM gold_team_ratings ORDER BY team_id").fetchall(),
            }
        finally:
            con.close()

    first = snapshot()
    # Delete the database file entirely, then rebuild from the same parquet.
    profile.duckdb_path().unlink()
    second = snapshot()
    assert first == second
    assert first["gold"] == [(10, 1.5), (20, -0.5)]


def test_rebuild_overwrites_an_existing_database(profile: StorageProfile) -> None:
    """A second rebuild after the parquet changes reflects only the new parquet."""
    _seed_bronze(profile)
    rebuild_database(profile=profile)
    # Remove a source's parquet and rebuild; its relation should disappear.
    (profile.zone(Zone.BRONZE, season=2023) / "datanba_2023.parquet").unlink()
    result = rebuild_database(profile=profile)
    names = {r.name for r in result.relations}
    assert "bronze_datanba" not in names
    assert "bronze_nbastats" in names


def test_in_database_gold_mart_is_materialized_from_relations(profile: StorageProfile) -> None:
    _seed_bronze(profile)
    marts = {"event_counts": "SELECT season, count(*) AS n FROM bronze_nbastats GROUP BY season"}
    result = rebuild_database(profile=profile, gold_marts=marts)
    mart = next(r for r in result.relations if r.name == "gold_event_counts")
    assert mart.materialized is True
    assert mart.sources == ()

    con = duckdb.connect(str(result.db_path), read_only=True)
    try:
        rows = con.execute("SELECT season, n FROM gold_event_counts ORDER BY season").fetchall()
    finally:
        con.close()
    assert rows == [(2023, 3), (2024, 1)]


def test_empty_zones_rebuild_to_an_empty_database(profile: StorageProfile) -> None:
    """A fresh checkout with no parquet yet still rebuilds cleanly."""
    result = rebuild_database(profile=profile)
    assert result.relations == ()
    assert result.db_path.exists()


def test_db_path_is_on_the_hot_tier(profile: StorageProfile) -> None:
    result = rebuild_database(profile=profile)
    assert result.db_path == profile.duckdb_path()
    assert result.db_path.is_relative_to(profile.hot_root)


def test_relation_is_immutable() -> None:
    relation = Relation(name="bronze_x", zone=Zone.BRONZE, materialized=False, sources=())
    with pytest.raises((AttributeError, TypeError)):
        relation.name = "y"  # type: ignore[misc]
