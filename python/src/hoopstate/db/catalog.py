"""DuckDB view conventions and the rebuild helper (``hoops-1lg.1.4``).

Parquet is the source of truth. The DuckDB database holds nothing that is not
derivable from it — today only views, later materialized gold marts — so it is
always safe to delete and rebuild. This module is what makes that claim
executable rather than aspirational: it is the only module that writes SQL DDL,
the same way :mod:`hoopstate.storage` is the only one that knows physical
layout.

The convention
--------------

**One schema per zone, one view per dataset.** A zone becomes a DuckDB schema
and each parquet dataset in it becomes a view inside that schema::

    SELECT * FROM bronze.nbastats WHERE season = 2023;
    SELECT * FROM silver.canonical_event;      -- when E3 lands

So the zone/table split is structural rather than a naming habit: ``SHOW ALL
TABLES`` groups by zone for free, and a zone with nothing on disk shows up as an
empty schema instead of an absence you have to notice.

Bronze is season-partitioned (``bronze/season=2023/nbastats_2023.parquet``), so
its views glob across seasons and restore the partition key as a column. Silver
and gold are flat: ``<zone>/<name>.parquet`` becomes ``<zone>.<name>``.

Two things about the generated SQL are deliberate and worth not "simplifying"
later:

*The season column is derived from the filename, not from hive partitioning.*
DuckDB 1.5.5 raises ``InternalException: Attempted to access index N within
vector of size N`` from the statistics-propagation optimizer when ``min()`` or
``max()`` is applied to a hive-partition column. ``count(*)``, ``DISTINCT`` and
``GROUP BY`` are unaffected, which is what makes the bug easy to ship past.
Casting inside the view does not help — the optimizer runs first. Reading the
season out of ``filename`` instead produces an ordinary computed column and
sidesteps the whole mechanism, while keeping the glob, so a newly ingested
season shows up without a rebuild.

*Views are created only where parquet actually exists.* ``read_parquet`` over a
glob that matches nothing fails at ``CREATE VIEW`` time with ``IOException: No
files found that match the pattern``. Declaring a view per known source
unconditionally would therefore fail on every checkout that has not ingested
that source — which is every checkout, for silver and gold. So the *names* are
declared (bronze's come from :data:`~hoopstate.ingest.bronze.BRONZE_SCHEMAS`)
while the catalog reflects what is on disk, and :func:`planned_views` reports
the difference rather than hiding it.

The quarantine
--------------

:data:`CATALOG_ZONES` is an explicit allow-list, not "every zone except one".
The validation answer key lives in its own zone that only ``hoopstate.validate``
may name, and ``tests/test_oracle_quarantine.py`` scans this module along with
every other. Deriving the list by exclusion would name the excluded zone and
fail that scan — correctly, since a catalog that published the answer key as a
queryable view would put it one ``JOIN`` away from the code being graded.
Bronze view names coming from the declared schema registry is a second barrier:
the quarantined source has no bronze schema, so it cannot acquire a view by
being present on disk.

Usage::

    python -m hoopstate.db.catalog          # rebuild the database
    python -m hoopstate.db.catalog --list   # show planned views, touch nothing
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from hoopstate.ingest.bronze import BRONZE_SCHEMAS
from hoopstate.storage import StorageProfile, Zone, resolve_profile

__all__ = [
    "CATALOG_ZONES",
    "RebuildResult",
    "ViewSpec",
    "connect",
    "planned_views",
    "rebuild",
]

# The zones the catalog publishes, as an explicit allow-list. See the module
# docstring: this must never be derived by excluding a zone.
CATALOG_ZONES: tuple[Zone, ...] = (Zone.BRONZE, Zone.SILVER, Zone.GOLD)

# Pulls the season out of the partition directory baked into ``filename``.
_SEASON_FROM_FILENAME = r"season=(\d+)"


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier, so a dataset name never has to be an identifier."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _quote_literal(value: str) -> str:
    """Quote a SQL string literal. Backslashes are literal in DuckDB strings."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


@dataclass(frozen=True)
class ViewSpec:
    """One view the catalog would publish, and whether its parquet is there.

    ``pattern`` is the glob handed to ``read_parquet``; ``present`` records
    whether it currently matches at least one file. A spec with
    ``present=False`` is reported rather than dropped, so ``--list`` can say
    "declared but not ingested" instead of staying silent.
    """

    zone: Zone
    name: str
    pattern: str
    present: bool

    @property
    def qualified_name(self) -> str:
        """The name the view is queried by, e.g. ``bronze.nbastats``."""
        return f"{self.zone.value}.{self.name}"

    def create_sql(self) -> str:
        """The ``CREATE VIEW`` statement for this spec."""
        target = f"{_quote_identifier(self.zone.value)}.{_quote_identifier(self.name)}"
        source = _quote_literal(self.pattern)
        if self.zone is Zone.BRONZE:
            # Restore the season partition key as an ordinary computed column,
            # leading the projection because that is what it is. BIGINT matches
            # bronze's rule that integers are Int64.
            #
            # ``hive_partitioning`` must be turned off explicitly: it is
            # auto-detected from the ``season=`` directory, and the column it
            # adds is both the one that trips the optimizer bug and a duplicate
            # of the one computed here, which DuckDB silently renames rather
            # than rejecting.
            return (
                f"CREATE VIEW {target} AS SELECT CAST(regexp_extract(filename, "
                f"{_quote_literal(_SEASON_FROM_FILENAME)}, 1) AS BIGINT) AS season, "
                f"* EXCLUDE (filename) FROM read_parquet({source}, filename = true, "
                f"hive_partitioning = false)"
            )
        return f"CREATE VIEW {target} AS SELECT * FROM read_parquet({source})"

    def format(self) -> str:
        """One line for the CLI."""
        suffix = self.pattern if self.present else f"{self.pattern}  (no parquet — skipped)"
        return f"  {self.qualified_name:<32} {suffix}"


@dataclass(frozen=True)
class RebuildResult:
    """The outcome of one rebuild: what was created, and what was not."""

    database: Path
    schemas: tuple[Zone, ...]
    created: tuple[ViewSpec, ...]
    skipped: tuple[ViewSpec, ...]

    def format(self) -> str:
        count = len(self.created)
        lines = [
            f"{self.database}: {count} view{'' if count == 1 else 's'} "
            f"in {len(self.schemas)} schemas"
        ]
        for spec in self.created:
            lines.append(spec.format())
        for spec in self.skipped:
            lines.append(spec.format())
        return "\n".join(lines)


def _bronze_views(profile: StorageProfile) -> list[ViewSpec]:
    """One spec per *declared* bronze source, present or not.

    Names come from the schema registry rather than from filenames on disk: a
    source with no declared schema has no bronze parquet to publish, and
    inventing a view for whatever happens to be in the directory would undo
    that.
    """
    root = profile.zone(Zone.BRONZE)
    specs: list[ViewSpec] = []
    for source in sorted(BRONZE_SCHEMAS):
        relative = f"season=*/{source}_*.parquet"
        pattern = (root / relative).as_posix()
        present = any(root.glob(relative))
        specs.append(ViewSpec(zone=Zone.BRONZE, name=source, pattern=pattern, present=present))
    return specs


def _flat_views(profile: StorageProfile, zone: Zone) -> list[ViewSpec]:
    """One spec per ``<zone>/<name>.parquet`` actually on disk.

    Silver and gold have no declared registry yet (E3 onward), so here the
    filesystem is the registry; every spec returned is by construction present.
    """
    root = profile.zone(zone)
    specs: list[ViewSpec] = []
    for path in sorted(root.glob("*.parquet")):
        specs.append(ViewSpec(zone=zone, name=path.stem, pattern=path.as_posix(), present=True))
    return specs


def planned_views(profile: StorageProfile | None = None) -> tuple[ViewSpec, ...]:
    """Every view :func:`rebuild` would consider, in catalog order.

    Pure: touches no database and creates no directories. Specs with
    ``present=False`` are declared sources whose parquet has not been ingested.
    """
    if profile is None:
        profile = resolve_profile()

    specs: list[ViewSpec] = []
    for zone in CATALOG_ZONES:
        if zone is Zone.BRONZE:
            specs.extend(_bronze_views(profile))
        else:
            specs.extend(_flat_views(profile, zone))
    return tuple(specs)


def rebuild(profile: StorageProfile | None = None) -> RebuildResult:
    """Rebuild the profile's DuckDB database from parquet, from scratch.

    The existing file is deleted rather than updated. Nothing in the database
    is unique to it, and views are metadata — no parquet is scanned to create
    one — so a full rebuild is both cheap and trivially idempotent, which makes
    "delete the file and rebuild" the ordinary path rather than a special case.
    """
    if profile is None:
        profile = resolve_profile()

    specs = planned_views(profile)
    created = tuple(spec for spec in specs if spec.present)
    skipped = tuple(spec for spec in specs if not spec.present)

    path = profile.duckdb_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # The write-ahead log is part of the database; leaving a stale one behind
    # would let deleted state reappear.
    for stale in (path, path.with_name(path.name + ".wal")):
        stale.unlink(missing_ok=True)

    connection = duckdb.connect(str(path))
    try:
        for zone in CATALOG_ZONES:
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_identifier(zone.value)}")
        for spec in created:
            connection.execute(spec.create_sql())
    finally:
        connection.close()

    return RebuildResult(database=path, schemas=CATALOG_ZONES, created=created, skipped=skipped)


def connect(
    profile: StorageProfile | None = None, *, read_only: bool = True
) -> duckdb.DuckDBPyConnection:
    """Open the profile's database. Read-only by default.

    Querying is the normal reason to open it; anything that would change its
    shape belongs in :func:`rebuild`, so writing is opt-in.
    """
    if profile is None:
        profile = resolve_profile()

    path = profile.duckdb_path()
    if read_only and not path.exists():
        raise FileNotFoundError(
            f"no database at {path}; run `python -m hoopstate.db.catalog` to build it"
        )
    return duckdb.connect(str(path), read_only=read_only)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hoopstate.db.catalog [--list]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild the DuckDB catalog of views over parquet."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show the views a rebuild would create, without touching the database.",
    )
    args = parser.parse_args(argv)

    profile = resolve_profile()
    if args.list:
        specs = planned_views(profile)
        print(f"{profile.duckdb_path()}: {sum(s.present for s in specs)} of {len(specs)} views")
        for spec in specs:
            print(spec.format())
        return 0

    print(rebuild(profile).format())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
