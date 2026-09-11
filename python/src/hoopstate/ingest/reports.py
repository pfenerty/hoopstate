"""Re-runnable bronze quality reports (``hoops-1lg.2.3``).

Every claim this project makes about how its sources fit together — "shotdetail
joins cleanly to nbastats", "matchups covers 1227 of 1230 games" — is a
measurement, and a measurement recorded as prose in a docstring rots the moment a
new season lands. This module makes those claims executable instead: each is a
named report that reads bronze parquet and prints the numbers for whichever
season you ask for.

Two primitives cover everything asked of the ingestion zone so far:

:func:`join_report`
    How well one table's key resolves against another's. Reports the match rate,
    a bounded sample of the keys that failed, and — separately — how many keys on
    the right side are duplicated, because a duplicated join key silently
    multiplies rows downstream and is invisible in a match rate.

:func:`key_coverage`
    How much of a reference key space a table actually populates, and how many
    distinct values it carries per group. Reports the reference keys with no rows
    at all alongside the min/median/mean/max spread.

Both return frozen dataclasses rather than printing. Reporting is the CLI's job,
so the numbers stay callable from tests and from later validation work — E7's
oracle comparison harness (``hoops-1lg.7.1``) is meant to build on these rather
than reinvent match-rate reporting, and because neither primitive knows anything
about the oracle, they carry over unchanged to the E10 Rust port's
implementation-versus-implementation comparison.

Every path resolves through :mod:`hoopstate.storage`, so nothing here learns
which storage profile is active.

Usage::

    python -m hoopstate.ingest.reports shotdetail_event_join --season 2023
    python -m hoopstate.ingest.reports --list
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hoopstate.ingest.bronze import bronze_parquet_path
from hoopstate.storage import StorageProfile, resolve_profile

__all__ = [
    "REPORTS",
    "CoverageReport",
    "JoinReport",
    "join_report",
    "key_coverage",
    "run_report",
]

# How many failing keys a report carries as evidence. Enough to start an
# investigation, bounded so a badly broken join does not produce a report the
# size of the table it is describing.
_SAMPLE = 10


def _format_key(key: tuple[object, ...]) -> str:
    """Render one group/join key. Composite keys join on ``/``."""
    return "/".join(str(part) for part in key)


def _sample_line(label: str, sample: tuple[tuple[object, ...], ...], total: int) -> list[str]:
    """Render a bounded evidence list as a footer line, or nothing if empty."""
    if not sample:
        return []
    shown = ", ".join(_format_key(key) for key in sample)
    scope = f"first {len(sample)} of {total:,}" if len(sample) < total else f"{total:,}"
    return [f"  {label} ({scope}): {shown}"]


def _table(rows: Sequence[tuple[str, str, str]]) -> list[str]:
    """Render ``(metric, value, detail)`` rows as an aligned table.

    Values are right-aligned so magnitudes line up and an outlier is visible at
    a glance. The detail column is dropped entirely when no row uses it.
    """
    has_detail = any(detail for _, _, detail in rows)
    headers = ("metric", "value", "detail") if has_detail else ("metric", "value")
    width = [
        max(len(headers[i]), max((len(row[i]) for row in rows), default=0))
        for i in range(len(headers))
    ]
    lines = [
        f"  {headers[0]:<{width[0]}}  {headers[1]:>{width[1]}}"
        + (f"  {headers[2]}" if has_detail else ""),
        "  " + "  ".join("-" * w for w in width),
    ]
    for metric, value, detail in rows:
        line = f"  {metric:<{width[0]}}  {value:>{width[1]}}"
        if has_detail:
            line += f"  {detail}"
        lines.append(line.rstrip())
    return lines


@dataclass(frozen=True)
class JoinReport:
    """How well ``left``'s key resolves against ``right``'s.

    ``duplicate_right_keys`` is deliberately separate from the match rate: a
    duplicated key on the right side matches perfectly and still corrupts every
    downstream aggregate by fanning rows out, so a report that folded the two
    together would hide the more dangerous of the two problems.
    """

    name: str
    left: str
    right: str
    left_key: tuple[str, ...]
    right_key: tuple[str, ...]
    left_rows: int
    matched: int
    unmatched: int
    duplicate_right_keys: int
    unmatched_sample: tuple[tuple[object, ...], ...]

    @property
    def match_rate(self) -> float:
        """Fraction of left rows whose key was found on the right. Empty is 1.0."""
        if self.left_rows == 0:
            return 1.0
        return self.matched / self.left_rows

    def format(self) -> str:
        rows = [
            (f"rows in {self.left}", f"{self.left_rows:,}", ""),
            (f"rows matched into {self.right}", f"{self.matched:,}", f"{self.match_rate:.6%}"),
            ("rows matching nothing", f"{self.unmatched:,}", ""),
            (
                f"duplicate keys in {self.right}",
                f"{self.duplicate_right_keys:,}",
                "each one fans rows out downstream" if self.duplicate_right_keys else "",
            ),
        ]
        return "\n".join(
            [
                f"{self.name} — {self.left} -> {self.right}",
                f"joined on ({', '.join(self.left_key)}) -> ({', '.join(self.right_key)})",
                "",
                *_table(rows),
                *(
                    [
                        "",
                        *_sample_line(
                            "keys matching nothing", self.unmatched_sample, self.unmatched
                        ),
                    ]
                    if self.unmatched_sample
                    else []
                ),
            ]
        )


def join_report(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_on: Sequence[str],
    right_on: Sequence[str],
    name: str,
    left_label: str = "left",
    right_label: str = "right",
    sample: int = _SAMPLE,
) -> JoinReport:
    """Measure how ``left``'s key resolves against ``right``'s.

    The right side is reduced to its distinct keys before joining, so the report
    itself never fans out; duplication is counted and reported instead.
    """
    left_on = tuple(left_on)
    right_on = tuple(right_on)

    marker = "__hoopstate_matched"
    right_keys = right.select(right_on).unique().with_columns(pl.lit(True).alias(marker))
    joined = left.join(right_keys, left_on=left_on, right_on=right_on, how="left")

    unmatched_frame = joined.filter(pl.col(marker).is_null())
    unmatched = unmatched_frame.height
    duplicates = right.group_by(right_on).len().filter(pl.col("len") > 1).height

    return JoinReport(
        name=name,
        left=left_label,
        right=right_label,
        left_key=left_on,
        right_key=right_on,
        left_rows=left.height,
        matched=left.height - unmatched,
        unmatched=unmatched,
        duplicate_right_keys=duplicates,
        # Sorted so re-running against unchanged data yields an identical report.
        unmatched_sample=tuple(
            unmatched_frame.select(left_on).unique().sort(list(left_on)).head(sample).rows()
        ),
    )


@dataclass(frozen=True)
class CoverageReport:
    """How much of a reference key space a dataset populates, and how densely.

    ``units`` counts groups the dataset actually has rows for; ``missing_units``
    counts reference keys it has none for. The distribution describes distinct
    values of ``measure`` per group — for matchups, distinct player ids per game.

    ``minimum_key`` and ``maximum_key`` name the groups at the extremes, so an
    outlier is directly investigable rather than merely visible as a number.
    Where several groups tie, the lexicographically first key is reported, which
    keeps a re-run of the same data diffable against the previous one. Both are
    ``None`` when the frame has no rows.
    """

    name: str
    dataset: str
    unit: str
    measure: str
    units: int
    reference_units: int | None
    missing_units: int
    missing_sample: tuple[tuple[object, ...], ...]
    minimum: int
    minimum_key: tuple[object, ...] | None
    median: float
    mean: float
    maximum: int
    maximum_key: tuple[object, ...] | None

    def format(self) -> str:
        def where(key: tuple[object, ...] | None) -> str:
            return f"{self.unit} {_format_key(key)}" if key else ""

        rows = [
            (
                f"{self.unit}s with data",
                f"{self.units:,}",
                "" if self.reference_units is None else f"of {self.reference_units:,} expected",
            )
        ]
        if self.reference_units is not None:
            rows.append((f"{self.unit}s with no rows", f"{self.missing_units:,}", ""))
        rows += [
            (
                f"fewest {self.measure} in a {self.unit}",
                f"{self.minimum:,}",
                where(self.minimum_key),
            ),
            (f"most {self.measure} in a {self.unit}", f"{self.maximum:,}", where(self.maximum_key)),
            (f"mean {self.measure} per {self.unit}", f"{self.mean:,.2f}", ""),
            (f"median {self.measure} per {self.unit}", f"{self.median:,.1f}", ""),
        ]
        return "\n".join(
            [
                f"{self.name} — {self.dataset}",
                f"{self.measure} per {self.unit}",
                "",
                *_table(rows),
                *(
                    [
                        "",
                        *_sample_line(
                            f"{self.unit}s with no rows", self.missing_sample, self.missing_units
                        ),
                    ]
                    if self.missing_sample
                    else []
                ),
            ]
        )


def key_coverage(
    frame: pl.DataFrame,
    *,
    group_by: Sequence[str],
    value_columns: Sequence[str],
    name: str,
    dataset: str,
    unit: str,
    measure: str,
    reference_keys: pl.DataFrame | None = None,
    sample: int = _SAMPLE,
) -> CoverageReport:
    """Measure per-group distinct-value counts, optionally against a reference key set.

    ``value_columns`` are pooled before counting, so several columns naming the
    same kind of entity — matchups' offensive ``person_id`` and defensive
    ``matchups_person_id`` — count as one population of players rather than two.
    They must therefore share a dtype.

    ``reference_keys`` is a frame of the groups that *should* be present (its
    first ``len(group_by)`` columns are used); groups absent from ``frame``
    entirely are counted and sampled.
    """
    group_by = tuple(group_by)
    pooled = pl.concat(
        [frame.select(*group_by, pl.col(column).alias("__value")) for column in value_columns]
    )
    per_group = pooled.group_by(group_by).agg(pl.col("__value").n_unique().alias("__distinct"))
    counts = per_group["__distinct"].to_list()

    # Sorting by the group key first makes ties resolve the same way every run,
    # so re-running a report against unchanged data produces an identical page.
    minimum_key: tuple[object, ...] | None = None
    maximum_key: tuple[object, ...] | None = None
    if counts:
        ordered = per_group.sort(list(group_by))
        extremes = ordered["__distinct"]
        minimum_key = ordered.filter(extremes == extremes.min()).select(group_by).rows()[0]
        maximum_key = ordered.filter(extremes == extremes.max()).select(group_by).rows()[0]

    missing_units = 0
    missing_sample: tuple[tuple[object, ...], ...] = ()
    reference_units: int | None = None
    if reference_keys is not None:
        reference = reference_keys.select(reference_keys.columns[: len(group_by)]).unique()
        reference_units = reference.height
        missing = reference.join(
            per_group.select(group_by), left_on=reference.columns, right_on=group_by, how="anti"
        )
        missing_units = missing.height
        missing_sample = tuple(missing.sort(missing.columns).head(sample).rows())

    return CoverageReport(
        name=name,
        dataset=dataset,
        unit=unit,
        measure=measure,
        units=per_group.height,
        reference_units=reference_units,
        missing_units=missing_units,
        missing_sample=missing_sample,
        minimum=min(counts, default=0),
        minimum_key=minimum_key,
        median=statistics.median(counts) if counts else 0.0,
        mean=statistics.fmean(counts) if counts else 0.0,
        maximum=max(counts, default=0),
        maximum_key=maximum_key,
    )


# --- named reports ----------------------------------------------------------
#
# Each takes a season and a resolved profile and returns one report. Registered
# in REPORTS so the CLI can address them by name.


def _read_bronze(name: str, profile: StorageProfile, columns: Sequence[str]) -> pl.DataFrame:
    """Read the named columns of a bronze dataset, with a pointed error if absent."""
    path: Path = bronze_parquet_path(name, profile=profile)
    if not path.exists():
        raise FileNotFoundError(
            f"{name} is not in bronze at {path}; convert it first with "
            f"`python -m hoopstate.ingest.bronze {name}`"
        )
    return pl.read_parquet(path, columns=list(columns))


def _shotdetail_event_join(season: int, profile: StorageProfile) -> JoinReport:
    """Does every shot resolve to an event in the primary feed?

    This is the join the whole shot-location layer rests on, and it is the one
    place a silent mismatch would strand coordinates on the wrong play.
    """
    shots = _read_bronze(f"shotdetail_{season}", profile, ["GAME_ID", "GAME_EVENT_ID"])
    events = _read_bronze(f"nbastats_{season}", profile, ["GAME_ID", "EVENTNUM"])
    return join_report(
        shots,
        events,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["GAME_ID", "EVENTNUM"],
        name="shotdetail_event_join",
        left_label=f"shotdetail_{season}",
        right_label=f"nbastats_{season}",
    )


def _matchups_game_coverage(season: int, profile: StorageProfile) -> CoverageReport:
    """How many players does the matchups feed observe, per game?

    E4's period-starter research leans on this. Note what it does *not* answer:
    matchups has no period column, so this is the set of players observed
    somewhere in the game, never the set on the floor at a given moment.
    """
    matchups = _read_bronze(
        f"matchups_{season}", profile, ["game_id", "person_id", "matchups_person_id"]
    )
    games = _read_bronze(f"nbastats_{season}", profile, ["GAME_ID"]).unique()
    return key_coverage(
        matchups,
        group_by=["game_id"],
        value_columns=["person_id", "matchups_person_id"],
        name="matchups_game_coverage",
        dataset=f"matchups_{season}",
        unit="game",
        measure="distinct players",
        reference_keys=games,
    )


def _datanba_game_coverage(season: int, profile: StorageProfile) -> CoverageReport:
    """How much of the primary feed's game set does the datanba feed cover?

    datanba supplies the offense-team id that cross-checks event ordering, so a
    game it does not cover is a game with no such cross-check available. This
    began as a prose claim in :mod:`hoopstate.ingest.bronze`'s docstring
    (``hoops-1lg.2.2``) and is measured here instead.
    """
    datanba = _read_bronze(f"datanba_{season}", profile, ["GAME_ID", "pid"])
    games = _read_bronze(f"nbastats_{season}", profile, ["GAME_ID"]).unique()
    return key_coverage(
        datanba,
        group_by=["GAME_ID"],
        value_columns=["pid"],
        name="datanba_game_coverage",
        dataset=f"datanba_{season}",
        unit="game",
        measure="distinct players",
        reference_keys=games,
    )


Report = JoinReport | CoverageReport

REPORTS: dict[str, Callable[[int, StorageProfile], Report]] = {
    "shotdetail_event_join": _shotdetail_event_join,
    "matchups_game_coverage": _matchups_game_coverage,
    "datanba_game_coverage": _datanba_game_coverage,
}


def run_report(name: str, *, season: int, profile: StorageProfile | None = None) -> Report:
    """Run one named report against the bronze zone."""
    try:
        report = REPORTS[name]
    except KeyError:
        known = ", ".join(sorted(REPORTS))
        raise KeyError(f"unknown report {name!r}; known reports: {known}") from None
    return report(season, profile if profile is not None else resolve_profile())


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hoopstate.ingest.reports <name> [--season YEAR]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Run bronze quality reports.")
    parser.add_argument("names", nargs="*", help="Report names; omit with --all to run every one.")
    parser.add_argument("--season", type=int, default=2023, help="Season starting year.")
    parser.add_argument("--all", action="store_true", help="Run every registered report.")
    parser.add_argument("--list", action="store_true", help="List the registered reports and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(REPORTS):
            print(name)
        return 0

    names = sorted(REPORTS) if args.all else args.names
    if not names:
        parser.error("give at least one report name, or --all, or --list")

    for index, name in enumerate(names):
        if index:
            print()
        print(run_report(name, season=args.season).format())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
