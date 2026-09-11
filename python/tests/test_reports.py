"""Tests for the re-runnable bronze quality reports (``hoops-1lg.2.3``).

These reports exist so cross-source facts stay measured rather than asserted in
prose: ``hoops-1lg.2.2`` recorded its datanba game-coverage finding in a module
docstring, which cannot go stale loudly. Everything here runs offline against
synthetic bronze parquet written into an ephemeral storage profile.

Acceptance criteria covered:

* The join primitive separates *unmatched left rows* from *duplicated right
  keys* — a duplicated key matches perfectly yet fans rows out downstream.
* The coverage primitive pools several value columns into one population and
  reports reference groups with no rows at all.
* The three named reports address real bronze datasets by name and fail with a
  usable remedy when a dataset has not been converted yet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from hoopstate.ingest import reports
from hoopstate.ingest.bronze import bronze_parquet_path
from hoopstate.ingest.reports import (
    REPORTS,
    CoverageReport,
    JoinReport,
    join_report,
    key_coverage,
    main,
    run_report,
)
from hoopstate.storage import ENV_COLD, ENV_HOT, ENV_PROFILE, resolve_profile

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def ephemeral_profile(tmp_path: Path):
    return resolve_profile(
        env={ENV_PROFILE: "ephemeral", ENV_HOT: str(tmp_path), ENV_COLD: str(tmp_path)}
    )


def _write_bronze(profile, name: str, frame: pl.DataFrame) -> Path:
    path = bronze_parquet_path(name, profile=profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


# A three-game season. G3 exists in the primary feed and nowhere else, so it is
# the "missing" unit every coverage report should surface. (G1, 5) is duplicated
# in the primary feed: it matches, and it still fans out.
_NBASTATS = pl.DataFrame(
    {
        "GAME_ID": ["G1", "G1", "G1", "G2", "G2", "G3"],
        "EVENTNUM": [1, 5, 5, 1, 2, 1],
    },
    schema={"GAME_ID": pl.String, "EVENTNUM": pl.Int64},
)

_SHOTDETAIL = pl.DataFrame(
    {
        "GAME_ID": ["G1", "G1", "G2", "G1"],
        "GAME_EVENT_ID": [1, 5, 2, 99],  # 99 resolves to nothing
    },
    schema={"GAME_ID": pl.String, "GAME_EVENT_ID": pl.Int64},
)

_MATCHUPS = pl.DataFrame(
    {
        "game_id": ["G1", "G1", "G2"],
        "person_id": [10, 11, 10],
        "matchups_person_id": [20, 20, 21],
    },
    schema={"game_id": pl.String, "person_id": pl.Int64, "matchups_person_id": pl.Int64},
)

_DATANBA = pl.DataFrame(
    {"GAME_ID": ["G1", "G1", "G2"], "pid": [10, 11, 10]},
    schema={"GAME_ID": pl.String, "pid": pl.Int64},
)


@pytest.fixture
def seeded_profile(ephemeral_profile):
    _write_bronze(ephemeral_profile, "nbastats_2023", _NBASTATS)
    _write_bronze(ephemeral_profile, "shotdetail_2023", _SHOTDETAIL)
    _write_bronze(ephemeral_profile, "matchups_2023", _MATCHUPS)
    _write_bronze(ephemeral_profile, "datanba_2023", _DATANBA)
    return ephemeral_profile


# --- join_report ------------------------------------------------------------


def test_join_report_counts_matched_and_unmatched() -> None:
    report = join_report(
        _SHOTDETAIL,
        _NBASTATS,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["GAME_ID", "EVENTNUM"],
        name="j",
        left_label="shots",
        right_label="events",
    )
    assert isinstance(report, JoinReport)
    assert report.left_rows == 4
    assert report.matched == 3
    assert report.unmatched == 1
    assert report.match_rate == pytest.approx(0.75)
    assert report.unmatched_sample == (("G1", 99),)


def test_join_report_does_not_fan_out_on_duplicate_right_keys() -> None:
    # (G1, 5) appears twice on the right. It matches one left row, and the
    # report must say "1 matched", not "2" -- while still flagging the duplicate.
    report = join_report(
        _SHOTDETAIL.filter(pl.col("GAME_EVENT_ID") == 5),
        _NBASTATS,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["GAME_ID", "EVENTNUM"],
        name="j",
    )
    assert report.left_rows == 1
    assert report.matched == 1
    assert report.match_rate == 1.0
    # Reported separately: a perfect match rate does not make this safe.
    assert report.duplicate_right_keys == 1


def test_join_report_empty_left_is_a_perfect_rate_not_a_crash() -> None:
    report = join_report(
        _SHOTDETAIL.head(0),
        _NBASTATS,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["GAME_ID", "EVENTNUM"],
        name="j",
    )
    assert report.left_rows == 0
    assert report.match_rate == 1.0
    assert report.unmatched_sample == ()


def _join_text() -> str:
    return join_report(
        _SHOTDETAIL,
        _NBASTATS,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["GAME_ID", "EVENTNUM"],
        name="shot_join",
        left_label="shots",
        right_label="events",
    ).format()


def test_join_report_format_mentions_both_failure_modes() -> None:
    text = _join_text()
    assert "shot_join" in text
    assert "rows matching nothing" in text
    # Named against the actual dataset, not an opaque "rhs".
    assert "duplicate keys in events" in text


def test_join_report_format_is_an_aligned_table() -> None:
    lines = _join_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.strip().startswith("metric"))
    assert set(lines[header + 1].strip()) <= {"-", " "}
    body = [line for line in lines[header + 2 : header + 6]]
    assert len(body) == 4
    # Values are right-aligned, so every count ends in the same column and a
    # magnitude outlier is visible without reading the numbers.
    value_col = lines[header].index("value") + len("value")
    for line in body:
        assert line[value_col - 1] != " ", f"value not right-aligned to {value_col}: {line!r}"


def test_join_report_sample_is_sorted_for_diffable_reruns() -> None:
    report = join_report(
        _SHOTDETAIL,
        _NBASTATS,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["GAME_ID", "EVENTNUM"],
        name="j",
    )
    assert list(report.unmatched_sample) == sorted(report.unmatched_sample)


# --- key_coverage -----------------------------------------------------------


def test_key_coverage_pools_value_columns_into_one_population() -> None:
    # G1 has person_ids {10, 11} and matchups_person_ids {20}: three distinct
    # players, not "2 offensive and 1 defensive" counted apart.
    report = key_coverage(
        _MATCHUPS,
        group_by=["game_id"],
        value_columns=["person_id", "matchups_person_id"],
        name="c",
        dataset="matchups_2023",
        unit="game",
        measure="distinct players",
    )
    assert isinstance(report, CoverageReport)
    assert report.units == 2
    assert report.maximum == 3  # G1
    assert report.minimum == 2  # G2: {10} and {21}
    assert report.mean == pytest.approx(2.5)
    assert report.reference_units is None
    assert report.missing_units == 0


def test_key_coverage_reports_reference_groups_with_no_rows() -> None:
    games = _NBASTATS.select("GAME_ID").unique()
    report = key_coverage(
        _MATCHUPS,
        group_by=["game_id"],
        value_columns=["person_id", "matchups_person_id"],
        name="c",
        dataset="matchups_2023",
        unit="game",
        measure="distinct players",
        reference_keys=games,
    )
    assert report.units == 2
    assert report.reference_units == 3
    assert report.missing_units == 1
    assert report.missing_sample == (("G3",),)
    text = report.format()
    assert "of 3 expected" in text
    assert "games with no rows" in text
    assert "G3" in text


def test_key_coverage_names_the_extreme_groups() -> None:
    # G1 has three distinct players and G2 has two; knowing *which* game is the
    # outlier is the difference between a number and something investigable.
    report = key_coverage(
        _MATCHUPS,
        group_by=["game_id"],
        value_columns=["person_id", "matchups_person_id"],
        name="c",
        dataset="matchups_2023",
        unit="game",
        measure="distinct players",
    )
    assert report.minimum_key == ("G2",)
    assert report.maximum_key == ("G1",)


def test_key_coverage_ties_resolve_to_the_first_key() -> None:
    # Both games have exactly two distinct players, so the reported extreme must
    # be stable across runs rather than whatever the group-by happened to emit.
    tied = pl.DataFrame(
        {"game_id": ["G2", "G2", "G1", "G1"], "person_id": [1, 2, 3, 4]},
        schema={"game_id": pl.String, "person_id": pl.Int64},
    )
    report = key_coverage(
        tied,
        group_by=["game_id"],
        value_columns=["person_id"],
        name="c",
        dataset="d",
        unit="game",
        measure="distinct players",
    )
    assert report.minimum == report.maximum == 2
    assert report.minimum_key == ("G1",)
    assert report.maximum_key == ("G1",)


def test_coverage_format_names_the_unit_and_measure() -> None:
    report = key_coverage(
        _MATCHUPS,
        group_by=["game_id"],
        value_columns=["person_id", "matchups_person_id"],
        name="c",
        dataset="matchups_2023",
        unit="game",
        measure="distinct players",
    )
    text = report.format()
    # Spelled out, rather than a bare "min / max" the reader has to decode.
    assert "fewest distinct players in a game" in text
    assert "most distinct players in a game" in text
    assert "mean distinct players per game" in text
    # The extreme groups are shown next to their counts.
    assert "game G1" in text
    assert "game G2" in text


def test_key_coverage_empty_frame_is_all_missing() -> None:
    report = key_coverage(
        _MATCHUPS.head(0),
        group_by=["game_id"],
        value_columns=["person_id"],
        name="c",
        dataset="matchups_2023",
        unit="game",
        measure="distinct players",
        reference_keys=_NBASTATS.select("GAME_ID").unique(),
    )
    assert report.units == 0
    assert report.missing_units == 3
    assert (report.minimum, report.maximum) == (0, 0)
    assert report.mean == 0.0
    # No groups, so no extremes to name — and format() must still render.
    assert report.minimum_key is None
    assert report.maximum_key is None
    assert "fewest distinct players in a game" in report.format()


# --- named reports ----------------------------------------------------------


def test_shotdetail_event_join_report(seeded_profile) -> None:
    report = run_report("shotdetail_event_join", season=2023, profile=seeded_profile)
    assert report.left == "shotdetail_2023"
    assert report.right == "nbastats_2023"
    assert (report.left_rows, report.matched, report.unmatched) == (4, 3, 1)
    assert report.duplicate_right_keys == 1


def test_matchups_game_coverage_report(seeded_profile) -> None:
    report = run_report("matchups_game_coverage", season=2023, profile=seeded_profile)
    assert report.dataset == "matchups_2023"
    assert (report.units, report.reference_units, report.missing_units) == (2, 3, 1)
    assert report.missing_sample == (("G3",),)
    assert (report.minimum_key, report.maximum_key) == (("G2",), ("G1",))


def test_datanba_game_coverage_report(seeded_profile) -> None:
    # The hoops-1lg.2.2 prose finding, now measured.
    report = run_report("datanba_game_coverage", season=2023, profile=seeded_profile)
    assert report.dataset == "datanba_2023"
    assert (report.units, report.reference_units, report.missing_units) == (2, 3, 1)


def test_report_on_unconverted_dataset_names_the_remedy(ephemeral_profile) -> None:
    with pytest.raises(FileNotFoundError, match=r"hoopstate\.ingest\.bronze shotdetail_2023"):
        run_report("shotdetail_event_join", season=2023, profile=ephemeral_profile)


def test_run_report_unknown_name_lists_known_reports() -> None:
    with pytest.raises(KeyError, match="unknown report 'nope'"):
        run_report("nope", season=2023)
    try:
        run_report("nope", season=2023)
    except KeyError as exc:
        for known in REPORTS:
            assert known in str(exc)


# --- CLI --------------------------------------------------------------------


def test_cli_list(capsys) -> None:
    assert main(["--list"]) == 0
    printed = capsys.readouterr().out.split()
    assert printed == sorted(REPORTS)


def test_cli_all_runs_every_report(seeded_profile, monkeypatch, capsys) -> None:
    monkeypatch.setenv(ENV_PROFILE, "ephemeral")
    monkeypatch.setenv(ENV_HOT, str(seeded_profile.hot_root))
    monkeypatch.setenv(ENV_COLD, str(seeded_profile.cold_root))
    assert main(["--all", "--season", "2023"]) == 0
    out = capsys.readouterr().out
    for name in REPORTS:
        assert name in out


def test_cli_requires_a_name() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_reports_module_public_api() -> None:
    assert set(reports.__all__) == {
        "REPORTS",
        "CoverageReport",
        "JoinReport",
        "join_report",
        "key_coverage",
        "run_report",
    }
