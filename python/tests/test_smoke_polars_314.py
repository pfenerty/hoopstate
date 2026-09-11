"""Smoke test: polars actually works on Python 3.14.

polars publishes `polars` as a pure-Python shim that depends on `polars-runtime-*`,
which ships `cp310-abi3` wheels. abi3 is the stable ABI, so those wheels are
forward-compatible with any CPython >= 3.10 — 3.14 included. But polars' own
classifiers stop at 3.13, so 3.14 is installable-and-should-work rather than
vendor-tested.

This test exercises the specific operations the pipeline depends on. If it fails,
the documented fallback is Python 3.13 (`python313` in the flox catalog); nothing
in the design depends on 3.14.
"""

import sys

import polars as pl


def test_running_on_expected_interpreter() -> None:
    assert sys.version_info >= (3, 14), f"expected Python >= 3.14, got {sys.version}"


def test_dataframe_construction_and_dtypes() -> None:
    df = pl.DataFrame(
        {
            "game_id": ["0022300001"] * 4,
            "event_order": [1, 2, 3, 4],
            "team_id": [1610612738, 1610612738, 1610612751, 1610612751],
            "event_type": ["made_fg", "def_reb", "turnover", "made_fg"],
        }
    )
    assert df.height == 4
    assert df.schema["event_order"] == pl.Int64


def test_group_by_aggregation() -> None:
    df = pl.DataFrame({"team_id": [1, 1, 2], "points": [2, 3, 2]})
    out = df.group_by("team_id").agg(pl.col("points").sum()).sort("team_id")
    assert out["points"].to_list() == [5, 2]


def test_shift_over_window() -> None:
    """The core sequence-feature primitive: previous event type within a game.

    This is the operation the whole `event_context` table is built on, so it is
    the one that actually matters for the 3.14 decision.
    """
    df = pl.DataFrame(
        {
            "game_id": ["A", "A", "A", "B", "B"],
            "event_order": [1, 2, 3, 1, 2],
            "event_type": ["made_fg", "def_reb", "turnover", "missed_fg", "off_reb"],
        }
    )
    out = df.sort("game_id", "event_order").with_columns(
        pl.col("event_type").shift(1).over("game_id").alias("prev_event_type")
    )
    prev = out["prev_event_type"].to_list()
    assert prev == [None, "made_fg", "def_reb", None, "missed_fg"], (
        "shift/over must not leak across game boundaries"
    )


def test_cumulative_over_window() -> None:
    df = pl.DataFrame({"game_id": ["A", "A", "B"], "points": [2, 3, 2]})
    out = df.with_columns(pl.col("points").cum_sum().over("game_id").alias("running"))
    assert out["running"].to_list() == [2, 5, 2]


def test_parquet_round_trip(tmp_path) -> None:
    df = pl.DataFrame(
        {
            "game_id": ["0022300001", "0022300002"],
            "elapsed_seconds": [12.5, 2880.0],
            "lineup_id": ["1-2-3-4-5", "6-7-8-9-10"],
        }
    )
    path = tmp_path / "round_trip.parquet"
    df.write_parquet(path)
    back = pl.read_parquet(path)
    assert back.equals(df)
    assert back.schema == df.schema


def test_lazy_scan_parquet(tmp_path) -> None:
    """Lazy scan is how the pipeline reads bronze without loading a season into RAM."""
    path = tmp_path / "lazy.parquet"
    pl.DataFrame({"a": list(range(1000)), "g": ["x"] * 500 + ["y"] * 500}).write_parquet(path)
    out = pl.scan_parquet(path).filter(pl.col("g") == "y").select(pl.col("a").min()).collect()
    assert out.item() == 500
