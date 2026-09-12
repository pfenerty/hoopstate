"""Tests for the local disk budget check (``hoops-1lg.1.3``).

Everything here runs against ``tmp_path`` with injected roots, so nothing
depends on the machine the suite runs on — the same file has to pass on a
developer laptop with 11 GiB free and in a cloud sandbox with 30 GB.

The load-bearing claim is the measurement: that ``disk_usage_of`` agrees with
``du``. Two tests make it, one structurally (a hardlink is counted once) and one
by comparison against ``du`` itself where it exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hoopstate.footprint import (
    Consumer,
    FootprintReport,
    Part,
    Usage,
    consumers,
    disk_usage_of,
    main,
    report,
)
from hoopstate.storage import ENV_PROFILE, ENV_ROOT, Profile, StorageProfile, Zone

_GIB = 1024**3


@pytest.fixture
def profile(tmp_path: Path) -> StorageProfile:
    root = tmp_path / "data"
    root.mkdir()
    return StorageProfile(name=Profile.LOCAL, root=root)


def write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def usage(
    name: str, used: int, budget: int, *, present: bool = True, reclaim: str | None = None
) -> Usage:
    consumer = Consumer(
        name=name, paths=(Path("/nowhere"),), budget=budget, why="", reclaim=reclaim
    )
    return Usage(consumer=consumer, used=used, present=present)


# --- measurement: does this agree with du? -----------------------------------


def test_a_hardlink_is_counted_once(tmp_path: Path) -> None:
    """The claim the whole measurement rests on.

    uv hardlinks out of its cache into the virtualenv, so a naive walk counts
    those bytes twice — once under the venv's budget and once under the cache's
    — and reports a footprint the machine does not actually have.
    """
    linked = tmp_path / "linked"
    write(linked / "a.bin", 200_000)
    os.link(linked / "a.bin", linked / "b.bin")

    copied = tmp_path / "copied"
    write(copied / "a.bin", 200_000)
    write(copied / "b.bin", 200_000)

    linked_bytes, _ = disk_usage_of(linked)
    copied_bytes, _ = disk_usage_of(copied)

    assert linked_bytes < copied_bytes
    # The difference is one whole copy of the payload, not a rounding artifact.
    assert copied_bytes - linked_bytes >= 200_000


def test_apparent_size_would_have_disagreed(tmp_path: Path) -> None:
    """Proof the hardlink test is not passing for a trivial reason.

    Summing ``st_size`` gives the same number for both trees, which is exactly
    the wrong answer and the reason blocks are used instead.
    """
    linked = tmp_path / "linked"
    write(linked / "a.bin", 200_000)
    os.link(linked / "a.bin", linked / "b.bin")

    apparent = sum(p.lstat().st_size for p in linked.rglob("*"))
    assert apparent == 400_000
    measured, _ = disk_usage_of(linked)
    assert measured < apparent


@pytest.mark.skipif(sys.platform == "win32", reason="du is a POSIX tool")
def test_matches_du(tmp_path: Path) -> None:
    """Compared against the tool a reader would check the report against."""
    import shutil
    import subprocess

    du = shutil.which("du")
    if du is None:  # pragma: no cover - present on both supported platforms
        pytest.skip("du not installed")

    tree = tmp_path / "tree"
    write(tree / "one.bin", 50_000)
    write(tree / "nested" / "two.bin", 3_000)
    write(tree / "nested" / "deeper" / "three.bin", 1)
    os.link(tree / "one.bin", tree / "nested" / "hardlink.bin")

    completed = subprocess.run([du, "-sk", str(tree)], capture_output=True, text=True, check=True)
    du_bytes = int(completed.stdout.split()[0]) * 1024

    measured, present = disk_usage_of(tree)
    assert present
    assert measured == du_bytes


def test_a_single_file_measures_its_own_blocks(tmp_path: Path) -> None:
    path = write(tmp_path / "one.bin", 4096)
    measured, present = disk_usage_of(path)
    assert present
    assert measured >= 4096


# --- absence and unreadability are not errors --------------------------------


def test_an_absent_path_is_reported_not_raised(tmp_path: Path) -> None:
    measured, present = disk_usage_of(tmp_path / "does-not-exist")
    assert (measured, present) == (0, False)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_an_unreadable_subtree_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A permission error somewhere deep must not take the whole report down."""
    tree = tmp_path / "tree"
    write(tree / "readable.bin", 10_000)
    locked = tree / "locked"
    write(locked / "hidden.bin", 10_000)
    locked.chmod(0o000)
    try:
        measured, present = disk_usage_of(tree)
    finally:
        locked.chmod(0o700)
    assert present
    assert measured > 0


# --- the registry -------------------------------------------------------------


def test_every_consumer_is_measured(profile: StorageProfile, tmp_path: Path) -> None:
    registry = consumers(profile, repo_root=tmp_path / "repo", home=tmp_path / "home", env={})
    names = [consumer.name for consumer in registry]
    assert names == ["data zones", "rust/target/", "rustup + cargo", "python/.venv", "uv cache"]
    assert all(consumer.budget > 0 for consumer in registry)
    assert all(consumer.why for consumer in registry)


def test_the_paths_that_do_not_exist_yet_are_the_expensive_ones(
    profile: StorageProfile, tmp_path: Path
) -> None:
    """``rust/target/`` and the toolchain are the reason to run this before E10."""
    registry = {c.name: c for c in consumers(profile, repo_root=tmp_path, home=tmp_path, env={})}
    assert registry["rust/target/"].paths == (tmp_path / "rust" / "target",)
    assert registry["rustup + cargo"].paths == (tmp_path / ".rustup", tmp_path / ".cargo")
    assert registry["python/.venv"].paths == (tmp_path / "python" / ".venv",)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"UV_CACHE_DIR": "/explicit/uv"}, Path("/explicit/uv")),
        ({"XDG_CACHE_HOME": "/xdg"}, Path("/xdg/uv")),
        ({}, None),
    ],
)
def test_uv_cache_resolution(
    profile: StorageProfile, tmp_path: Path, env: dict[str, str], expected: Path | None
) -> None:
    home = tmp_path / "home"
    (home / ".cache" / "uv").mkdir(parents=True)
    registry = {c.name: c for c in consumers(profile, repo_root=tmp_path, home=home, env=env)}
    assert registry["uv cache"].paths == ((expected or home / ".cache" / "uv"),)


# --- the report ---------------------------------------------------------------


def test_absent_consumers_still_count_against_the_projection(
    profile: StorageProfile, tmp_path: Path
) -> None:
    """The point of running this early: budget for what is not installed yet."""
    measured = report(profile, repo_root=tmp_path / "repo", home=tmp_path / "home", env={})
    by_name = {u.consumer.name: u for u in measured.usages}
    assert by_name["rust/target/"].present is False
    assert by_name["rust/target/"].used == 0
    assert measured.budgeted == sum(u.consumer.budget for u in measured.usages)
    assert measured.budgeted > sum(u.used for u in measured.usages)


def test_present_consumers_contribute_their_measured_size(
    profile: StorageProfile, tmp_path: Path
) -> None:
    write(profile.zone(Zone.BRONZE, season=2023) / "nbastats_2023.parquet", 500_000)
    measured = report(profile, repo_root=tmp_path / "repo", home=tmp_path / "home", env={})
    data = next(u for u in measured.usages if u.consumer.name == "data zones")
    assert data.present
    assert data.used >= 500_000


def test_the_data_breakdown_has_a_line_per_zone(profile: StorageProfile, tmp_path: Path) -> None:
    """Adding a zone without wiring it into the report fails here."""
    measured = report(profile, repo_root=tmp_path / "repo", home=tmp_path / "home", env={})
    data = next(u for u in measured.usages if u.consumer.name == "data zones")
    # One part per zone, plus the DuckDB file.
    assert len(data.parts) == len(Zone) + 1
    assert {part.name for part in data.parts} >= {zone.value for zone in Zone}


def test_free_space_comes_from_a_real_volume(profile: StorageProfile, tmp_path: Path) -> None:
    """Read from the filesystem holding the path, never from ``df /``."""
    measured = report(profile, repo_root=tmp_path / "repo", home=tmp_path / "home", env={})
    assert measured.free > 0
    assert measured.volume.exists()


def test_a_missing_data_root_resolves_free_space_from_its_parent(tmp_path: Path) -> None:
    absent = StorageProfile(name=Profile.EPHEMERAL, root=tmp_path / "never" / "created")
    measured = report(absent, repo_root=tmp_path, home=tmp_path, env={})
    assert measured.free > 0


# --- the arithmetic -----------------------------------------------------------


def test_fits_when_the_budget_is_under_available_space() -> None:
    measured = FootprintReport(
        volume=Path("/"),
        free=10 * _GIB,
        usages=(usage("a", used=_GIB, budget=2 * _GIB), usage("b", used=0, budget=3 * _GIB)),
    )
    assert measured.budgeted == 5 * _GIB
    assert measured.available == 11 * _GIB
    assert measured.shortfall == 0
    assert measured.fits()


def test_does_not_fit_when_the_budget_exceeds_available_space() -> None:
    """The real machine's situation: 13 GiB of budget against ~11.5 GiB."""
    measured = FootprintReport(
        volume=Path("/"),
        free=11 * _GIB,
        usages=(usage("a", used=0, budget=8 * _GIB), usage("b", used=0, budget=5 * _GIB)),
    )
    assert measured.shortfall == 2 * _GIB
    assert not measured.fits()


def test_current_usage_counts_as_available_to_its_own_consumer() -> None:
    """A consumer already holding its bytes does not have to fit them twice."""
    measured = FootprintReport(
        volume=Path("/"),
        free=_GIB,
        usages=(usage("a", used=3 * _GIB, budget=4 * _GIB),),
    )
    assert measured.available == 4 * _GIB
    assert measured.fits()


def test_a_consumer_over_its_own_budget_does_not_fit_even_with_room() -> None:
    measured = FootprintReport(
        volume=Path("/"),
        free=100 * _GIB,
        usages=(usage("a", used=5 * _GIB, budget=_GIB),),
    )
    assert [u.consumer.name for u in measured.over] == ["a"]
    assert measured.shortfall == 0
    assert not measured.fits()


# --- reclaim ------------------------------------------------------------------


def test_reclaimable_sums_only_what_can_actually_be_freed() -> None:
    measured = FootprintReport(
        volume=Path("/"),
        free=_GIB,
        usages=(
            usage("target", used=3 * _GIB, budget=5 * _GIB, reclaim="cargo clean"),
            usage("derived", used=2 * _GIB, budget=4 * _GIB),
        ),
    )
    assert measured.reclaimable == 3 * _GIB


def test_a_consumer_reclaims_through_its_parts(profile: StorageProfile, tmp_path: Path) -> None:
    """The data root is not disposable, but ``raw/`` and the database are."""
    write(profile.zone(Zone.RAW) / "archive.zip", 300_000)
    write(profile.zone(Zone.BRONZE, season=2023) / "nbastats_2023.parquet", 300_000)
    measured = report(profile, repo_root=tmp_path / "repo", home=tmp_path / "home", env={})
    data = next(u for u in measured.usages if u.consumer.name == "data zones")

    reclaimable = {p.name for p in data.parts if p.reclaim is not None}
    assert Zone.RAW.value in reclaimable
    assert Zone.BRONZE.value not in reclaimable
    assert Zone.SILVER.value not in reclaimable
    assert Zone.GOLD.value not in reclaimable
    assert 300_000 <= data.reclaimable < data.used


# --- rendering ----------------------------------------------------------------


def test_the_report_renders_without_a_real_volume() -> None:
    measured = FootprintReport(
        volume=Path("/"),
        free=6 * _GIB,
        usages=(
            usage("data zones", used=20 * 1024 * 1024, budget=4 * _GIB),
            usage("rust/target/", used=0, budget=5 * _GIB, present=False, reclaim="cargo clean"),
        ),
    )
    rendered = measured.format()
    assert "data zones" in rendered
    assert "absent" in rendered
    assert "cargo clean" in rendered
    assert "SHORT BY" in rendered
    assert "reclaimable" in rendered


def test_a_consumer_over_budget_is_marked_in_the_output() -> None:
    measured = FootprintReport(
        volume=Path("/"),
        free=100 * _GIB,
        usages=(usage("uv cache", used=2 * _GIB, budget=_GIB, reclaim="uv cache prune"),),
    )
    assert "OVER BUDGET" in measured.format()


def test_parts_are_rendered_beneath_their_consumer() -> None:
    measured = FootprintReport(
        volume=Path("/"),
        free=100 * _GIB,
        usages=(
            Usage(
                consumer=Consumer(
                    name="data zones", paths=(Path("/data"),), budget=4 * _GIB, why=""
                ),
                used=2048,
                present=True,
                parts=(Part(name="bronze", used=2048), Part(name="gold", used=0)),
            ),
        ),
    )
    rendered = measured.format()
    assert "  bronze" in rendered
    # A zero-byte part is noise in the table; the "empty:" line carries it instead.
    assert "gold" not in rendered


# --- the gate -----------------------------------------------------------------


def gated(monkeypatch: pytest.MonkeyPatch, measured: FootprintReport) -> None:
    monkeypatch.setattr("hoopstate.footprint.report", lambda *a, **k: measured)


def test_the_cli_exits_nonzero_when_the_budget_does_not_fit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gated(
        monkeypatch,
        FootprintReport(volume=Path("/"), free=_GIB, usages=(usage("a", used=0, budget=9 * _GIB),)),
    )
    assert main([]) == 1
    assert "SHORT BY" in capsys.readouterr().out


def test_the_cli_exits_zero_when_it_fits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gated(
        monkeypatch,
        FootprintReport(
            volume=Path("/"), free=100 * _GIB, usages=(usage("a", used=0, budget=_GIB),)
        ),
    )
    assert main([]) == 0
    assert "fits" in capsys.readouterr().out


def test_no_gate_always_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """For a machine where the answer is known and the report is still wanted."""
    measured = FootprintReport(
        volume=Path("/"), free=_GIB, usages=(usage("a", used=0, budget=9 * _GIB),)
    )
    gated(monkeypatch, measured)
    assert not measured.fits()
    assert main(["--no-gate"]) == 0
    assert "SHORT BY" in capsys.readouterr().out


def test_the_cli_runs_end_to_end_against_a_real_root(
    profile: StorageProfile, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No monkeypatched report: the whole path, resolved from the environment."""
    monkeypatch.setenv(ENV_PROFILE, "local")
    monkeypatch.setenv(ENV_ROOT, str(profile.root))
    write(profile.zone(Zone.BRONZE, season=2023) / "nbastats_2023.parquet", 10_000)

    assert main(["--no-gate"]) == 0
    out = capsys.readouterr().out
    assert "local disk footprint" in out
    assert "rust/target/" in out
