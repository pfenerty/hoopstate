"""Tests for portable storage-profile resolution (``hoops-1lg.1.2``).

These assert the acceptance criteria directly:

* ``HOOPSTATE_PROFILE`` selects ephemeral vs local; ephemeral is the default.
* ``HOOPSTATE_HOT`` / ``HOOPSTATE_COLD`` override individual tiers.
* A fresh checkout with no environment variables resolves every zone.
* The DuckDB path is on the local (hot) tier under every profile.
* Only this module branches on which profile is active — enforced indirectly by
  callers never needing ``profile.name`` to get a path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hoopstate.storage import (
    ENV_COLD,
    ENV_HOT,
    ENV_PROFILE,
    Profile,
    StorageProfile,
    Zone,
    resolve_profile,
)


def test_default_profile_is_ephemeral() -> None:
    """No environment variables set -> ephemeral, the zero-config portable path."""
    profile = resolve_profile(env={})
    assert profile.name is Profile.EPHEMERAL


def test_profile_selection() -> None:
    assert resolve_profile(env={ENV_PROFILE: "ephemeral"}).name is Profile.EPHEMERAL
    assert resolve_profile(env={ENV_PROFILE: "local"}).name is Profile.LOCAL


def test_profile_selection_is_case_insensitive_and_trimmed() -> None:
    assert resolve_profile(env={ENV_PROFILE: "  LOCAL "}).name is Profile.LOCAL


def test_unknown_profile_is_rejected_with_a_helpful_message() -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve_profile(env={ENV_PROFILE: "nas"})
    msg = str(excinfo.value)
    assert "nas" in msg
    assert "ephemeral" in msg and "local" in msg


def test_every_zone_resolves_under_ephemeral() -> None:
    """A fresh checkout must be able to resolve *every* zone with no config."""
    profile = resolve_profile(env={})
    for zone in Zone:
        path = profile.zone(zone)
        assert isinstance(path, Path)
        # Ephemeral collapses the split: everything under one scratch root.
        assert profile.hot_root == profile.cold_root
        assert path.is_relative_to(profile.hot_root)


def test_every_zone_resolves_under_local() -> None:
    profile = resolve_profile(env={ENV_PROFILE: "local"})
    for zone in Zone:
        assert isinstance(profile.zone(zone), Path)


def test_hot_and_cold_overrides(tmp_path: Path) -> None:
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    profile = resolve_profile(env={ENV_PROFILE: "local", ENV_HOT: str(hot), ENV_COLD: str(cold)})
    assert profile.hot_root == hot
    assert profile.cold_root == cold
    # Silver is a hot zone, raw is a cold zone.
    assert profile.zone(Zone.SILVER).is_relative_to(hot)
    assert profile.zone(Zone.RAW).is_relative_to(cold)


def test_overrides_apply_to_ephemeral_too(tmp_path: Path) -> None:
    hot = tmp_path / "scratch"
    profile = resolve_profile(env={ENV_PROFILE: "ephemeral", ENV_HOT: str(hot)})
    assert profile.hot_root == hot
    assert profile.zone(Zone.GOLD).is_relative_to(hot)


def test_local_tier_assignment(tmp_path: Path) -> None:
    """Read-mostly bulk goes cold; the working set stays hot."""
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    profile = resolve_profile(env={ENV_PROFILE: "local", ENV_HOT: str(hot), ENV_COLD: str(cold)})
    for zone in (Zone.SILVER, Zone.GOLD, Zone.ORACLE):
        assert profile.zone(zone).is_relative_to(hot), f"{zone} should be hot"
    for zone in (Zone.RAW, Zone.CACHE):
        assert profile.zone(zone).is_relative_to(cold), f"{zone} should be cold"


def test_bronze_active_season_is_hot_others_cold(tmp_path: Path) -> None:
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    profile = resolve_profile(env={ENV_PROFILE: "local", ENV_HOT: str(hot), ENV_COLD: str(cold)})
    active = profile.zone(Zone.BRONZE, season=2025, active_season=2025)
    stale = profile.zone(Zone.BRONZE, season=2019, active_season=2025)
    assert active.is_relative_to(hot)
    assert stale.is_relative_to(cold)
    # Season is reflected in the path so partitions don't collide.
    assert active.name == "season=2025"
    assert stale.name == "season=2019"


def test_bronze_without_season_is_cold_under_local(tmp_path: Path) -> None:
    """Unknown season -> cold: bulk backfill is the safe default, not the NAS-free hot disk."""
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    profile = resolve_profile(env={ENV_PROFILE: "local", ENV_HOT: str(hot), ENV_COLD: str(cold)})
    assert profile.zone(Zone.BRONZE).is_relative_to(cold)


@pytest.mark.parametrize("profile_name", [p.value for p in Profile])
def test_duckdb_is_always_on_local_disk(profile_name: str, tmp_path: Path) -> None:
    """The DuckDB file must never resolve onto the cold (NAS) tier."""
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    profile = resolve_profile(
        env={ENV_PROFILE: profile_name, ENV_HOT: str(hot), ENV_COLD: str(cold)}
    )
    duckdb = profile.duckdb_path()
    assert duckdb.is_relative_to(hot), "DuckDB must live on the hot/local tier"
    # For a genuine split (local profile), assert it is not on cold at all.
    if profile.hot_root != profile.cold_root:
        assert not duckdb.is_relative_to(cold)


def test_duckdb_default_layout_is_local_without_overrides() -> None:
    """Even with real profile defaults (no overrides), DuckDB stays off the NAS."""
    local = resolve_profile(env={ENV_PROFILE: "local"})
    duckdb = local.duckdb_path()
    assert duckdb.is_relative_to(local.hot_root)
    assert not duckdb.is_relative_to(local.cold_root)


def test_profile_is_immutable() -> None:
    profile = resolve_profile(env={})
    with pytest.raises((AttributeError, TypeError)):
        profile.name = Profile.LOCAL  # type: ignore[misc]


def test_storage_profile_can_be_constructed_directly(tmp_path: Path) -> None:
    """The dataclass is a plain value; construction shouldn't need the resolver."""
    profile = StorageProfile(name=Profile.LOCAL, hot_root=tmp_path / "h", cold_root=tmp_path / "c")
    assert profile.zone(Zone.GOLD).is_relative_to(tmp_path / "h")
