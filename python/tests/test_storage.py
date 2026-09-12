"""Tests for portable storage-profile resolution (``hoops-1lg.1.2``, ``hoops-03c``).

These assert the acceptance criteria directly:

* ``HOOPSTATE_PROFILE`` selects ephemeral vs local; ephemeral is the default.
* ``HOOPSTATE_ROOT`` overrides the root for whichever profile is selected.
* A fresh checkout with no environment variables resolves every zone.
* Bronze partitions by season; no other zone takes a season.
* The DuckDB path sits under the profile root.
* Only this module knows the physical layout — enforced indirectly by callers
  never needing ``profile.name`` to get a path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hoopstate.storage import (
    ENV_PROFILE,
    ENV_ROOT,
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


@pytest.mark.parametrize("profile_name", [p.value for p in Profile])
def test_every_zone_resolves_under_one_root(profile_name: str) -> None:
    """A fresh checkout must be able to resolve *every* zone with no config."""
    profile = resolve_profile(env={ENV_PROFILE: profile_name})
    for zone in Zone:
        path = profile.zone(zone)
        assert isinstance(path, Path)
        assert path.is_relative_to(profile.root)


def test_root_override(tmp_path: Path) -> None:
    profile = resolve_profile(env={ENV_PROFILE: "local", ENV_ROOT: str(tmp_path)})
    assert profile.root == tmp_path
    for zone in Zone:
        assert profile.zone(zone).is_relative_to(tmp_path)


def test_root_override_applies_to_ephemeral_too(tmp_path: Path) -> None:
    profile = resolve_profile(env={ENV_PROFILE: "ephemeral", ENV_ROOT: str(tmp_path)})
    assert profile.root == tmp_path
    assert profile.zone(Zone.GOLD).is_relative_to(tmp_path)


def test_the_two_profiles_have_different_default_roots() -> None:
    """Ephemeral is throwaway scratch; local persists. That is the whole distinction."""
    ephemeral = resolve_profile(env={ENV_PROFILE: "ephemeral"})
    local = resolve_profile(env={ENV_PROFILE: "local"})
    assert ephemeral.root != local.root
    assert local.root.is_relative_to(Path.home())


def test_bronze_partitions_by_season(tmp_path: Path) -> None:
    """The season partition survives the collapse of the hot/cold split."""
    profile = resolve_profile(env={ENV_ROOT: str(tmp_path)})
    assert profile.zone(Zone.BRONZE, season=2023).name == "season=2023"
    assert profile.zone(Zone.BRONZE, season=2019).name == "season=2019"
    # Two seasons must not collide.
    assert profile.zone(Zone.BRONZE, season=2023) != profile.zone(Zone.BRONZE, season=2019)


def test_bronze_without_a_season_has_no_season_component(tmp_path: Path) -> None:
    profile = resolve_profile(env={ENV_ROOT: str(tmp_path)})
    assert profile.zone(Zone.BRONZE) == tmp_path / "bronze"


def test_season_is_ignored_for_other_zones(tmp_path: Path) -> None:
    profile = resolve_profile(env={ENV_ROOT: str(tmp_path)})
    assert profile.zone(Zone.SILVER, season=2023) == tmp_path / "silver"


@pytest.mark.parametrize("profile_name", [p.value for p in Profile])
def test_duckdb_sits_under_the_profile_root(profile_name: str, tmp_path: Path) -> None:
    profile = resolve_profile(env={ENV_PROFILE: profile_name, ENV_ROOT: str(tmp_path)})
    assert profile.duckdb_path().is_relative_to(tmp_path)


def test_duckdb_default_layout_needs_no_overrides() -> None:
    local = resolve_profile(env={ENV_PROFILE: "local"})
    assert local.duckdb_path().is_relative_to(local.root)


def test_profile_is_immutable() -> None:
    profile = resolve_profile(env={})
    with pytest.raises((AttributeError, TypeError)):
        profile.name = Profile.LOCAL  # type: ignore[misc]


def test_storage_profile_can_be_constructed_directly(tmp_path: Path) -> None:
    """The dataclass is a plain value; construction shouldn't need the resolver."""
    profile = StorageProfile(name=Profile.LOCAL, root=tmp_path)
    assert profile.zone(Zone.GOLD).is_relative_to(tmp_path)
