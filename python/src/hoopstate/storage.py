"""Portable storage-profile resolution.

Every data path in the project resolves through a storage *profile* rather than
assuming one machine. This is the only module that knows which profile is
active or how a zone maps onto physical storage; ingestion, the model, and the
tests all ask for a zone path and get one back. That containment is deliberate
(see ``hoops-1lg.1.2``): keep the ``if profile == ...`` branching here and
nowhere else, so a new profile or a moved mount never ripples downstream.

Two profiles exist today:

``ephemeral`` (the default; cloud sessions and CI)
    Every zone lives under one scratch directory on local disk. Nothing is
    assumed to persist between sessions, so raw archives are re-fetched on
    demand and the download cache is keyed to a stable path so a repeat run
    *within* one session is cheap. A cloud VM has ample disk for a full season
    end to end, so there is no hot/cold split and no preflight.

``local`` (the development machine)
    A hot/cold split. Local disk is nearly full while the NAS has terabytes to
    spare, so read-mostly cold data — raw archives and non-active-season bronze
    — lives on the NAS, and everything the working set touches constantly —
    silver, gold, oracle, the active season's bronze, and the DuckDB file —
    stays on fast local disk.

Selection is by ``HOOPSTATE_PROFILE``; individual tiers are overridable by
``HOOPSTATE_HOT`` and ``HOOPSTATE_COLD``. The DuckDB file MUST stay on local
disk under every profile: DuckDB does not support database files on network
filesystems, and SMB locking is unreliable enough to risk corruption. That
invariant is enforced by :meth:`StorageProfile.duckdb_path` deriving only from
the hot (local) tier, and asserted by the test suite for both profiles.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "Profile",
    "StorageProfile",
    "Zone",
    "resolve_profile",
]

# Environment variables. Kept together so the full surface a caller can set is
# visible in one place.
ENV_PROFILE = "HOOPSTATE_PROFILE"
ENV_HOT = "HOOPSTATE_HOT"
ENV_COLD = "HOOPSTATE_COLD"

# Default cold root on the development machine: the NAS mount. Only ever used by
# the ``local`` profile, and overridable by HOOPSTATE_COLD. The ephemeral
# profile never touches it.
_DEFAULT_NAS_ROOT = Path("/Volumes/home/hoopstate")


class Profile(StrEnum):
    """The available storage profiles.

    A ``StrEnum`` so the value round-trips through the environment and compares
    equal to the plain string, e.g. ``Profile.EPHEMERAL == "ephemeral"``.
    """

    EPHEMERAL = "ephemeral"
    LOCAL = "local"


class Zone(StrEnum):
    """A logical data zone, independent of where it physically lands.

    Ordering mirrors the pipeline's flow: raw archives are downloaded, cached,
    converted 1:1 into typed ``bronze``, refined into canonical ``silver``,
    published as versioned ``gold`` marts, and validated against the quarantined
    ``oracle`` reference.
    """

    RAW = "raw"
    """Downloaded source archives (tar.xz), exactly as fetched."""

    CACHE = "cache"
    """Keyed download cache, so a re-fetch within a session is cheap."""

    BRONZE = "bronze"
    """Typed tables, 1:1 with source."""

    SILVER = "silver"
    """Canonical model: canonical_event, lineup_stint, possession, chance."""

    GOLD = "gold"
    """Versioned analysis marts — the future API contract."""

    ORACLE = "oracle"
    """pbpstats reference data (quarantined; read only by the test suite)."""


# Which physical tier each zone falls on under the ``local`` profile. Under
# ``ephemeral`` both tiers are the same directory, so this table is inert there.
# Cold = read-mostly bulk that can live on the NAS; hot = the constantly
# touched working set that must stay on fast local disk. Bronze is the one zone
# that straddles the split — the active season is hot, everything else cold —
# so it is resolved separately in :meth:`StorageProfile.zone` and left out here.
_COLD_ZONES: frozenset[Zone] = frozenset({Zone.RAW, Zone.CACHE})
_HOT_ZONES: frozenset[Zone] = frozenset({Zone.SILVER, Zone.GOLD, Zone.ORACLE})


def _ephemeral_scratch_root() -> Path:
    """Stable per-session scratch directory for the ephemeral profile.

    A fixed name under the system temp dir (rather than a fresh
    ``mkdtemp``) is what makes the download cache *keyed*: a second run in the
    same session finds the archives the first run left behind, instead of
    re-downloading them.
    """
    return Path(tempfile.gettempdir()) / "hoopstate-ephemeral"


@dataclass(frozen=True)
class StorageProfile:
    """A resolved storage layout: two physical roots plus the mapping onto them.

    Instances are immutable and cheap to pass around. Construct one with
    :func:`resolve_profile`; callers then use :meth:`zone` and
    :meth:`duckdb_path` and never inspect :attr:`name`.
    """

    name: Profile
    hot_root: Path
    cold_root: Path

    def zone(
        self,
        zone: Zone,
        *,
        season: int | None = None,
        active_season: int | None = None,
    ) -> Path:
        """Resolve the directory for ``zone``.

        ``season`` and ``active_season`` only matter for :attr:`Zone.BRONZE`
        under the ``local`` profile, where the active season is kept hot and
        every other season is pushed cold. Passing them for any other zone is
        harmless and ignored. Callers pass the season and the active season and
        receive a path; they never learn which tier it resolved to — that is the
        whole point of routing through here.
        """
        root = self._root_for(zone, season=season, active_season=active_season)
        if zone is Zone.BRONZE and season is not None:
            return root / zone.value / f"season={season}"
        return root / zone.value

    def duckdb_path(self, name: str = "hoopstate.duckdb") -> Path:
        """Path to the DuckDB database file — always on the hot (local) tier.

        Never resolves through the cold tier under any profile: DuckDB on a
        network filesystem risks corruption. The database is always rebuildable
        from parquet, so keeping it local costs nothing.
        """
        return self.hot_root / name

    def _root_for(
        self,
        zone: Zone,
        *,
        season: int | None,
        active_season: int | None,
    ) -> Path:
        """Pick the hot or cold root for ``zone``. The only profile branch."""
        # Ephemeral collapses the split: both roots are the same scratch dir,
        # so every zone lands in the same place regardless of the table below.
        if self.hot_root == self.cold_root:
            return self.hot_root

        if zone is Zone.BRONZE:
            # Active season hot, everything else (including "season unknown")
            # cold. Treating an unspecified season as cold is the safe default:
            # bulk backfill is cold, and the active season is always named
            # explicitly.
            if season is not None and season == active_season:
                return self.hot_root
            return self.cold_root

        if zone in _COLD_ZONES:
            return self.cold_root
        # Everything not explicitly cold is hot. Listing hot zones as well keeps
        # the intent auditable and turns a newly added, unclassified zone into a
        # loud KeyError-adjacent failure below rather than a silent NAS write.
        if zone in _HOT_ZONES:
            return self.hot_root
        raise ValueError(f"zone {zone!r} is not assigned to a storage tier")


def _default_roots(profile: Profile) -> tuple[Path, Path]:
    """The (hot, cold) roots for a profile before env overrides are applied."""
    if profile is Profile.EPHEMERAL:
        scratch = _ephemeral_scratch_root()
        # No split: hot and cold are the same local scratch directory. This is
        # what makes the DuckDB-is-local invariant hold trivially here.
        return scratch, scratch
    # Profile.LOCAL: hot on local disk (repo-adjacent ``data/`` working set),
    # cold on the NAS mount.
    hot = Path.home() / "hoopstate" / "data"
    return hot, _DEFAULT_NAS_ROOT


def resolve_profile(env: dict[str, str] | None = None) -> StorageProfile:
    """Resolve the active storage profile from the environment.

    ``HOOPSTATE_PROFILE`` selects the profile (default ``ephemeral``);
    ``HOOPSTATE_HOT`` and ``HOOPSTATE_COLD`` override the respective root for
    whichever profile is selected. ``env`` defaults to ``os.environ`` and is
    injectable for tests.

    A fresh checkout with no variables set resolves to the ephemeral profile
    with every zone under one local scratch directory — zero configuration,
    which is the standing CI proof that the portable path works.
    """
    if env is None:
        env = dict(os.environ)

    raw_name = env.get(ENV_PROFILE, Profile.EPHEMERAL.value).strip().lower()
    try:
        profile = Profile(raw_name)
    except ValueError:
        valid = ", ".join(p.value for p in Profile)
        raise ValueError(
            f"{ENV_PROFILE}={raw_name!r} is not a known profile; expected one of: {valid}"
        ) from None

    hot_default, cold_default = _default_roots(profile)
    hot_root = Path(env[ENV_HOT]) if env.get(ENV_HOT) else hot_default
    cold_root = Path(env[ENV_COLD]) if env.get(ENV_COLD) else cold_default

    return StorageProfile(name=profile, hot_root=hot_root, cold_root=cold_root)
