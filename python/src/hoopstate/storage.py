"""Portable storage-profile resolution.

Every data path in the project resolves through a storage *profile* rather than
assuming one machine. This is the only module that knows which profile is
active or where a zone physically lands; ingestion, the model, and the tests all
ask for a zone path and get one back. That containment is deliberate (see
``hoops-1lg.1.2``): keep the knowledge of physical layout here and nowhere else,
so a moved directory never ripples downstream.

Two profiles exist today, and they differ only in *where the root is*:

``ephemeral`` (the default; cloud sessions and CI)
    One scratch directory under the system temp dir. Nothing is assumed to
    persist between sessions, so raw archives are re-fetched on demand and the
    root is a stable name rather than a fresh ``mkdtemp`` — that is what makes a
    repeat run *within* one session cheap.

``local`` (the development machine)
    The same layout under ``~/hoopstate/data``, which persists between sessions.

There is deliberately no hot/cold tiering (see ``hoops-03c``). It existed to
keep read-mostly bulk off a nearly-full boot disk, but the numbers do not
justify it: one season of raw plus bronze is about 55 MB, so thirty seasons of
everything — silver and gold included — is roughly 3-4 GB, against 12 GiB free.
Raw archives are a download cache keyed to a manifest on GitHub and are
re-fetchable, so they need no durable home at all.

``HOOPSTATE_ROOT`` overrides the root for whichever profile is selected.

**``HOOPSTATE_ROOT`` must be on local disk.** The DuckDB database lives under
it, and DuckDB does not support database files on network filesystems; SMB
locking in particular is unreliable enough to risk corruption. This is a
precondition on the caller, not something this module enforces — checking the
filesystem type would mean reintroducing exactly the mount machinery that
``hoops-03c`` removed. The database is always rebuildable from parquet, so the
cost of getting this wrong is a rebuild, not data loss.
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
ENV_ROOT = "HOOPSTATE_ROOT"


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
    """A resolved storage layout: one root plus the zones beneath it.

    Instances are immutable and cheap to pass around. Construct one with
    :func:`resolve_profile`; callers then use :meth:`zone` and
    :meth:`duckdb_path` and never inspect :attr:`name`.
    """

    name: Profile
    root: Path

    def zone(self, zone: Zone, *, season: int | None = None) -> Path:
        """Resolve the directory for ``zone``.

        ``season`` partitions :attr:`Zone.BRONZE` so seasons do not collide; it
        is ignored for every other zone. Callers pass a zone and receive a path,
        and never learn how the layout is arranged — that is the whole point of
        routing through here.
        """
        if zone is Zone.BRONZE and season is not None:
            return self.root / zone.value / f"season={season}"
        return self.root / zone.value

    def duckdb_path(self, name: str = "hoopstate.duckdb") -> Path:
        """Path to the DuckDB database file, under the profile root.

        The root is required to be on local disk — see the module docstring.
        DuckDB on a network filesystem risks corruption, and the database is
        always rebuildable from parquet, so keeping it local costs nothing.
        """
        return self.root / name


def _default_root(profile: Profile) -> Path:
    """The root for a profile before the env override is applied."""
    if profile is Profile.EPHEMERAL:
        return _ephemeral_scratch_root()
    # Profile.LOCAL: a persistent working set on local disk.
    return Path.home() / "hoopstate" / "data"


def resolve_profile(env: dict[str, str] | None = None) -> StorageProfile:
    """Resolve the active storage profile from the environment.

    ``HOOPSTATE_PROFILE`` selects the profile (default ``ephemeral``) and
    ``HOOPSTATE_ROOT`` overrides the root for whichever profile is selected.
    ``env`` defaults to ``os.environ`` and is injectable for tests.

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

    root = Path(env[ENV_ROOT]) if env.get(ENV_ROOT) else _default_root(profile)
    return StorageProfile(name=profile, root=root)
