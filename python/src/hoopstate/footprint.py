"""The local disk budget, and a check that measures against it (``hoops-1lg.1.3``).

This project's data is small: about 55 MB per season of raw plus bronze, so
thirty seasons with silver and gold on top lands at 3-4 GB. The pressure on a
development machine does not come from data at all — it comes from build
artifacts, and it arrives with the Rust port (E10), where a ``target/`` carrying
arrow, polars and duckdb routinely reaches 3-5 GB.

Summing the budget below gives more than the volume has free. That is not a bug
in the budget; it is the finding. ``docs/disk-budget.md`` says what to do about
it, and this module is what makes the claim measurable instead of remembered::

    python -m hoopstate.footprint            # measure, print, exit 1 if it does not fit
    python -m hoopstate.footprint --no-gate  # same report, always exit 0

Run it *before* E10 creates ``rust/crates/``, which is the whole point.

This is a local-machine tool. A cloud session gets a fresh 30 GB VM every time,
so it is deliberately not part of ``make check`` — the numbers it reports are
properties of a machine, not of the code.

Two measurement details, both verified rather than assumed, and both easy to get
wrong a second time:

*Sizes are counted in allocated blocks with inodes deduplicated, exactly as
``du`` does.* ``uv`` hardlinks out of its cache into the virtualenv — 233 such
entries in ``python/.venv`` on the machine this was written on — so summing
``st_size`` counts the same bytes twice and disagrees with every tool a reader
would check against. Block counting with an inode set reproduces ``du -sh`` to
the megabyte.

*Free space is read from the filesystem, not from ``df /``.* On macOS ``df /``
describes the sealed system snapshot, whose ``Used`` and ``Capacity`` columns
say nothing about the data volume; only ``Avail`` is shared between them. Taking
the number from :func:`shutil.disk_usage` on a real path avoids the question.

Note on the quarantine: the per-zone breakdown iterates
:class:`~hoopstate.storage.Zone` rather than listing members, so no zone is named
in this file and ``tests/test_oracle_quarantine.py``, which scans this module,
stays satisfied. Do not "simplify" that loop into an explicit list. Reporting how
many bytes a directory occupies is not reading what is in it, but naming the
quarantined zone in source is what the guard forbids, and correctly so.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path

from hoopstate._format import human_bytes, table
from hoopstate.storage import StorageProfile, Zone, resolve_profile

__all__ = [
    "Consumer",
    "FootprintReport",
    "Part",
    "Usage",
    "consumers",
    "disk_usage_of",
    "report",
]

_GIB = 1024**3

# Where the budget numbers come from: one season measured end to end (25.6 MB of
# raw archives plus 29.3 MB of bronze parquet), and published sizes for the
# toolchains. They are round on purpose — a disk budget is argued in halves of a
# gigabyte, and pretending otherwise would imply a precision the inputs lack.
_DATA_BUDGET = 4 * _GIB
_TARGET_BUDGET = 5 * _GIB
_TOOLCHAIN_BUDGET = 3 * _GIB // 2
_VENV_BUDGET = 3 * _GIB // 2
_UV_CACHE_BUDGET = _GIB

# How a zone's bytes come back, for the zones where they come back at all.
# Everything not listed here is derived work that would have to be recomputed,
# which is a cost rather than a reclaim, so it is reported as not reclaimable.
_ZONE_RECLAIM: dict[Zone, str] = {
    Zone.RAW: "re-fetchable from the source manifest",
    Zone.CACHE: "a keyed download cache; delete freely",
}


@dataclass(frozen=True)
class Consumer:
    """One thing that occupies disk, and what it is allowed to occupy.

    ``paths`` is a tuple because a single budget can cover more than one
    directory — the Rust toolchain is ``~/.rustup`` and ``~/.cargo``, and
    splitting 1.5 GiB between them would be inventing a number.
    """

    name: str
    paths: tuple[Path, ...]
    budget: int
    why: str
    reclaim: str | None = None


@dataclass(frozen=True)
class Part:
    """A line beneath a consumer: a zone, or the database file."""

    name: str
    used: int
    reclaim: str | None = None


@dataclass(frozen=True)
class Usage:
    """What a consumer actually occupies right now."""

    consumer: Consumer
    used: int
    present: bool
    parts: tuple[Part, ...] = ()

    @property
    def over(self) -> bool:
        """Is this consumer already past its budget?"""
        return self.used > self.consumer.budget

    @property
    def reclaimable(self) -> int:
        """Bytes recoverable here without recomputing anything.

        A consumer with no reclaim action of its own can still have reclaimable
        parts — the data root is not disposable, but the download cache and the
        DuckDB file inside it are.
        """
        if self.consumer.reclaim is not None:
            return self.used
        return sum(part.used for part in self.parts if part.reclaim is not None)


@dataclass(frozen=True)
class FootprintReport:
    """Every consumer measured, against the budget and against free space."""

    volume: Path
    free: int
    usages: tuple[Usage, ...]
    spans_volumes: bool = False
    empty_parts: tuple[str, ...] = field(default=())

    @property
    def used(self) -> int:
        return sum(usage.used for usage in self.usages)

    @property
    def budgeted(self) -> int:
        """What the budget asks for in total, once everything has grown into it."""
        return sum(usage.consumer.budget for usage in self.usages)

    @property
    def available(self) -> int:
        """What these consumers may collectively occupy: free space plus their own bytes.

        Their current usage counts as available to them because it is already
        theirs — the question the budget asks is whether the *end state* fits,
        not whether it fits a second time alongside today's.
        """
        return self.free + self.used

    @property
    def shortfall(self) -> int:
        return max(0, self.budgeted - self.available)

    @property
    def over(self) -> tuple[Usage, ...]:
        return tuple(usage for usage in self.usages if usage.over)

    @property
    def reclaimable(self) -> int:
        return sum(usage.reclaimable for usage in self.usages)

    def fits(self) -> bool:
        """Does the full budget fit, and is nothing already past its own line?"""
        return self.shortfall == 0 and not self.over

    def format(self) -> str:
        rows: list[tuple[str, str, str]] = []
        for usage in self.usages:
            detail = "" if usage.present else "absent"
            if usage.over:
                detail = "OVER BUDGET" if not detail else f"{detail}, OVER BUDGET"
            if usage.consumer.reclaim is not None:
                detail = f"{detail}; {usage.consumer.reclaim}" if detail else usage.consumer.reclaim
            value = f"{human_bytes(usage.used)} / {human_bytes(usage.consumer.budget)}"
            rows.append((usage.consumer.name, value, detail))
            for part in usage.parts:
                if part.used:
                    rows.append((f"  {part.name}", human_bytes(part.used), part.reclaim or ""))

        lines = [
            f"local disk footprint — {human_bytes(self.free)} free on {self.volume}",
            "",
            *table(rows, headers=("consumer", "used / budget", "notes")),
            "",
            f"  budgeted {human_bytes(self.budgeted)} against {human_bytes(self.available)} "
            f"available ({human_bytes(self.free)} free + {human_bytes(self.used)} already used)",
        ]
        if self.shortfall:
            lines.append(f"  SHORT BY {human_bytes(self.shortfall)} — see docs/disk-budget.md")
        else:
            lines.append(f"  fits, with {human_bytes(self.available - self.budgeted)} to spare")
        lines.append(f"  reclaimable right now: {human_bytes(self.reclaimable)}")
        if self.empty_parts:
            lines.append(f"  empty: {', '.join(self.empty_parts)}")
        if self.spans_volumes:
            lines.append("  note: these consumers are not all on one volume; free space is shared")
        return "\n".join(lines)


def disk_usage_of(path: Path) -> tuple[int, bool]:
    """``(bytes, present)`` for one path, with ``du``'s semantics.

    Allocated blocks rather than apparent size, and a hardlinked inode counted
    once — see the module docstring for why both matter. An unreadable subtree is
    skipped rather than raised: a report that dies on one permission error is
    less useful than one that is slightly low and says so by omission.
    """
    try:
        root_stat = path.lstat()
    except OSError:
        return 0, False

    seen: set[tuple[int, int]] = set()
    total = 0

    def add(entry: os.stat_result) -> None:
        nonlocal total
        if entry.st_nlink > 1:
            key = (entry.st_dev, entry.st_ino)
            if key in seen:
                return
            seen.add(key)
        total += entry.st_blocks * 512

    add(root_stat)
    if not stat.S_ISDIR(root_stat.st_mode):
        return total, True

    for dirpath, dirnames, filenames in os.walk(path, onerror=lambda _error: None):
        for name in (*dirnames, *filenames):
            try:
                add(os.lstat(os.path.join(dirpath, name)))
            except OSError:
                continue
    return total, True


def _repo_root() -> Path:
    """The checkout this module lives in: ``<repo>/python/src/hoopstate/footprint.py``."""
    return Path(__file__).resolve().parents[3]


def _uv_cache_dir(home: Path, env: dict[str, str]) -> Path:
    """Where uv keeps its cache, without shelling out to ``uv cache dir``.

    ``UV_CACHE_DIR`` wins, then XDG, then whichever of the two platform
    defaults exists. uv follows XDG on macOS too, so ``~/.cache/uv`` is checked
    before ``~/Library/Caches/uv`` rather than after.
    """
    override = env.get("UV_CACHE_DIR")
    if override:
        return Path(override)
    xdg = env.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "uv"
    candidates = (home / ".cache" / "uv", home / "Library" / "Caches" / "uv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def consumers(
    profile: StorageProfile | None = None,
    *,
    repo_root: Path | None = None,
    home: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[Consumer, ...]:
    """The budget, as an explicit registry.

    Every path is injectable so the check can be tested without touching the
    real ``$HOME`` — the same reason :func:`hoopstate.storage.resolve_profile`
    takes an ``env``.
    """
    if profile is None:
        profile = resolve_profile()
    if repo_root is None:
        repo_root = _repo_root()
    if home is None:
        home = Path.home()
    if env is None:
        env = dict(os.environ)

    return (
        Consumer(
            name="data zones",
            paths=(profile.root,),
            budget=_DATA_BUDGET,
            why="~55 MB/season measured, x30 seasons, plus silver and gold",
        ),
        Consumer(
            name="rust/target/",
            paths=(repo_root / "rust" / "target",),
            budget=_TARGET_BUDGET,
            why="arrow/polars/duckdb build artifacts reach 3-5 GB",
            reclaim="cargo clean",
        ),
        Consumer(
            name="rustup + cargo",
            paths=(home / ".rustup", home / ".cargo"),
            budget=_TOOLCHAIN_BUDGET,
            why="one toolchain plus the registry and its source checkouts",
        ),
        Consumer(
            name="python/.venv",
            paths=(repo_root / "python" / ".venv",),
            budget=_VENV_BUDGET,
            why="polars, duckdb and pyarrow wheels unpacked",
            reclaim="delete it; uv sync rebuilds it",
        ),
        Consumer(
            name="uv cache",
            paths=(_uv_cache_dir(home, env),),
            budget=_UV_CACHE_BUDGET,
            why="wheel and source cache, shared across checkouts",
            reclaim="uv cache prune",
        ),
    )


def _data_parts(profile: StorageProfile) -> tuple[tuple[Part, ...], tuple[str, ...]]:
    """One part per zone, plus the database file. Returns ``(parts, empty names)``.

    Zones come from iterating the enum, never from a list written out here — see
    the note on the quarantine in the module docstring.
    """
    parts: list[Part] = []
    empty: list[str] = []
    for zone in Zone:
        used, present = disk_usage_of(profile.zone(zone))
        parts.append(Part(name=zone.value, used=used, reclaim=_ZONE_RECLAIM.get(zone)))
        if not present or not used:
            empty.append(zone.value)

    database = profile.duckdb_path()
    used, present = disk_usage_of(database)
    parts.append(
        Part(
            name=database.name,
            used=used,
            reclaim="rebuildable: python -m hoopstate.db.catalog",
        )
    )
    if not present:
        empty.append(database.name)
    return tuple(parts), tuple(empty)


def _free_bytes(path: Path) -> tuple[Path, int]:
    """Free space on the volume holding ``path``, or its nearest existing parent."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe, shutil.disk_usage(probe).free


def report(
    profile: StorageProfile | None = None,
    *,
    repo_root: Path | None = None,
    home: Path | None = None,
    env: dict[str, str] | None = None,
) -> FootprintReport:
    """Measure every consumer and compare it to the budget."""
    if profile is None:
        profile = resolve_profile()

    registry = consumers(profile, repo_root=repo_root, home=home, env=env)
    usages: list[Usage] = []
    devices: set[int] = set()
    empty_parts: tuple[str, ...] = ()

    for consumer in registry:
        used = 0
        present = False
        for path in consumer.paths:
            measured, exists = disk_usage_of(path)
            used += measured
            present = present or exists
            if exists:
                devices.add(path.stat().st_dev)

        parts: tuple[Part, ...] = ()
        if consumer.paths == (profile.root,):
            parts, empty_parts = _data_parts(profile)

        usages.append(Usage(consumer=consumer, used=used, present=present, parts=parts))

    volume, free = _free_bytes(profile.root)
    return FootprintReport(
        volume=volume,
        free=free,
        usages=tuple(usages),
        spans_volumes=len(devices) > 1,
        empty_parts=empty_parts,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hoopstate.footprint [--no-gate]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure local disk usage against the documented budget."
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Print the report but always exit 0, instead of failing when it does not fit.",
    )
    args = parser.parse_args(argv)

    measured = report()
    print(measured.format())
    if args.no_gate or measured.fits():
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
