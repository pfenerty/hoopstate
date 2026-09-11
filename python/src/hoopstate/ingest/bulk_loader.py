"""Manifest-driven bulk loader (``hoops-1lg.2.1``).

Ingests from the `shufinskiy/nba_data
<https://github.com/shufinskiy/nba_data>`_ archive, which publishes
pre-scraped, ``tar.xz``-wrapped CSVs. A single manifest lists every dataset as
``name=url`` pairs; each archive contains exactly one CSV.

The pipeline this module implements, per dataset:

1. **Download** the ``tar.xz`` into the ``raw`` zone (cold under the ``local``
   profile), streamed to a temporary ``.part`` file and atomically renamed, so
   an interrupted transfer never leaves a truncated archive behind.
2. **Checksum and cache.** A SHA-256 of the archive is written as a sidecar
   next to it. A completed dataset is therefore never re-fetched: a second run
   finds the cached archive (and its already-written parquet) and no-ops.
3. **Stream to parquet** without ever spilling the extracted CSV to disk. The
   CSV inside the archive can be ~300 MB, a transient spike the development
   machine's boot disk cannot afford, so we decompress in memory and write
   parquet straight out. Columns are read losslessly as strings; per-source
   typing is the job of the bronze-conversion issues that depend on this one.

Every filesystem path is resolved through :mod:`hoopstate.storage`, so this
module never learns which storage profile is active.

Network access is injected as a :class:`ByteSource` (see
:func:`requests_byte_source`), which keeps the core logic — manifest parsing,
name parsing, caching, atomic writes, decompression — fully exercisable offline
by the test suite.

**URL rewriting.** The manifest points at
``github.com/<owner>/<repo>/raw/<ref>/<path>`` URLs. Those are rewritten to the
equivalent ``raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>`` form: it is
the host the project's network allowlist covers, and it serves the bytes
directly instead of routing through the authenticated GitHub API surface.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl

from hoopstate.storage import StorageProfile, Zone, resolve_profile

__all__ = [
    "MANIFEST_URL",
    "ByteSource",
    "LoadResult",
    "ParsedName",
    "bronze_parquet_path",
    "load_dataset",
    "parse_dataset_name",
    "parse_manifest",
    "read_manifest",
    "requests_byte_source",
    "stream_archive_to_parquet",
]

# The canonical manifest. Served from raw.githubusercontent.com, which the
# project's network allowlist covers.
MANIFEST_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"

# Streaming chunk size for downloads. Large enough to keep syscall overhead
# down, small enough that the compressed archive never sits fully in memory.
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB

# github.com/<owner>/<repo>/raw/<ref>/<path> -> raw.githubusercontent.com form.
_GITHUB_RAW_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/raw/(?P<rest>.+)$"
)


class ByteSource(Protocol):
    """Opens a URL and yields its body as a stream of byte chunks.

    Injected so the loader's logic can be driven from local fixtures in tests
    without touching the network. The returned context manager owns the
    connection and closes it on exit.
    """

    def __call__(self, url: str) -> AbstractContextManager[Iterator[bytes]]: ...


@dataclass(frozen=True)
class ParsedName:
    """A dataset name decomposed into its meaningful parts.

    Names look like ``nbastats_2020``, ``matchups_po_2019`` or
    ``wnba_shotdetail_po_2021``: a source, an optional ``po`` playoff marker,
    and a starting-year season. ``season`` drives the bronze partition path; the
    rest is preserved for callers that want to filter the manifest.
    """

    source: str
    season: int | None
    playoffs: bool


# source(possibly with wnba_ prefix)  _[po_]  season(4 digits)
_NAME_RE = re.compile(r"^(?P<source>.+?)(?P<po>_po)?_(?P<season>\d{4})$")


def parse_dataset_name(name: str) -> ParsedName:
    """Decompose a manifest dataset name.

    Falls back gracefully: a name that does not end in a 4-digit season is
    returned with ``season=None`` and ``playoffs=False`` rather than raising, so
    an unfamiliar manifest entry can still be fetched (it simply lands in the
    season-less bronze location).
    """
    match = _NAME_RE.match(name)
    if match is None:
        return ParsedName(source=name, season=None, playoffs=False)
    return ParsedName(
        source=match.group("source"),
        season=int(match.group("season")),
        playoffs=match.group("po") is not None,
    )


def rewrite_github_raw_url(url: str) -> str:
    """Rewrite a ``github.com/.../raw/...`` URL to ``raw.githubusercontent.com``.

    Any other URL is returned unchanged. See the module docstring for why: the
    ``raw.githubusercontent.com`` host is the one the network allowlist covers.
    """
    match = _GITHUB_RAW_RE.match(url)
    if match is None:
        return url
    owner = match.group("owner")
    repo = match.group("repo")
    rest = match.group("rest")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{rest}"


def parse_manifest(text: str) -> dict[str, str]:
    """Parse the manifest's ``name=url`` lines into a mapping.

    Blank lines and ``#`` comments are ignored. URLs are rewritten to their
    ``raw.githubusercontent.com`` form on the way in, so every consumer of the
    mapping gets a fetchable URL. A duplicated name is a corrupt manifest and
    raises rather than silently keeping one arbitrary URL.
    """
    manifest: dict[str, str] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, url = line.partition("=")
        if not sep or not name.strip() or not url.strip():
            raise ValueError(f"manifest line {lineno} is not a name=url pair: {raw_line!r}")
        name = name.strip()
        if name in manifest:
            raise ValueError(f"manifest line {lineno} redefines dataset {name!r}")
        manifest[name] = rewrite_github_raw_url(url.strip())
    return manifest


def _read_all(byte_source: ByteSource, url: str) -> bytes:
    """Drain a byte source fully into memory. For the small manifest only."""
    with byte_source(url) as chunks:
        return b"".join(chunks)


def read_manifest(byte_source: ByteSource, *, url: str | None = None) -> dict[str, str]:
    """Fetch and parse the manifest through ``byte_source``.

    ``url`` defaults to :data:`MANIFEST_URL`, resolved at call time so the module
    global can be overridden.
    """
    return parse_manifest(_read_all(byte_source, url or MANIFEST_URL).decode("utf-8"))


@dataclass(frozen=True)
class LoadResult:
    """The outcome of loading one dataset.

    ``skipped`` is true when the dataset was already ingested and the run was a
    no-op. ``rows`` is ``None`` in that case, since we do not re-open the parquet
    just to count it.
    """

    name: str
    url: str
    archive_path: Path
    parquet_path: Path
    sha256: str
    skipped: bool
    rows: int | None


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(byte_source: ByteSource, url: str, dest: Path) -> str:
    """Stream ``url`` to ``dest``, returning the SHA-256 of the bytes written.

    Writes to a sibling ``.part`` file and atomically renames on success, so a
    failure mid-transfer never leaves a usable-looking archive behind. The
    SHA-256 is computed on the fly, so a full second pass over the file is not
    needed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with byte_source(url) as chunks, part.open("wb") as out:
            for chunk in chunks:
                digest.update(chunk)
                out.write(chunk)
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def stream_archive_to_parquet(archive_path: Path, parquet_path: Path) -> int:
    """Decompress the single CSV inside ``archive_path`` straight to parquet.

    The extracted CSV is never written to disk: it is decompressed in memory and
    handed to polars, which writes parquet via a temporary file that is only
    renamed into place once the write completes. Every column is read as a
    string — a lossless 1:1 capture of the source, leaving typing to the
    bronze-conversion stage. Returns the row count written.
    """
    with tarfile.open(archive_path, mode="r:xz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".csv")]
        if len(members) != 1:
            names = [m.name for m in members]
            raise ValueError(
                f"expected exactly one CSV in {archive_path.name}, found {len(members)}: {names}"
            )
        extracted = tar.extractfile(members[0])
        if extracted is None:  # pragma: no cover - isfile() already guarantees this
            raise ValueError(f"could not extract {members[0].name} from {archive_path.name}")
        data = extracted.read()

    frame = pl.read_csv(io.BytesIO(data), infer_schema_length=0)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet_path.with_name(parquet_path.name + ".tmp")
    try:
        frame.write_parquet(tmp)
        tmp.replace(parquet_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return frame.height


def _archive_path(profile: StorageProfile, name: str) -> Path:
    return profile.zone(Zone.RAW) / f"{name}.tar.xz"


def bronze_parquet_path(profile: StorageProfile, name: str) -> Path:
    """Resolve the bronze parquet path for a dataset.

    The single source of truth for where a dataset's bronze table lands:
    ``bronze/season=<year>/<name>.parquet``, season-partitioned via
    :meth:`StorageProfile.zone`. The bulk loader writes the initial lossless
    string capture here; the bronze conversion stage
    (:mod:`hoopstate.ingest.bronze`) rewrites the same path with explicit types.
    Both share this helper so the location is defined in exactly one place.
    """
    parsed = parse_dataset_name(name)
    bronze = profile.zone(Zone.BRONZE, season=parsed.season)
    return bronze / f"{name}.parquet"


def load_dataset(
    name: str,
    *,
    byte_source: ByteSource,
    profile: StorageProfile | None = None,
    manifest: dict[str, str] | None = None,
    force: bool = False,
) -> LoadResult:
    """Download, cache, and convert one named dataset to parquet.

    ``manifest`` may be supplied to avoid a second network round-trip when
    loading many datasets; otherwise it is fetched through ``byte_source``.
    ``profile`` defaults to the process's resolved storage profile.

    Re-running is a no-op: if the parquet already exists (and ``force`` is not
    set) the function returns immediately with ``skipped=True``. If only the
    archive is cached, it is reused rather than re-downloaded. ``force`` re-does
    both the download and the conversion.
    """
    if profile is None:
        profile = resolve_profile()
    if manifest is None:
        manifest = read_manifest(byte_source)
    if name not in manifest:
        raise KeyError(f"dataset {name!r} is not in the manifest")

    url = manifest[name]
    archive_path = _archive_path(profile, name)
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    parquet_path = bronze_parquet_path(profile, name)

    if parquet_path.exists() and not force:
        sha = sidecar.read_text().strip() if sidecar.exists() else ""
        return LoadResult(
            name=name,
            url=url,
            archive_path=archive_path,
            parquet_path=parquet_path,
            sha256=sha,
            skipped=True,
            rows=None,
        )

    # Reuse a cached archive when its sidecar checksum still matches; otherwise
    # (re)download. The checksum both proves the cached bytes are intact and is
    # what lets a completed dataset skip the fetch entirely.
    have_valid_cache = (
        not force
        and archive_path.exists()
        and sidecar.exists()
        and sidecar.read_text().strip() == _sha256_of(archive_path)
    )
    if have_valid_cache:
        sha = sidecar.read_text().strip()
    else:
        sha = _download(byte_source, url, archive_path)
        sidecar.write_text(sha + "\n")

    rows = stream_archive_to_parquet(archive_path, parquet_path)
    return LoadResult(
        name=name,
        url=url,
        archive_path=archive_path,
        parquet_path=parquet_path,
        sha256=sha,
        skipped=False,
        rows=rows,
    )


def requests_byte_source(
    session: object | None = None, *, chunk_size: int = _DOWNLOAD_CHUNK, timeout: float = 60.0
) -> ByteSource:
    """A :class:`ByteSource` backed by :mod:`requests`.

    ``requests`` is imported lazily so that importing this module — and running
    the offline-testable logic — never requires the dependency to be present or
    the network to be reachable.
    """
    import requests

    active = session if session is not None else requests.Session()

    @contextmanager
    def _open(url: str) -> Iterator[Iterator[bytes]]:
        response = active.get(url, stream=True, timeout=timeout)
        try:
            response.raise_for_status()
            yield response.iter_content(chunk_size=chunk_size)
        finally:
            response.close()

    return _open


def main(argv: list[str] | None = None) -> int:
    """Minimal CLI: ``python -m hoopstate.ingest.bulk_loader <name> [<name> ...]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bulk-load shufinskiy/nba_data datasets to parquet."
    )
    parser.add_argument(
        "names", nargs="+", help="Dataset names from the manifest, e.g. nbastats_2020."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download and re-convert even if cached."
    )
    args = parser.parse_args(argv)

    byte_source = requests_byte_source()
    profile = resolve_profile()
    manifest = read_manifest(byte_source)
    for name in args.names:
        result = load_dataset(
            name, byte_source=byte_source, profile=profile, manifest=manifest, force=args.force
        )
        if result.skipped:
            print(f"{name}: already ingested -> {result.parquet_path}")
        else:
            print(f"{name}: {result.rows} rows -> {result.parquet_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
