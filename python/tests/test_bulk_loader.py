"""Tests for the manifest-driven bulk loader (``hoops-1lg.2.1``).

These assert the acceptance criteria directly and run entirely offline: the
network is stubbed by a :class:`ByteSource` that serves bytes from an in-memory
map, and archives are built in-process with :mod:`tarfile`/:mod:`lzma`.

Acceptance criteria covered:

* Loader parses the manifest.
* Downloads a named dataset and writes parquet with no intermediate CSV on disk.
* Verifies a checksum and caches so a completed dataset is never re-fetched.
* Re-running is a no-op.
* Raw archives land in the (cold) raw zone.
* Failure mid-download leaves no partial parquet (and no usable archive).
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pytest

from hoopstate.ingest import bulk_loader
from hoopstate.ingest.bulk_loader import (
    LoadResult,
    parse_dataset_name,
    parse_manifest,
    read_manifest,
    rewrite_github_raw_url,
    stream_archive_to_parquet,
)
from hoopstate.storage import ENV_COLD, ENV_HOT, ENV_PROFILE, Zone, resolve_profile

# --- fixtures ---------------------------------------------------------------


def _make_archive(csv_name: str, csv_text: str) -> bytes:
    """Build a ``tar.xz`` holding a single CSV, exactly like the real archives."""
    csv_bytes = csv_text.encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        info = tarfile.TarInfo(name=csv_name)
        info.size = len(csv_bytes)
        tar.addfile(info, io.BytesIO(csv_bytes))
    return buf.getvalue()


def _byte_source_from(mapping: dict[str, bytes]) -> bulk_loader.ByteSource:
    """A ByteSource serving fixed bytes per URL; raises on an unknown URL."""

    @contextmanager
    def _open(url: str) -> Iterator[Iterator[bytes]]:
        if url not in mapping:
            raise KeyError(f"no fixture for {url}")
        # Chunk it, to exercise the streaming/reassembly path.
        payload = mapping[url]
        yield iter(payload[i : i + 7] for i in range(0, len(payload), 7))

    return _open


CSV_TEXT = "game_id,points\n42000101,24\n42000102,7\n"


@pytest.fixture
def ephemeral_profile(tmp_path: Path):
    # Override BOTH roots so the ephemeral hot==cold collapse holds under an
    # isolated per-test directory; overriding only hot would leave cold pointing
    # at the shared scratch root and leak archives between tests.
    return resolve_profile(
        env={ENV_PROFILE: "ephemeral", ENV_HOT: str(tmp_path), ENV_COLD: str(tmp_path)}
    )


# --- pure helpers -----------------------------------------------------------


def test_parse_manifest_basic() -> None:
    text = "\n".join(
        [
            "# a comment",
            "",
            "nbastats_2020=https://raw.githubusercontent.com/x/y/main/a.tar.xz",
            "matchups_po_2019=https://github.com/x/y/raw/main/b.tar.xz",
        ]
    )
    manifest = parse_manifest(text)
    assert set(manifest) == {"nbastats_2020", "matchups_po_2019"}
    # github.com/.../raw/... is rewritten to raw.githubusercontent.com.
    assert manifest["matchups_po_2019"] == "https://raw.githubusercontent.com/x/y/main/b.tar.xz"


def test_parse_manifest_rejects_malformed_line() -> None:
    with pytest.raises(ValueError, match="not a name=url"):
        parse_manifest("this-has-no-equals-sign")


def test_parse_manifest_rejects_duplicate_name() -> None:
    with pytest.raises(ValueError, match="redefines"):
        parse_manifest("a=http://x\na=http://y")


def test_rewrite_github_raw_url() -> None:
    assert (
        rewrite_github_raw_url("https://github.com/shufinskiy/nba_data/raw/main/datasets/x.tar.xz")
        == "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/x.tar.xz"
    )
    # Non-matching URLs pass through untouched.
    other = "https://example.com/a.tar.xz"
    assert rewrite_github_raw_url(other) == other


@pytest.mark.parametrize(
    ("name", "source", "season", "playoffs"),
    [
        ("nbastats_2020", "nbastats", 2020, False),
        ("matchups_po_2019", "matchups", 2019, True),
        ("wnba_shotdetail_po_2021", "wnba_shotdetail", 2021, True),
        ("weird_no_season", "weird_no_season", None, False),
    ],
)
def test_parse_dataset_name(name: str, source: str, season: int | None, playoffs: bool) -> None:
    parsed = parse_dataset_name(name)
    assert (parsed.source, parsed.season, parsed.playoffs) == (source, season, playoffs)


def test_read_manifest_uses_byte_source() -> None:
    url = "https://example.com/list.txt"
    source = _byte_source_from({url: b"nbastats_2020=https://example.com/a.tar.xz\n"})
    manifest = read_manifest(source, url=url)
    assert manifest == {"nbastats_2020": "https://example.com/a.tar.xz"}


# --- stream_archive_to_parquet ---------------------------------------------


def test_stream_archive_to_parquet_is_lossless_strings(tmp_path: Path) -> None:
    archive = tmp_path / "d.tar.xz"
    archive.write_bytes(_make_archive("d.csv", CSV_TEXT))
    parquet = tmp_path / "out" / "d.parquet"

    rows = stream_archive_to_parquet(archive, parquet)

    assert rows == 2
    frame = pl.read_parquet(parquet)
    assert frame.columns == ["game_id", "points"]
    assert set(frame.dtypes) == {pl.String}
    assert frame["points"].to_list() == ["24", "7"]


def test_stream_archive_rejects_multi_csv(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        for n in ("a.csv", "b.csv"):
            b = b"x\n1\n"
            info = tarfile.TarInfo(name=n)
            info.size = len(b)
            tar.addfile(info, io.BytesIO(b))
    archive = tmp_path / "multi.tar.xz"
    archive.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="exactly one CSV"):
        stream_archive_to_parquet(archive, tmp_path / "x.parquet")


def test_stream_archive_corrupt_raises_and_writes_no_parquet(tmp_path: Path) -> None:
    archive = tmp_path / "corrupt.tar.xz"
    archive.write_bytes(b"not a real xz archive")
    parquet = tmp_path / "out.parquet"
    with pytest.raises(tarfile.ReadError):
        stream_archive_to_parquet(archive, parquet)
    assert not parquet.exists()
    assert not parquet.with_name(parquet.name + ".tmp").exists()


# --- load_dataset end to end ------------------------------------------------


MANIFEST_FIXTURE_URL = "https://example.com/list_data.txt"


def _one_dataset(name: str = "matchups_po_2019") -> tuple[str, dict[str, bytes], dict[str, str]]:
    """name + byte map (manifest + archive) + parsed manifest for one dataset."""
    url = f"https://example.com/{name}.tar.xz"
    manifest_text = f"{name}={url}\n"
    archive = _make_archive(f"{name}.csv", CSV_TEXT)
    byte_map = {MANIFEST_FIXTURE_URL: manifest_text.encode(), url: archive}
    return name, byte_map, {name: url}


def test_load_dataset_downloads_and_writes_parquet(ephemeral_profile) -> None:
    name, byte_map, manifest = _one_dataset()
    source = _byte_source_from(byte_map)

    result = bulk_loader.load_dataset(
        name, byte_source=source, profile=ephemeral_profile, manifest=manifest
    )

    assert isinstance(result, LoadResult)
    assert result.skipped is False
    assert result.rows == 2
    # Parquet exists and is real.
    assert result.parquet_path.exists()
    assert pl.read_parquet(result.parquet_path).height == 2
    # Raw archive landed in the raw zone, with a checksum sidecar.
    raw_zone = ephemeral_profile.zone(Zone.RAW)
    assert result.archive_path.parent == raw_zone
    assert result.archive_path.exists()
    sidecar = result.archive_path.with_name(result.archive_path.name + ".sha256")
    assert sidecar.exists()
    assert sidecar.read_text().strip() == result.sha256
    # Parquet landed under bronze, season-partitioned.
    assert result.parquet_path.is_relative_to(ephemeral_profile.zone(Zone.BRONZE))
    assert "season=2019" in str(result.parquet_path)


def test_no_intermediate_csv_on_disk(ephemeral_profile) -> None:
    name, byte_map, manifest = _one_dataset()
    source = _byte_source_from(byte_map)
    bulk_loader.load_dataset(name, byte_source=source, profile=ephemeral_profile, manifest=manifest)
    # Nothing with a .csv suffix is ever written anywhere under the profile root.
    stray = list(ephemeral_profile.hot_root.rglob("*.csv"))
    assert stray == []


def test_rerun_is_a_noop(ephemeral_profile) -> None:
    name, byte_map, manifest = _one_dataset()

    calls: list[str] = []

    def counting_source(url: str):
        calls.append(url)
        return _byte_source_from(byte_map)(url)

    first = bulk_loader.load_dataset(
        name, byte_source=counting_source, profile=ephemeral_profile, manifest=manifest
    )
    assert first.skipped is False
    downloads_after_first = list(calls)

    second = bulk_loader.load_dataset(
        name, byte_source=counting_source, profile=ephemeral_profile, manifest=manifest
    )
    assert second.skipped is True
    assert second.rows is None
    dataset_url = byte_map_url(byte_map, name)
    assert dataset_url in downloads_after_first
    assert calls.count(dataset_url) == 1, "dataset was re-downloaded on the no-op run"


def byte_map_url(byte_map: dict[str, bytes], name: str) -> str:
    return next(u for u in byte_map if u.endswith(f"{name}.tar.xz"))


def test_cached_archive_is_reused_when_parquet_missing(ephemeral_profile) -> None:
    name, byte_map, manifest = _one_dataset()
    dataset_url = byte_map_url(byte_map, name)

    calls: list[str] = []

    def counting_source(url: str):
        calls.append(url)
        return _byte_source_from(byte_map)(url)

    result = bulk_loader.load_dataset(
        name, byte_source=counting_source, profile=ephemeral_profile, manifest=manifest
    )
    # Delete only the parquet, keep the cached archive + sidecar.
    result.parquet_path.unlink()

    again = bulk_loader.load_dataset(
        name, byte_source=counting_source, profile=ephemeral_profile, manifest=manifest
    )
    assert again.skipped is False
    assert again.parquet_path.exists()
    # The archive was NOT downloaded a second time.
    assert calls.count(dataset_url) == 1


def test_force_redownloads(ephemeral_profile) -> None:
    name, byte_map, manifest = _one_dataset()
    dataset_url = byte_map_url(byte_map, name)

    calls: list[str] = []

    def counting_source(url: str):
        calls.append(url)
        return _byte_source_from(byte_map)(url)

    bulk_loader.load_dataset(
        name, byte_source=counting_source, profile=ephemeral_profile, manifest=manifest
    )
    bulk_loader.load_dataset(
        name, byte_source=counting_source, profile=ephemeral_profile, manifest=manifest, force=True
    )
    assert calls.count(dataset_url) == 2


def test_unknown_dataset_raises(ephemeral_profile) -> None:
    _name, byte_map, manifest = _one_dataset()
    source = _byte_source_from(byte_map)
    with pytest.raises(KeyError, match="not in the manifest"):
        bulk_loader.load_dataset(
            "does_not_exist", byte_source=source, profile=ephemeral_profile, manifest=manifest
        )


def test_failed_download_leaves_no_partial_archive_or_parquet(ephemeral_profile) -> None:
    name, byte_map, manifest = _one_dataset()
    dataset_url = byte_map_url(byte_map, name)

    @contextmanager
    def flaky_source(url: str):
        if url == dataset_url:

            def gen():
                yield b"partial-bytes-then-boom"
                raise OSError("connection reset mid-download")

            yield gen()
        else:
            with _byte_source_from(byte_map)(url) as chunks:
                yield chunks

    with pytest.raises(OSError, match="connection reset"):
        bulk_loader.load_dataset(
            name, byte_source=flaky_source, profile=ephemeral_profile, manifest=manifest
        )

    # No archive, no .part, no parquet survived the failure.
    raw_zone = ephemeral_profile.zone(Zone.RAW)
    assert list(raw_zone.glob("*.tar.xz")) == []
    assert list(raw_zone.glob("*.part")) == []
    assert list(ephemeral_profile.zone(Zone.BRONZE).rglob("*.parquet")) == []


def test_manifest_is_fetched_when_not_passed(ephemeral_profile, monkeypatch) -> None:
    name, byte_map, _manifest = _one_dataset()
    source = _byte_source_from(byte_map)
    # Point the module's default manifest URL at our fixture; pass no manifest,
    # so the loader must fetch and parse it itself.
    monkeypatch.setattr(bulk_loader, "MANIFEST_URL", MANIFEST_FIXTURE_URL)
    result = bulk_loader.load_dataset(name, byte_source=source, profile=ephemeral_profile)
    assert result.rows == 2
