"""Tests for the quarantined pbpstats oracle ingest (``hoops-1lg.2.4``).

Offline, like the other ingest tests: fixture ``tar.xz`` archives are built
in-process and served through an in-memory
:class:`~hoopstate.ingest.bulk_loader.ByteSource`.

Acceptance criteria covered:

* ``pbpstats_2023`` lands in the oracle zone as typed parquet.
* The archive and its checksum sidecar land there too — **nothing** belonging to
  the oracle is written to the shared raw zone, so no core-model glob over
  ``raw/`` can sweep the answer key in.
* Types are explicit, per :data:`~hoopstate.validate.oracle.ORACLE_SCHEMA`:
  ``GAMEID`` a string in the same zero-stripped form bronze stores, the
  ``STARTTIME``/``ENDTIME`` clock strings kept as strings, counts Int64.
* Schema drift fails loudly rather than being guessed at.
* A non-oracle dataset is refused, so the oracle zone stays single-purpose.

The static half of the quarantine — that no module outside ``validate/`` may
name the oracle at all — lives in ``test_oracle_quarantine.py``.
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
from hoopstate.storage import ENV_COLD, ENV_HOT, ENV_PROFILE, Zone, resolve_profile
from hoopstate.validate.oracle import (
    DEFAULT_ORACLE_DATASET,
    ORACLE_SCHEMA,
    ingest_oracle,
    oracle_archive_dir,
    oracle_parquet_path,
)

# --- fixtures ---------------------------------------------------------------


def _make_archive(csv_name: str, csv_text: str) -> bytes:
    csv_bytes = csv_text.encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        info = tarfile.TarInfo(name=csv_name)
        info.size = len(csv_bytes)
        tar.addfile(info, io.BytesIO(csv_bytes))
    return buf.getvalue()


def _byte_source_from(mapping: dict[str, bytes]) -> bulk_loader.ByteSource:
    @contextmanager
    def _open(url: str) -> Iterator[Iterator[bytes]]:
        if url not in mapping:
            raise KeyError(f"no fixture for {url}")
        payload = mapping[url]
        yield iter(payload[i : i + 7] for i in range(0, len(payload), 7))

    return _open


# Two possessions from one game, shaped like the real file: an "MM:SS" clock
# counting down, the "Off " start-type vocabulary, a multi-line EVENTS field,
# and a null URL on the possession with no video.
_HEADER = ",".join(ORACLE_SCHEMA)
_ROWS = [
    (
        '"11:41","Jump Ball Embiid vs. Adebayo\nMaxey 3PT Shot: Made (3 PTS)",'
        "0,0,1,1,2023-10-25,22300061,0,0,MIA,1,0,0,"
        '"12:00","Off Dead Ball",0,"Maxey 26\' 3PT Jump Shot (3 PTS)",'
        "https://videos.nba.com/nba/pbp/media/1.mp4"
    ),
    (
        '"11:20","Butler 2PT Shot: Missed\nOubre REBOUND (Off:1)",'
        "1,0,0,0,2023-10-25,22300061,0,1,PHI,1,0,-3,"
        '"11:41","Off Arc 3 Make",0,"Butler Driving Layup Shot: Missed",'
    ),
]
_CSV = "\n".join([_HEADER, *_ROWS]) + "\n"

# Simple, quote-free values keyed by dtype, for the drift fixtures below. The
# realistic rows above are the wrong tool there: their quoted multi-line fields
# make adding or removing a column by string surgery unreliable.
_PLACEHOLDER = {pl.String(): "x", pl.Int64(): "1"}


def _drifted_csv(*, added: str | None = None, dropped: str | None = None) -> str:
    """A one-row CSV whose column set deliberately differs from the schema."""
    columns = [c for c in ORACLE_SCHEMA if c != dropped]
    values = [_PLACEHOLDER[ORACLE_SCHEMA[c]] for c in columns]
    if added is not None:
        columns.append(added)
        values.append("1")
    return ",".join(columns) + "\n" + ",".join(values) + "\n"


_URL = f"https://example.com/{DEFAULT_ORACLE_DATASET}.tar.xz"
_MANIFEST = {DEFAULT_ORACLE_DATASET: _URL}


@pytest.fixture
def ephemeral_profile(tmp_path: Path):
    return resolve_profile(
        env={ENV_PROFILE: "ephemeral", ENV_HOT: str(tmp_path), ENV_COLD: str(tmp_path)}
    )


def _ingest(profile, *, csv: str = _CSV, **kwargs):
    archive = _make_archive(f"{DEFAULT_ORACLE_DATASET}.csv", csv)
    return ingest_oracle(
        byte_source=_byte_source_from({_URL: archive}),
        profile=profile,
        manifest=_MANIFEST,
        **kwargs,
    )


# --- the parquet ------------------------------------------------------------


def test_oracle_lands_as_typed_parquet(ephemeral_profile) -> None:
    result = _ingest(ephemeral_profile)

    expected = oracle_parquet_path(DEFAULT_ORACLE_DATASET, profile=ephemeral_profile)
    assert result.parquet_path == expected
    assert expected.exists()
    assert result.rows == len(_ROWS)

    frame = pl.read_parquet(expected)
    assert frame.height == len(_ROWS)
    assert dict(frame.schema) == ORACLE_SCHEMA


def test_gameid_keeps_the_form_bronze_stores(ephemeral_profile) -> None:
    """The oracle joins to bronze on GAMEID without re-padding.

    Both sides carry the leading-zero-stripped 8-character id. Asserting it
    keeps a well-meant "normalisation" from breaking the join later.
    """
    result = _ingest(ephemeral_profile)
    frame = pl.read_parquet(result.parquet_path)
    assert frame.schema["GAMEID"] == pl.String()
    assert frame["GAMEID"].to_list() == ["22300061", "22300061"]


def test_clock_columns_stay_strings(ephemeral_profile) -> None:
    """``"11:41"`` is a clock reading, not a number; turning it into one loses it."""
    result = _ingest(ephemeral_profile)
    frame = pl.read_parquet(result.parquet_path)
    assert frame.schema["STARTTIME"] == pl.String()
    assert frame.schema["ENDTIME"] == pl.String()
    assert frame["STARTTIME"].to_list() == ["12:00", "11:41"]


def test_counts_are_integers_and_differentials_signed(ephemeral_profile) -> None:
    result = _ingest(ephemeral_profile)
    frame = pl.read_parquet(result.parquet_path)
    assert frame.schema["FG3M"] == pl.Int64()
    assert frame.schema["PERIOD"] == pl.Int64()
    assert frame["STARTSCOREDIFFERENTIAL"].to_list() == [0, -3]


def test_multiline_events_survive(ephemeral_profile) -> None:
    """EVENTS embeds newlines in 477,607 of the real file's rows."""
    result = _ingest(ephemeral_profile)
    frame = pl.read_parquet(result.parquet_path)
    assert "\n" in frame["EVENTS"][0]


def test_missing_url_is_null(ephemeral_profile) -> None:
    result = _ingest(ephemeral_profile)
    frame = pl.read_parquet(result.parquet_path)
    assert frame["URL"][1] is None


# --- the quarantine, physically ---------------------------------------------


def test_nothing_lands_in_the_shared_raw_zone(ephemeral_profile) -> None:
    """The whole answer key stays in one directory, source bytes included."""
    _ingest(ephemeral_profile)

    archive_dir = oracle_archive_dir(ephemeral_profile)
    archive = archive_dir / f"{DEFAULT_ORACLE_DATASET}.tar.xz"
    assert archive.exists()
    assert (archive_dir / f"{DEFAULT_ORACLE_DATASET}.tar.xz.sha256").exists()

    oracle_root = ephemeral_profile.zone(Zone.ORACLE)
    assert archive.is_relative_to(oracle_root)

    raw = ephemeral_profile.zone(Zone.RAW)
    # Under the ephemeral profile every zone shares one root, so an existence
    # check on the directory would be meaningless; look for the files instead.
    assert list(raw.glob("pbpstats*")) == []


def test_oracle_tree_is_hot_under_the_local_profile(tmp_path: Path) -> None:
    """The answer key is read constantly during validation; it belongs on fast disk."""
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    profile = resolve_profile(env={ENV_PROFILE: "local", ENV_HOT: str(hot), ENV_COLD: str(cold)})
    result = _ingest(profile)
    assert result.parquet_path.is_relative_to(hot)
    assert oracle_archive_dir(profile).is_relative_to(hot)
    assert not list(cold.rglob("pbpstats*"))


# --- refusals ---------------------------------------------------------------


def test_a_non_oracle_dataset_is_refused(ephemeral_profile) -> None:
    """The oracle zone stays single-purpose: only pbpstats may be written there."""
    with pytest.raises(ValueError, match="not an oracle dataset"):
        ingest_oracle(
            "nbastats_2023",
            byte_source=_byte_source_from({}),
            profile=ephemeral_profile,
            manifest={"nbastats_2023": "https://example.com/nbastats_2023.tar.xz"},
        )


def test_an_extra_column_fails_loudly(ephemeral_profile) -> None:
    """A column appearing upstream is drift; it must not be silently dropped."""
    with pytest.raises(ValueError, match=r"unexpected=\['SURPRISE'\]"):
        _ingest(ephemeral_profile, csv=_drifted_csv(added="SURPRISE"))


def test_a_missing_column_fails_loudly(ephemeral_profile) -> None:
    """A column disappearing upstream must not become a silent null column."""
    with pytest.raises(ValueError, match=r"missing=\['TURNOVERS'\]"):
        _ingest(ephemeral_profile, csv=_drifted_csv(dropped="TURNOVERS"))


# --- caching ----------------------------------------------------------------


def test_a_second_run_reuses_the_cached_archive(ephemeral_profile) -> None:
    """A verified checksum sidecar means the second run never touches the network."""
    _ingest(ephemeral_profile)

    def _refuse(url: str):
        raise AssertionError(f"re-downloaded {url} despite a valid cache")

    ingest_oracle(
        byte_source=_refuse,
        profile=ephemeral_profile,
        manifest=_MANIFEST,
    )
