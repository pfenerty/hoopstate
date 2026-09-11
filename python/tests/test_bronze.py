"""Tests for bronze conversion of play-by-play sources (``hoops-1lg.2.2``).

Fully offline: the string-capture parquet the bulk loader would produce is
built in-process and written to the dataset's real bronze path, so the
conversion is exercised end to end without touching the network.

Acceptance criteria covered:

* ``nbastats_2023`` and ``datanba_2023`` land as typed parquet in bronze.
* Column types are explicit (from the registered schema), not inferred.
* Row count is invariant across the conversion (1:1 with source).
* The ``datanba`` offense-team-id cross-check column survives as an integer.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from hoopstate.ingest import bronze
from hoopstate.ingest.bronze import (
    SOURCE_SCHEMAS,
    apply_schema,
    convert_dataset,
    schema_for_dataset,
)
from hoopstate.ingest.bulk_loader import bronze_parquet_path
from hoopstate.storage import ENV_COLD, ENV_HOT, ENV_PROFILE, resolve_profile


@pytest.fixture
def ephemeral_profile(tmp_path: Path):
    return resolve_profile(
        env={ENV_PROFILE: "ephemeral", ENV_HOT: str(tmp_path), ENV_COLD: str(tmp_path)}
    )


def _string_frame(schema: dict[str, pl.DataType], rows: int) -> pl.DataFrame:
    """Build a valid all-String capture frame for a schema, like the loader's.

    Integer columns get integer-looking text, string columns get plain text;
    every column is polars ``String``, mirroring ``infer_schema_length=0``.
    """
    data: dict[str, list[str]] = {}
    for col, dtype in schema.items():
        if dtype == pl.Int64:
            data[col] = [str(i) for i in range(rows)]
        else:
            data[col] = [f"{col}-{i}" for i in range(rows)]
    return pl.DataFrame(data, schema={c: pl.String for c in schema})


def _write_capture(profile, name: str, frame: pl.DataFrame) -> Path:
    """Write ``frame`` to the dataset's bronze path, as the loader would."""
    path = bronze_parquet_path(profile, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


# --- schema registry --------------------------------------------------------


def test_schema_for_dataset_resolves_known_sources() -> None:
    src, schema = schema_for_dataset("nbastats_2023")
    assert src == "nbastats"
    assert schema is SOURCE_SCHEMAS["nbastats"]
    # The playoff variant shares the same source schema.
    assert schema_for_dataset("nbastats_po_2019")[1] is SOURCE_SCHEMAS["nbastats"]
    assert schema_for_dataset("datanba_2023")[0] == "datanba"


def test_schema_for_dataset_rejects_unknown_source() -> None:
    with pytest.raises(KeyError, match="no bronze schema for source 'matchups'"):
        schema_for_dataset("matchups_2023")


def test_datanba_schema_carries_offense_team_id_as_int() -> None:
    # oftid is the whole reason datanba is converted alongside nbastats.
    assert SOURCE_SCHEMAS["datanba"]["oftid"] == pl.Int64


# --- apply_schema semantics -------------------------------------------------

_MINI: dict[str, pl.DataType] = {"id": pl.Int64, "margin": pl.String, "note": pl.String}


def test_apply_schema_casts_and_preserves_nulls_and_sentinels() -> None:
    frame = pl.DataFrame(
        {
            "id": ["0", "5", None],  # 0 is a real sentinel, not null
            "margin": ["3", "TIE", None],  # "TIE" must survive as text
            "note": ["hi", None, "x"],
        },
        schema={"id": pl.String, "margin": pl.String, "note": pl.String},
    )
    out = apply_schema(frame, _MINI)

    assert out.schema == {"id": pl.Int64, "margin": pl.String, "note": pl.String}
    assert out["id"].to_list() == [0, 5, None]
    assert out["margin"].to_list() == ["3", "TIE", None]
    assert out.height == 3


def test_apply_schema_returns_columns_in_schema_order() -> None:
    frame = pl.DataFrame(
        {"note": ["a"], "id": ["1"], "margin": ["2"]},
        schema={"note": pl.String, "id": pl.String, "margin": pl.String},
    )
    assert apply_schema(frame, _MINI).columns == ["id", "margin", "note"]


def test_apply_schema_rejects_column_mismatch() -> None:
    frame = pl.DataFrame(
        {"id": ["1"], "extra": ["x"]}, schema={"id": pl.String, "extra": pl.String}
    )
    with pytest.raises(ValueError, match=r"missing=\['margin', 'note'\] unexpected=\['extra'\]"):
        apply_schema(frame, _MINI)


def test_apply_schema_strict_cast_failure_raises() -> None:
    frame = pl.DataFrame(
        {"id": ["not-an-int"], "margin": ["1"], "note": ["x"]},
        schema={"id": pl.String, "margin": pl.String, "note": pl.String},
    )
    with pytest.raises(pl.exceptions.PolarsError):
        apply_schema(frame, _MINI)


# --- convert_dataset end to end --------------------------------------------


@pytest.mark.parametrize("name", ["nbastats_2023", "datanba_2023"])
def test_convert_dataset_writes_typed_parquet(ephemeral_profile, name: str) -> None:
    _src, schema = schema_for_dataset(name)
    capture = _string_frame(schema, rows=4)
    assert set(capture.dtypes) == {pl.String}  # starts as a pure string capture
    path = _write_capture(ephemeral_profile, name, capture)

    result = convert_dataset(name, profile=ephemeral_profile)

    assert result.parquet_path == path
    assert result.rows == 4  # row count is invariant: 1:1 with source
    on_disk = pl.read_parquet(path)
    assert on_disk.schema == dict(schema)  # explicit types, not string
    assert on_disk.columns == list(schema)  # source column order preserved
    assert result.schema == {c: str(dt) for c, dt in schema.items()}
    # No temp file left behind by the atomic write.
    assert list(path.parent.glob("*.tmp")) == []


def test_convert_dataset_is_idempotent(ephemeral_profile) -> None:
    name = "datanba_2023"
    _src, schema = schema_for_dataset(name)
    _write_capture(ephemeral_profile, name, _string_frame(schema, rows=3))

    first = convert_dataset(name, profile=ephemeral_profile)
    # Second run re-reads the now-typed parquet; the cast is a no-op.
    second = convert_dataset(name, profile=ephemeral_profile)

    assert first.schema == second.schema
    assert second.rows == 3
    assert pl.read_parquet(second.parquet_path).schema == dict(schema)


def test_convert_dataset_missing_capture_raises(ephemeral_profile) -> None:
    with pytest.raises(FileNotFoundError, match="run the bulk loader first"):
        convert_dataset("nbastats_2023", profile=ephemeral_profile)


def test_convert_dataset_resolves_profile_when_omitted(ephemeral_profile, monkeypatch) -> None:
    # With no profile passed, convert_dataset falls back to resolve_profile();
    # pin that to the isolated ephemeral profile so the test stays hermetic.
    monkeypatch.setattr(bronze, "resolve_profile", lambda: ephemeral_profile)
    name = "nbastats_2023"
    _src, schema = schema_for_dataset(name)
    _write_capture(ephemeral_profile, name, _string_frame(schema, rows=2))

    result = convert_dataset(name)
    assert result.rows == 2
    assert pl.read_parquet(result.parquet_path).schema == dict(schema)
