"""The oracle quarantine, enforced mechanically (``hoops-1lg.2.4``).

pbpstats is this project's answer key. If a module that derives possessions
could read it, validation would end up comparing pbpstats to itself and
reporting a match rate that means nothing. The rule is therefore absolute:
**only ``hoopstate.validate`` may name the oracle.**

A rule nobody can forget is better than one written down, so this file scans
every module under ``src/hoopstate/`` and fails if one outside ``validate/``
reaches for the oracle — by the :class:`~hoopstate.storage.Zone` member, by
importing the ``validate`` package, or by spelling a path or source name out as
a string to dodge the enum. ``pyproject.toml`` sets ``testpaths = ["tests"]``,
so this runs as part of the default ``pytest`` invocation with no extra wiring
— which is the acceptance criterion that the guard be part of the default test
run.

The scan lives in the test rather than in the package on purpose: it is an
assertion about the shape of the repository, not behaviour the library offers.

``storage.py`` is the one unavoidable exemption — it *defines*
``Zone.ORACLE``, so it cannot be written without naming it. The ingester itself
needs no exemption because it lives in ``validate/`` (see
:mod:`hoopstate.validate.oracle`), which keeps the rule free of carve-outs for
the very code most able to leak.

Docstrings are exempt: prose *about* the quarantine is how a reader learns the
rule exists. :mod:`hoopstate.storage` and :mod:`hoopstate.ingest.reports` both
discuss the oracle in their module docstrings and must keep passing. Comments
never reach the AST, so they need no handling.
"""

from __future__ import annotations

import ast
from pathlib import Path

import duckdb
import polars as pl
import pytest

from hoopstate.db.catalog import rebuild
from hoopstate.ingest.bronze import schema_for_source
from hoopstate.storage import Profile, StorageProfile, Zone

# python/tests/ -> python/ -> python/src/hoopstate/
SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "hoopstate"

# The allowlisted package: the only place the oracle may be named.
ALLOWED_PACKAGE = "validate"

# Modules exempt from the scan, relative to SRC_ROOT. Keep this list as close to
# empty as the code allows; every entry is a hole in the guard.
EXEMPT = frozenset({Path("storage.py")})

# Strings that would let a module reach the oracle without touching the enum.
_FORBIDDEN_SUBSTRING = "pbpstats"
_FORBIDDEN_SEGMENT = "oracle"


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """Identity of every docstring expression in ``tree``.

    Matched by ``id()`` rather than by value so an ordinary string constant that
    happens to equal a docstring is still scanned.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def _string_violation(value: str) -> str | None:
    """Why ``value`` looks like an attempt to reach the oracle by string, if it does."""
    lowered = value.lower()
    if _FORBIDDEN_SUBSTRING in lowered:
        return f"string names the oracle source: {value!r}"
    if lowered == _FORBIDDEN_SEGMENT or _FORBIDDEN_SEGMENT in lowered.split("/"):
        return f"string names the oracle zone path: {value!r}"
    return None


def _violations(source: str, path: str) -> list[str]:
    """Every way ``source`` reaches for the oracle, as human-readable messages.

    ``path`` is used only to make failures locatable.
    """
    tree = ast.parse(source, filename=path)
    docstrings = _docstring_nodes(tree)
    found: list[str] = []

    def report(node: ast.AST, message: str) -> None:
        found.append(f"{path}:{getattr(node, 'lineno', '?')}: {message}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "ORACLE":
            report(node, "references Zone.ORACLE")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "hoopstate.validate" or alias.name.startswith(
                    "hoopstate.validate."
                ):
                    report(node, f"imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "hoopstate.validate" or module.startswith("hoopstate.validate."):
                report(node, f"imports from {module}")
            elif module == "hoopstate":
                for alias in node.names:
                    if alias.name == ALLOWED_PACKAGE:
                        report(node, "imports the validate package")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            message = _string_violation(node.value)
            if message is not None:
                report(node, message)

    return found


def _scanned_modules() -> list[Path]:
    """Every module the quarantine applies to, as paths relative to ``SRC_ROOT``."""
    modules = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(SRC_ROOT)
        if relative.parts[0] == ALLOWED_PACKAGE or relative in EXEMPT:
            continue
        modules.append(relative)
    return modules


# --- the guard --------------------------------------------------------------


def test_the_scan_actually_covers_modules() -> None:
    """A guard over an empty file list would pass for the wrong reason."""
    modules = _scanned_modules()
    assert len(modules) >= 4, f"expected several modules to scan, found {modules}"
    # The derivation code is the whole reason the quarantine exists; if the
    # package layout moves, this is the assertion that notices.
    assert Path("ingest/bronze.py") in modules


@pytest.mark.parametrize("relative", _scanned_modules(), ids=str)
def test_no_module_outside_validate_reaches_the_oracle(relative: Path) -> None:
    path = SRC_ROOT / relative
    found = _violations(path.read_text(encoding="utf-8"), str(relative))
    assert not found, (
        "the oracle is quarantined; only hoopstate.validate may name it:\n" + "\n".join(found)
    )


# --- proof the guard is not vacuous -----------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "from hoopstate.storage import Zone\np = profile.zone(Zone.ORACLE)\n", id="enum"
        ),
        pytest.param("import hoopstate.validate.oracle\n", id="import"),
        pytest.param("from hoopstate.validate.oracle import ORACLE_SCHEMA\n", id="from-import"),
        pytest.param("from hoopstate import validate\n", id="package-import"),
        pytest.param('p = root / "oracle" / "pbpstats_2023.parquet"\n', id="path-segment"),
        pytest.param('SOURCE = "pbpstats"\n', id="source-name"),
    ],
)
def test_the_guard_fires(source: str) -> None:
    """Each way a module could reach the oracle is actually detected.

    Without this the guard above could pass forever by detecting nothing.
    """
    assert _violations(source, "synthetic.py")


def test_docstrings_are_not_violations() -> None:
    """Prose about the quarantine is how the rule gets taught; it must pass."""
    source = (
        '"""Compares derived possessions against the pbpstats oracle."""\n'
        "\n"
        "def f():\n"
        '    """Reads nothing from the oracle zone."""\n'
        "    return 1\n"
    )
    assert _violations(source, "synthetic.py") == []


def test_a_non_docstring_string_with_the_same_text_is_a_violation() -> None:
    """Docstrings are exempt by position, not by content."""
    source = 'x = "Compares derived possessions against the pbpstats oracle."\n'
    assert _violations(source, "synthetic.py")


# --- bronze structurally cannot type the oracle -----------------------------


def test_bronze_has_no_schema_for_pbpstats() -> None:
    """Even if the archive were fetched, bronze refuses to type it.

    ``convert_dataset`` types strictly against a declared schema, so a missing
    one means the oracle cannot land in bronze by accident — a second,
    independent barrier to the static guard above.
    """
    with pytest.raises(KeyError):
        schema_for_source("pbpstats")


def test_the_rebuilt_catalog_has_no_view_over_the_oracle(tmp_path: Path) -> None:
    """A third barrier, this one over the DuckDB catalog (``hoops-1lg.1.4``).

    The static guard above stops a module from *naming* the oracle. This checks
    the other end: that the catalog a rebuild actually produces cannot hand the
    answer key to a derivation module as a queryable view — which would put it
    one ``JOIN`` away from the code it is supposed to grade, with no Python
    import to notice.

    The oracle parquet is planted here on purpose. A catalog that published
    whatever it found on disk would pass this test only by luck.
    """
    profile = StorageProfile(name=Profile.EPHEMERAL, root=tmp_path)

    oracle_zone = profile.zone(Zone.ORACLE)
    oracle_zone.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"STARTTYPE": ["Off Steal"]}).write_parquet(oracle_zone / "pbpstats_2023.parquet")

    bronze = profile.zone(Zone.BRONZE, season=2023)
    bronze.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"GAME_ID": ["22300001"]}).write_parquet(bronze / "nbastats_2023.parquet")

    result = rebuild(profile)
    assert result.created, "the bronze fixture must produce at least one view to compare against"

    connection = duckdb.connect(str(result.database), read_only=True)
    try:
        schemas = {
            row[0]
            for row in connection.execute("SELECT schema_name FROM duckdb_schemas()").fetchall()
        }
        views = connection.execute(
            "SELECT schema_name, view_name, sql FROM duckdb_views() WHERE NOT internal"
        ).fetchall()
    finally:
        connection.close()

    assert Zone.ORACLE.value not in schemas
    for schema, view, sql in views:
        assert Zone.ORACLE.value not in schema
        assert Zone.ORACLE.value not in view
        # The view's SQL carries the parquet path, so this also catches a view
        # in another schema reading out of the quarantined directory.
        assert Zone.ORACLE.value not in sql.lower()
        assert _FORBIDDEN_SUBSTRING not in sql.lower()
