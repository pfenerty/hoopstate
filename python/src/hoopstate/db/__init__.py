"""DuckDB layer: conventions and the parquet-to-DuckDB rebuild helper.

The DuckDB file is always rebuildable from parquet and safe to delete; see
:mod:`hoopstate.db.rebuild` for the view/materialization conventions and the
rebuild command.

The public names are re-exported lazily. Importing them eagerly here would pull
:mod:`hoopstate.db.rebuild` into ``sys.modules`` at package-import time, which
makes ``python -m hoopstate.db.rebuild`` warn that the module was imported before
being run as ``__main__``. Lazy access via :pep:`562` ``__getattr__`` avoids that
while still allowing ``from hoopstate.db import rebuild_database``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hoopstate.db.rebuild import (
        VIEW_ZONES,
        GoldMarts,
        RebuildResult,
        Relation,
        plan_relations,
        rebuild_database,
    )

__all__ = [
    "VIEW_ZONES",
    "GoldMarts",
    "RebuildResult",
    "Relation",
    "plan_relations",
    "rebuild_database",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from hoopstate.db import rebuild

        return getattr(rebuild, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
