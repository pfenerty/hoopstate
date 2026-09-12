"""Shared rendering helpers for the modules that print numbers.

Small and deliberately internal. Two callers today —
:mod:`hoopstate.ingest.reports` and :mod:`hoopstate.footprint` — and the reason
this exists rather than each keeping a copy is that a table renderer that drifts
between two reports is worse than either version of it.

Nothing here knows what it is rendering.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["human_bytes", "table"]

# Binary units throughout, because every tool this project is checked against
# (``du -h``, ``df -h``, cargo's own output) reports powers of two.
_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")


def human_bytes(count: int) -> str:
    """Render a byte count the way ``du -h`` does.

    Whole bytes below a kibibyte, one decimal place above it — enough to
    distinguish 3.2 GiB from 4.9 GiB, which is the resolution a disk budget is
    argued at, without implying precision the measurement does not have.
    """
    if count < 1024:
        return f"{count} B"
    value = float(count)
    for unit in _UNITS[1:]:
        value /= 1024
        if value < 1024 or unit == _UNITS[-1]:
            return f"{value:.1f} {unit}"
    raise AssertionError("unreachable: the loop returns on the last unit")


def table(
    rows: Sequence[tuple[str, str, str]],
    headers: tuple[str, str, str] = ("metric", "value", "detail"),
) -> list[str]:
    """Render three-column rows as an aligned table.

    Values are right-aligned so magnitudes line up and an outlier is visible at
    a glance. The detail column is dropped entirely when no row uses it.

    ``headers`` is overridable because the columns are only nominally metrics:
    the footprint report puts consumers and budgets through the same renderer,
    and a table whose header lies about its contents is worse than no header.
    """
    has_detail = any(detail for _, _, detail in rows)
    if not has_detail:
        headers = (headers[0], headers[1], "")
    columns = 3 if has_detail else 2
    width = [
        max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(columns)
    ]
    lines = [
        f"  {headers[0]:<{width[0]}}  {headers[1]:>{width[1]}}"
        + (f"  {headers[2]}" if has_detail else ""),
        "  " + "  ".join("-" * w for w in width),
    ]
    for metric, value, detail in rows:
        line = f"  {metric:<{width[0]}}  {value:>{width[1]}}"
        if has_detail:
            line += f"  {detail}"
        lines.append(line.rstrip())
    return lines
