"""Shared temporal reduction helpers."""

from __future__ import annotations

import pandas as pd

VALID_HOW = {"mean", "sum", "min", "max", "median", "std", "var", "count"}


def reduce_series(
    series: pd.Series, how: str, min_count: int | None
) -> float:
    """Reduce one series to a scalar, respecting ``min_count`` valid values."""
    count = int(series.count())
    threshold = 1 if min_count is None else int(min_count)
    if how == "count":
        return count
    if count < threshold:
        return float("nan")
    if how == "sum":
        return float(series.sum(min_count=threshold))
    if how == "mean":
        return float(series.mean())
    if how == "min":
        return float(series.min())
    if how == "max":
        return float(series.max())
    if how == "median":
        return float(series.median())
    if how == "std":
        return float(series.std(ddof=1))
    if how == "var":
        return float(series.var(ddof=1))
    raise ValueError(f"unsupported aggregation {how!r}")


def reduce_grouped(grouped, how: str, min_count: int | None):
    """Apply a named reduction to a pandas groupby or resampler object."""
    if how == "count":
        return grouped.count()
    if how == "sum":
        result = grouped.sum(
            min_count=0 if min_count is None else int(min_count)
        )
    else:
        result = grouped.agg(how)
    if min_count is not None and how != "count":
        counts = grouped.count()
        result = result.where(counts >= int(min_count))
    return result
