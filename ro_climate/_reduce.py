"""Shared temporal reduction helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .exceptions import NoDataError

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


def extreme_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return minima, maxima, and their dates for each column.

    Empty or all-NaN columns produce ``NaN`` values and ``NaT`` dates.
    """

    rows: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty:
            rows[column] = {
                "min": float("nan"),
                "min_date": pd.NaT,
                "max": float("nan"),
                "max_date": pd.NaT,
            }
            continue
        min_index = series.idxmin()
        max_index = series.idxmax()
        rows[column] = {
            "min": float(series.loc[min_index]),
            "min_date": pd.Timestamp(min_index),
            "max": float(series.loc[max_index]),
            "max_date": pd.Timestamp(max_index),
        }
    result = pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=["min", "min_date", "max", "max_date"],
    )
    result.index.name = "series"
    return result


def extreme_index(series: pd.Series, how: str) -> tuple[Any, float]:
    """Return the index label and value of a series minimum or maximum."""

    values = series.dropna()
    if values.empty:
        raise NoDataError("no valid values for the requested extreme")
    if how == "max":
        index = values.idxmax()
    elif how == "min":
        index = values.idxmin()
    else:
        raise ValueError(f"unsupported extreme {how!r}; use 'min' or 'max'")
    return index, float(values.loc[index])
