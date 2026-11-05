"""Pure climate-index helpers.

``RoClimatePoint`` exposes these as methods; the functions here are
intentionally free of point or cache state so they can be tested
directly.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def max_true_run(mask: Iterable[bool]) -> int:
    """Return the length of the longest consecutive ``True`` run."""

    arr = np.asarray(list(mask), dtype=bool)
    if arr.size == 0:
        return 0
    padded = np.concatenate(([False], arr, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


def count_threshold(
    series: pd.Series,
    threshold: float,
    *,
    op: str = "ge",
) -> int:
    """Count values above/below a threshold, ignoring NaNs."""

    if op == "ge":
        mask = series >= threshold
    elif op == "gt":
        mask = series > threshold
    elif op == "le":
        mask = series <= threshold
    elif op == "lt":
        mask = series < threshold
    else:
        raise ValueError("op must be one of 'ge', 'gt', 'le', 'lt'")
    return int(mask.sum())


def max_n_day_sum(series: pd.Series, n: int) -> float:
    """Return the maximum rolling ``n``-day sum."""

    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    return float(series.rolling(n, min_periods=n).sum().max())


def growing_degree_days(series: pd.Series, base: float = 10.0) -> float:
    """Sum of positive degree-day exceedances above ``base``."""

    return float((series - float(base)).clip(lower=0).sum(min_count=1))
