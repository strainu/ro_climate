"""Grid and time-index helpers for the ANM daily cube."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd

from .exceptions import InvalidTimeRangeError, OutsideGridError

# Cell edges of the regular 0.01 degree grid.
LON_MIN = 20.13
LON_MAX = 29.85
LAT_MIN = 43.53
LAT_MAX = 48.36

# Pixel centres.
X_MIN = 20.135
Y_MAX = 48.355

STEP = 0.01
N_X = 972
N_Y = 483

TIME_ORIGIN = pd.Timestamp("1961-01-01")
TIME_STEPS = 32872
TIME_END = pd.Timestamp("2050-12-31")

SEASONS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

DateLike = str | date | datetime | np.datetime64 | pd.Timestamp


def normalize_date(value: DateLike) -> pd.Timestamp:
    """Convert a date-like object to a naive, day-normalised Timestamp."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def date_to_index(value: DateLike) -> int:
    """Return the integer day index since 1961-01-01."""

    return int((normalize_date(value) - TIME_ORIGIN).days)


def index_to_date(index: int) -> pd.Timestamp:
    """Return the date for an integer day index since 1961-01-01."""

    return TIME_ORIGIN + pd.Timedelta(days=int(index))


def lonlat_to_index(lon: float, lat: float) -> tuple[int, int]:
    """Map a coordinate to the containing grid cell.

    The returned ``(i, j)`` indices are the column and row of the cell whose
    edges contain the coordinate. Coordinates exactly on the outer eastern or
    southern edge are clamped to the last valid cell.
    """

    lon = float(lon)
    lat = float(lat)
    if not np.isfinite(lon) or not np.isfinite(lat):
        raise OutsideGridError("longitude and latitude must be finite numbers")
    if not (LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX):
        raise OutsideGridError(
            f"coordinate ({lon}, {lat}) is outside the ANM grid "
            f"[{LON_MIN}, {LAT_MIN}, {LON_MAX}, {LAT_MAX}]"
        )

    i = int(np.floor((lon - LON_MIN) / STEP))
    j = int(np.floor((LAT_MAX - lat) / STEP))
    return min(max(i, 0), N_X - 1), min(max(j, 0), N_Y - 1)


def index_to_lonlat(i: int, j: int) -> tuple[float, float]:
    """Return the pixel-centre coordinate for grid indices ``(i, j)``."""

    return X_MIN + STEP * int(i), Y_MAX - STEP * int(j)


def cell_bounds(i: int, j: int) -> tuple[float, float, float, float]:
    """Return ``(lon_min, lon_max, lat_min, lat_max)`` for a grid cell."""

    lon0 = LON_MIN + STEP * int(i)
    lat1 = LAT_MAX - STEP * int(j)
    return lon0, lon0 + STEP, lat1 - STEP, lat1


def month_numbers(
    months: int | Iterable[int] | None = None,
    season: str | None = None,
) -> tuple[int, ...] | None:
    """Resolve a month list or season name to month numbers."""

    if months is not None and season is not None:
        raise InvalidTimeRangeError("pass either months or season, not both")

    if season is not None:
        key = str(season).upper()
        if key not in SEASONS:
            raise InvalidTimeRangeError(
                f"unknown season {season!r}; expected one of {sorted(SEASONS)}"
            )
        return SEASONS[key]

    if months is None:
        return None
    if isinstance(months, int):
        result = (months,)
    else:
        result = tuple(int(m) for m in months)
    if not result:
        raise InvalidTimeRangeError("months must not be empty")
    bad = [m for m in result if not 1 <= m <= 12]
    if bad:
        raise InvalidTimeRangeError(f"invalid month numbers: {bad}")
    return tuple(dict.fromkeys(result))


def season_year_index(index: pd.DatetimeIndex) -> pd.Index:
    """Return a year index where December belongs to the next winter."""

    years = index.year.to_numpy(copy=True)
    years[index.month == 12] += 1
    return pd.Index(years, name="season_year")
