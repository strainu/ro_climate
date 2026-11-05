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


def normalize_polygon(
    polygon: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Validate a polygon and return its vertices as ``(lon, lat)`` tuples.

    A duplicated closing vertex is removed. The polygon must contain at least
    three distinct finite vertices. Self-intersecting rings are not validated
    and should be avoided by callers.
    """

    points = [(float(lon), float(lat)) for lon, lat in polygon]
    if len(points) < 3:
        raise ValueError("polygon must contain at least three vertices")
    for lon, lat in points:
        if not np.isfinite(lon) or not np.isfinite(lat):
            raise ValueError("polygon coordinates must be finite")
    if points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError(
            "polygon must contain at least three distinct vertices"
        )
    return tuple(points)


def polygon_bounds(
    polygon: Iterable[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Return ``(west, south, east, north)`` for a polygon."""

    points = normalize_polygon(polygon)
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return min(lons), min(lats), max(lons), max(lats)


def bbox_to_index(
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[int, int, int, int]:
    """Return inclusive ``(i0, i1, j0, j1)`` indices for cell centres.

    The window is clipped to the grid. If it is narrower than one cell, the
    cell containing the window midpoint is returned. A window that does not
    overlap the grid raises :class:`OutsideGridError`.
    """

    values = (west, south, east, north)
    if not all(np.isfinite(float(value)) for value in values):
        raise OutsideGridError("bounds must be finite numbers")
    west, south, east, north = (float(value) for value in values)
    if west >= east or south >= north:
        raise ValueError("bounds must satisfy west < east and south < north")
    if (
        east < LON_MIN
        or west > LON_MAX
        or north < LAT_MIN
        or south > LAT_MAX
    ):
        raise OutsideGridError(
            f"bounds [{west}, {south}, {east}, {north}] do not overlap "
            f"the ANM grid [{LON_MIN}, {LAT_MIN}, {LON_MAX}, {LAT_MAX}]"
        )

    west = max(west, LON_MIN)
    east = min(east, LON_MAX)
    south = max(south, LAT_MIN)
    north = min(north, LAT_MAX)

    # Cell centres are X_MIN + i * STEP and Y_MAX - j * STEP.
    eps = 1e-9
    i0 = int(np.ceil((west - X_MIN) / STEP - eps))
    i1 = int(np.floor((east - X_MIN) / STEP + eps))
    j0 = int(np.ceil((Y_MAX - north) / STEP - eps))
    j1 = int(np.floor((Y_MAX - south) / STEP + eps))

    if i0 > i1 or j0 > j1:
        mid_i, mid_j = lonlat_to_index(
            (west + east) / 2.0, (south + north) / 2.0
        )
        if i0 > i1:
            i0 = i1 = mid_i
        if j0 > j1:
            j0 = j1 = mid_j

    i0 = min(max(i0, 0), N_X - 1)
    i1 = min(max(i1, 0), N_X - 1)
    j0 = min(max(j0, 0), N_Y - 1)
    j1 = min(max(j1, 0), N_Y - 1)
    return i0, i1, j0, j1


def polygon_mask(
    polygon: Iterable[tuple[float, float]],
    i0: int,
    i1: int,
    j0: int,
    j1: int,
) -> np.ndarray:
    """Return a boolean mask of cell centres inside ``polygon``.

    The returned array has shape ``(j1 - j0 + 1, i1 - i0 + 1)``. The polygon
    is treated as a single simple ring; holes and multipart polygons are not
    supported by this helper.
    """

    points = normalize_polygon(polygon)
    if i0 > i1 or j0 > j1:
        raise ValueError("invalid grid window")

    lon = X_MIN + STEP * np.arange(i0, i1 + 1, dtype=float)
    lat = Y_MAX - STEP * np.arange(j0, j1 + 1, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    inside = np.zeros(lon_grid.shape, dtype=bool)

    x1, y1 = points[-1]
    for x2, y2 in points:
        crosses = (y1 > lat_grid) != (y2 > lat_grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (x2 - x1) * (lat_grid - y1) / (y2 - y1) + x1
        inside ^= crosses & (lon_grid < x_cross)
        x1, y1 = x2, y2
    return inside
