"""High-level client for the ANM cube: discovery, Zarr access, and points."""

from __future__ import annotations

import hashlib
import struct
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import zarr

from . import grid
from ._chunk_cache import CachingRequestsStore, ChunkCache
from .area import (
    RoClimateArea,
    column_name,
    resolve_reductions,
)
from .cache import (
    AreaCacheKey,
    CacheKey,
    CachedPoint,
    RoClimateCache,
    default_cache_dir,
)
from .exceptions import (
    CacheMissError,
    NoDataError,
    NonAdditiveVariableError,
    StoreUnavailableError,
)
from ._store import RequestsStore
from .point import RoClimatePoint
from .variables import (
    ATTRIBUTION,
    DEFAULT_VARIABLES,
    Variable,
    resolve_variables,
)

DEFAULT_STORE = "https://storage.svc.uvt-01.eocube.ro/meteo/cube/open-v1/"
ITEM_URL = (
    "https://stac.eocube.ro/collections/anm-daily-grids/items/"
    "anm-daily-grids-open-v1"
)


def _dedupe(items: Iterable[str]) -> list[str]:
    """Return unique items, preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _time_blocks(array, t0: int, t1: int):
    """Yield chunk-aligned, clipped ``(start, stop)`` time slices."""

    try:
        chunk = int(array.chunks[0])
    except (AttributeError, IndexError, TypeError):
        chunk = t1 - t0 + 1
    if chunk <= 0:
        chunk = t1 - t0 + 1
    start = (t0 // chunk) * chunk
    while start <= t1:
        yield max(start, t0), min(start + chunk, t1 + 1)
        start += chunk


def _spatial_chunks(
    array,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    mask: np.ndarray | None,
):
    """Yield chunk windows containing at least one selected cell.

    Each item is ``(j_start, j_stop, i_start, i_stop, local_mask)``. A local
    mask of ``None`` means the whole chunk window is selected.
    """

    try:
        chunk_y = int(array.chunks[1])
        chunk_x = int(array.chunks[2])
    except (AttributeError, IndexError, TypeError):
        chunk_y = j1 - j0 + 1
        chunk_x = i1 - i0 + 1
    if chunk_y <= 0:
        chunk_y = j1 - j0 + 1
    if chunk_x <= 0:
        chunk_x = i1 - i0 + 1

    for chunk_j in range(j0 // chunk_y, j1 // chunk_y + 1):
        j_start = max(j0, chunk_j * chunk_y)
        j_stop = min(j1 + 1, (chunk_j + 1) * chunk_y)
        for chunk_i in range(i0 // chunk_x, i1 // chunk_x + 1):
            i_start = max(i0, chunk_i * chunk_x)
            i_stop = min(i1 + 1, (chunk_i + 1) * chunk_x)
            if mask is None:
                local_mask = None
            else:
                local_mask = mask[
                    j_start - j0:j_stop - j0,
                    i_start - i0:i_stop - i0,
                ]
                if not local_mask.any():
                    continue
            yield j_start, j_stop, i_start, i_stop, local_mask


def _read_area_variable(
    group,
    variable: Variable,
    t0: int,
    t1: int,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    mask: np.ndarray | None,
    reductions: tuple[str, ...],
    min_count: int | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Read one variable chunk-by-chunk and return daily spatial summaries."""

    array = group[variable.name]
    n_days = t1 - t0 + 1
    count_total = np.zeros(n_days, dtype=np.int64)
    need_sum = "mean" in reductions or "sum" in reductions
    sum_total = (
        np.zeros(n_days, dtype=np.float64) if need_sum else None
    )
    min_total = (
        np.full(n_days, np.nan, dtype=np.float64)
        if "min" in reductions
        else None
    )
    max_total = (
        np.full(n_days, np.nan, dtype=np.float64)
        if "max" in reductions
        else None
    )

    for j_start, j_stop, i_start, i_stop, local_mask in _spatial_chunks(
        array, i0, i1, j0, j1, mask
    ):
        for start, stop in _time_blocks(array, t0, t1):
            try:
                slab = np.asarray(
                    array[
                        start:stop,
                        j_start:j_stop,
                        i_start:i_stop,
                    ],
                    dtype="float32",
                )
            except KeyError as exc:
                raise NoDataError(
                    f"missing Zarr chunk for {variable.name} at "
                    f"{grid.index_to_date(start).date()}"
                ) from exc
            except Exception as exc:
                raise StoreUnavailableError(
                    f"cannot read {variable.name} for area "
                    f"({i0}:{i1}, {j0}:{j1}): {exc}"
                ) from exc
            if slab.ndim != 3:
                raise StoreUnavailableError(
                    f"unexpected {variable.name} slab shape {slab.shape}"
                )
            if local_mask is None:
                values = slab.reshape(slab.shape[0], -1)
            else:
                values = slab[:, local_mask]
            counts = np.count_nonzero(
                ~np.isnan(values), axis=1
            ).astype(np.int64)
            day_slice = slice(start - t0, stop - t0)
            count_total[day_slice] += counts
            if sum_total is not None:
                sum_total[day_slice] += np.nansum(
                    values, axis=1, dtype=np.float64
                )
            if min_total is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    day_min = np.nanmin(values, axis=1)
                min_total[day_slice] = np.fmin(
                    min_total[day_slice], day_min
                )
            if max_total is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    day_max = np.nanmax(values, axis=1)
                max_total[day_slice] = np.fmax(
                    max_total[day_slice], day_max
                )

    threshold = 1 if min_count is None else max(1, int(min_count))
    valid = count_total >= threshold
    output: dict[str, np.ndarray] = {}
    if "mean" in reductions:
        output["mean"] = np.divide(
            sum_total,
            count_total,
            out=np.full(n_days, np.nan, dtype=np.float64),
            where=valid,
        ).astype("float32")
    if "sum" in reductions:
        output["sum"] = np.where(
            valid, sum_total, np.nan
        ).astype("float32")
    if "min" in reductions:
        output["min"] = np.where(
            valid, min_total, np.nan
        ).astype("float32")
    if "max" in reductions:
        output["max"] = np.where(
            valid, max_total, np.nan
        ).astype("float32")
    if "count" in reductions:
        output["count"] = count_total.astype("float32")
    return output, count_total


class RoClimate:
    """Client for the ANM daily gridded meteorological cube.

    Parameters
    ----------
    store_url:
        Zarr store root. By default the ``zarr-https`` asset is discovered
        from STAC, falling back to the known public URL.
    cache:
        Enable the local zip cache. Default ``True``.
    cache_dir:
        Cache directory. Defaults to the platform cache directory.
    offline:
        Use only the local cache. Missing data raise :class:`CacheMissError`.
    """

    def __init__(
        self,
        store_url: str | None = None,
        *,
        discover: bool = True,
        cache: bool = True,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        chunk_cache: bool = False,
        chunk_cache_max_mb: float | None = None,
        timeout: float = 30.0,
        retries: int = 3,
        frontier_ttl: int = 3600,
    ) -> None:
        """Create a client for the ANM daily gridded cube.

        Parameters
        ----------
        store_url:
            Zarr store root. When omitted, the STAC ``zarr-https`` asset is
            discovered unless ``offline`` is true.
        discover:
            Discover the store URL from STAC when ``store_url`` is omitted.
        cache:
            Enable the local zip cache.
        cache_dir:
            Cache directory. Defaults to the platform cache directory.
        offline:
            Use only the local cache and never open the remote store.
        chunk_cache:
            Cache raw compressed Zarr chunks locally so overlapping point and
            area requests can reuse them.
        chunk_cache_max_mb:
            Optional size limit for the chunk cache in megabytes. Oldest
            chunks are evicted when the limit is exceeded.
        timeout:
            HTTP timeout in seconds.
        retries:
            Number of attempts for each remote read.
        frontier_ttl:
            Seconds to cache variable frontier dates in memory.
        """
        self.timeout = float(timeout)
        self.offline = bool(offline)
        self.frontier_ttl = max(0, int(frontier_ttl))
        self._session = requests.Session()
        self._session.headers.setdefault("User-Agent", "ro-climate/0.1")
        self._cache = RoClimateCache(cache_dir) if cache else None
        self._group = None
        self._frontier_cache: dict[str, pd.Timestamp] | None = None
        self._frontier_time = 0.0

        if store_url is None:
            if discover and not self.offline:
                store_url = self._discover_store()
            else:
                store_url = DEFAULT_STORE
        self.store_url = store_url.rstrip("/") + "/"
        remote_store = RequestsStore(
            self.store_url,
            self._session,
            timeout=self.timeout,
            retries=retries,
        )
        self._chunk_cache = None
        if chunk_cache:
            if cache_dir is not None:
                chunk_root = Path(cache_dir).expanduser()
            else:
                base = default_cache_dir()
                chunk_root = base.parent if base.name == "points" else base
            max_bytes = (
                int(float(chunk_cache_max_mb) * 1_000_000)
                if chunk_cache_max_mb is not None
                else None
            )
            self._chunk_cache = ChunkCache(
                chunk_root / "chunks",
                self.store_url,
                DEFAULT_VARIABLES,
                max_bytes=max_bytes,
            )
            self._store = CachingRequestsStore(
                remote_store, self._chunk_cache
            )
        else:
            self._store = remote_store

    # ------------------------------------------------------------------
    # Discovery and low-level access
    # ------------------------------------------------------------------
    def _discover_store(self) -> str:
        """Return the Zarr store URL from STAC, or the known default URL."""
        try:
            response = self._session.get(ITEM_URL, timeout=self.timeout)
            response.raise_for_status()
            item = response.json()
            href = item["assets"]["zarr-https"]["href"]
            return str(href)
        except (requests.RequestException, KeyError, TypeError, ValueError):
            return DEFAULT_STORE

    def _open_group(self):
        """Open and cache the consolidated Zarr root group."""
        if self._group is None:
            try:
                self._group = zarr.open_consolidated(self._store, mode="r")
            except Exception as exc:  # zarr raises several exception types
                raise StoreUnavailableError(
                    f"cannot open Zarr store {self.store_url}: {exc}"
                ) from exc
        return self._group

    def frontiers(self, *, force: bool = False) -> dict[str, pd.Timestamp]:
        """Return the last written day of each variable.

        The scalar arrays in the store are authoritative; the STAC item's
        ``end_datetime`` can lag behind them.
        """

        now = time.monotonic()
        if (
            not force
            and self._frontier_cache is not None
            and now - self._frontier_time < self.frontier_ttl
        ):
            return dict(self._frontier_cache)

        result: dict[str, pd.Timestamp] = {}
        for name in DEFAULT_VARIABLES:
            key = f"last_{name}_time/0"
            try:
                raw = self._store[key]
                days = struct.unpack("<q", raw)[0]
            except Exception:
                try:
                    group = self._open_group()
                    days = int(group[f"last_{name}_time"][()])
                except Exception as exc:
                    raise StoreUnavailableError(
                        f"cannot read the frontier for {name}: {exc}"
                    ) from exc
            result[name] = grid.index_to_date(days)

        self._frontier_cache = dict(result)
        self._frontier_time = now
        return result

    # ------------------------------------------------------------------
    # Point creation
    # ------------------------------------------------------------------
    def point(
        self,
        lon: float,
        lat: float,
        start: grid.DateLike | None = None,
        end: grid.DateLike | None = None,
        variables: Iterable[str] | str | None = None,
        *,
        cache: bool | None = None,
        refresh: bool | None = None,
        strict: bool = False,
    ) -> RoClimatePoint:
        """Return the daily series for the cell containing ``lon, lat``.

        Parameters
        ----------
        lon, lat:
            Coordinate in WGS84 degrees. The containing 0.01° cell is used.
        start, end:
            Optional inclusive date bounds. ``start`` defaults to 1961-01-01
            and ``end`` defaults to the common frontier of the requested
            variables.
        variables:
            Variable names to load. Defaults to all three variables.
        cache:
            Override the client-level cache setting for this call.
        refresh:
            Ignore an existing archive and fetch fresh data.
        strict:
            Raise :class:`NoDataError` when the cell has no valid values.

        Returns
        -------
        RoClimatePoint
            The loaded point time series.

        Raises
        ------
        OutsideGridError
            If the coordinate is outside the ANM grid.
        NoDataError
            If the requested range has no data.
        CacheMissError
            If offline mode is enabled and no covering archive exists.
        """
        selected = resolve_variables(variables)
        # Keep a stable variable order so the same set shares one cache entry.
        selected = tuple(
            sorted(selected, key=lambda var: DEFAULT_VARIABLES.index(var.name))
        )
        names = tuple(v.name for v in selected)
        i, j = grid.lonlat_to_index(lon, lat)
        grid_lon, grid_lat = grid.index_to_lonlat(i, j)

        cache_enabled = (
            self._cache is not None if cache is None else bool(cache)
        )
        refresh = False if refresh is None else bool(refresh)
        warnings: list[str] = []

        requested_start = (
            grid.normalize_date(start)
            if start is not None
            else grid.TIME_ORIGIN
        )
        if requested_start < grid.TIME_ORIGIN:
            warnings.append(
                f"start clamped from {requested_start.date()} to "
                f"{grid.TIME_ORIGIN.date()}"
            )
            requested_start = grid.TIME_ORIGIN
        if requested_start > grid.TIME_END:
            raise NoDataError(
                f"start {requested_start.date()} is after the cube's last day "
                f"{grid.TIME_END.date()}"
            )

        requested_end = grid.normalize_date(end) if end is not None else None
        key = CacheKey(
            self.store_url, i, j, names, grid_lon, grid_lat
        )

        cached: CachedPoint | None = None
        if cache_enabled and self._cache is not None and not refresh:
            cached = self._cache.load(key)

        # Determine the latest day that can be returned.
        if self.offline:
            if cached is None:
                raise CacheMissError(
                    "offline mode requested, but no cached point exists for "
                    f"({lon}, {lat}) and variables {names}"
                )
            frontiers = {
                name: pd.Timestamp(cached.metadata["frontiers"][name])
                for name in names
            }
            target_end = min(frontiers.values())
            if requested_end is not None:
                target_end = min(requested_end, target_end)
        else:
            all_frontiers = self.frontiers()
            frontiers = {name: all_frontiers[name] for name in names}
            target_end = min(frontiers.values())
            if requested_end is not None:
                target_end = min(requested_end, target_end)

        if requested_start > target_end:
            raise NoDataError(
                f"no data between {requested_start.date()} and "
                f"{target_end.date()}"
            )
        if requested_end is not None and requested_end > target_end:
            warnings.append(
                f"end clamped from {requested_end.date()} to "
                f"{target_end.date()}"
            )

        if cached is not None:
            cstart = pd.Timestamp(cached.metadata["actual"]["start"])
            cend = pd.Timestamp(cached.metadata["actual"]["end"])
            if cstart <= requested_start and cend >= target_end:
                point = self._point_from_cached(
                    cached,
                    lon=float(lon),
                    lat=float(lat),
                    selected=selected,
                    start=requested_start,
                    end=target_end,
                    warnings=warnings,
                    frontiers=frontiers,
                    strict=strict,
                )
                return point

        if self.offline:
            raise CacheMissError(
                "offline mode requested, but the cached point does not cover "
                f"{requested_start.date()}..{target_end.date()}"
            )

        # Fetch the union of the cached range and the requested range.
        fetch_start = requested_start
        fetch_end = target_end
        if cached is not None:
            fetch_start = min(
                fetch_start, pd.Timestamp(cached.metadata["actual"]["start"])
            )
            fetch_end = max(
                fetch_end, pd.Timestamp(cached.metadata["actual"]["end"])
            )
        fetch_start = max(fetch_start, grid.TIME_ORIGIN)
        fetch_end = min(fetch_end, target_end)

        t0 = grid.date_to_index(fetch_start)
        t1 = grid.date_to_index(fetch_end)
        dates = pd.date_range(
            grid.index_to_date(t0), grid.index_to_date(t1), freq="D"
        )

        group = self._open_group()
        if refresh and self._chunk_cache is not None:
            for var in selected:
                try:
                    chunk_size = int(group[var.name].chunks[0])
                except (AttributeError, IndexError, TypeError):
                    chunk_size = t1 - t0 + 1
                self._chunk_cache.invalidate(var.name, t0, t1, chunk_size)
        columns: dict[str, pd.Series] = {}
        for var in selected:
            try:
                values = np.asarray(
                    group[var.name][t0:t1 + 1, j, i], dtype="float32"
                )
            except KeyError as exc:
                raise NoDataError(
                    f"missing Zarr chunk for {var.name} at "
                    f"{fetch_start.date()}"
                ) from exc
            except Exception as exc:
                raise StoreUnavailableError(
                    f"cannot read {var.name} for ({lon}, {lat}): {exc}"
                ) from exc
            columns[var.name] = pd.Series(values, index=dates, name=var.name)

        data = pd.DataFrame(columns, columns=[v.name for v in selected])

        try:
            actual_grid_lon = float(group["x"][i])
            actual_grid_lat = float(group["y"][j])
        except Exception:
            actual_grid_lon, actual_grid_lat = grid_lon, grid_lat

        metadata = {
            "format_version": 1,
            "store_url": self.store_url,
            "item_url": ITEM_URL,
            "requested": {
                "start": requested_start.strftime("%Y-%m-%d"),
                "end": requested_end.strftime("%Y-%m-%d")
                if requested_end is not None
                else None,
                "variables": list(names),
            },
            "grid": {
                "i": i,
                "j": j,
                "lon": actual_grid_lon,
                "lat": actual_grid_lat,
                "cell_bounds": list(grid.cell_bounds(i, j)),
            },
            "actual": {
                "start": fetch_start.strftime("%Y-%m-%d"),
                "end": fetch_end.strftime("%Y-%m-%d"),
                "n_days": len(data),
            },
            "frontiers": {
                name: value.strftime("%Y-%m-%d")
                for name, value in frontiers.items()
            },
            "units": {v.name: v.units for v in selected},
            "dtype": "float32",
            "column_order": [v.name for v in selected],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "license": "CC-BY-4.0",
            "attribution": ATTRIBUTION,
            "warnings": warnings,
        }

        cache_path = None
        if cache_enabled and self._cache is not None:
            cache_path = self._cache.save(key, data, metadata)

        sliced = data.loc[
            requested_start:target_end, [v.name for v in selected]
        ]
        if sliced.empty:
            raise NoDataError(
                f"no data between {requested_start.date()} and "
                f"{target_end.date()}"
            )

        point = RoClimatePoint(
            lon=float(lon),
            lat=float(lat),
            i=i,
            j=j,
            grid_lon=actual_grid_lon,
            grid_lat=actual_grid_lat,
            data=sliced.copy(),
            variables=selected,
            frontiers=frontiers,
            store_url=self.store_url,
            item_url=ITEM_URL,
            warnings=warnings,
            from_cache=False,
            cache_path=cache_path,
            retrieved_at=metadata["retrieved_at"],
        )
        if strict and not point.is_covered:
            raise NoDataError(
                f"the grid cell at ({lon}, {lat}) contains no valid data"
            )
        return point

    # ------------------------------------------------------------------
    # Area loading
    # ------------------------------------------------------------------
    def area(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        start: grid.DateLike | None = None,
        end: grid.DateLike | None = None,
        variables: Iterable[str] | str | None = None,
        reductions: Iterable[str] | str | None = None,
        *,
        polygon: Iterable[tuple[float, float]] | None = None,
        min_count: int | None = 1,
        allow_non_additive: bool = False,
        cache: bool | None = None,
        refresh: bool | None = None,
        strict: bool = False,
    ) -> RoClimateArea:
        """Return daily spatial summaries for a grid window or polygon.

        Parameters
        ----------
        bounds:
            ``(west, south, east, north)`` window in WGS84 degrees. When
            omitted, the polygon's bounding box is used.
        start, end:
            Optional inclusive date bounds. ``start`` defaults to 1961-01-01
            and ``end`` defaults to the common frontier of the requested
            variables.
        variables:
            Variable names to load. Defaults to all three variables.
        reductions:
            Spatial reductions to compute: ``mean``, ``min``, ``max``,
            ``sum``, or ``count``. Defaults to ``("mean",)``.
        polygon:
            Optional simple polygon as ``(lon, lat)`` vertices. Cells whose
            centres lie inside the polygon are selected.
        min_count:
            Minimum number of valid cells required for a non-count reduction.
        allow_non_additive:
            Permit spatial ``sum`` for temperature variables.
        cache:
            Override the client-level cache setting for this call.
        refresh:
            Ignore existing area and chunk caches for the requested range.
        strict:
            Raise :class:`NoDataError` when every summary value is invalid.

        Returns
        -------
        RoClimateArea
            The loaded daily spatial summaries.
        """
        selected = resolve_variables(variables)
        selected = tuple(
            sorted(
                selected,
                key=lambda item: DEFAULT_VARIABLES.index(item.name),
            )
        )
        names = tuple(item.name for item in selected)
        reductions = resolve_reductions(reductions)
        if "sum" in reductions:
            bad = [item.name for item in selected if not item.additive]
            if bad and not allow_non_additive:
                raise NonAdditiveVariableError(
                    "sum is only meaningful for additive variables "
                    f"({', '.join(bad)} cannot be summed); "
                    "use mean/max/min or pass allow_non_additive=True"
                )

        if bounds is None and polygon is None:
            raise ValueError("pass bounds and/or polygon to select an area")
        warnings_list: list[str] = []
        poly = grid.normalize_polygon(polygon) if polygon is not None else None
        if bounds is None:
            west, south, east, north = grid.polygon_bounds(poly)
        else:
            if len(bounds) != 4:
                raise ValueError("bounds must contain four values")
            west, south, east, north = (float(value) for value in bounds)
            if poly is not None:
                poly_west, poly_south, poly_east, poly_north = (
                    grid.polygon_bounds(poly)
                )
                if (
                    poly_west < west
                    or poly_south < south
                    or poly_east > east
                    or poly_north > north
                ):
                    warnings_list.append(
                        "polygon extends outside bounds; cells outside bounds "
                        "were ignored"
                    )

        if (
            west < grid.LON_MIN
            or east > grid.LON_MAX
            or south < grid.LAT_MIN
            or north > grid.LAT_MAX
        ):
            warnings_list.append(
                f"bounds clipped from ({west}, {south}, {east}, {north}) "
                f"to the ANM grid"
            )
        i0, i1, j0, j1 = grid.bbox_to_index(west, south, east, north)
        mask = (
            grid.polygon_mask(poly, i0, i1, j0, j1)
            if poly is not None
            else None
        )
        if mask is not None and not mask.any():
            raise NoDataError("polygon contains no grid cell centres")
        n_cells = (
            int(mask.sum())
            if mask is not None
            else (i1 - i0 + 1) * (j1 - j0 + 1)
        )
        min_count = 1 if min_count is None else int(min_count)
        if min_count < 1:
            raise ValueError("min_count must be at least 1")
        if min_count > n_cells:
            raise ValueError(
                f"min_count {min_count} exceeds the {n_cells} "
                "selected cells"
            )
        mask_digest = (
            hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
            if mask is not None
            else ""
        )

        requested_start = (
            grid.normalize_date(start)
            if start is not None
            else grid.TIME_ORIGIN
        )
        if requested_start < grid.TIME_ORIGIN:
            warnings_list.append(
                f"start clamped from {requested_start.date()} to "
                f"{grid.TIME_ORIGIN.date()}"
            )
            requested_start = grid.TIME_ORIGIN
        if requested_start > grid.TIME_END:
            raise NoDataError(
                f"start {requested_start.date()} is after the cube's last day "
                f"{grid.TIME_END.date()}"
            )
        requested_end = grid.normalize_date(end) if end is not None else None

        requested = {
            "start": requested_start.strftime("%Y-%m-%d"),
            "end": (
                requested_end.strftime("%Y-%m-%d")
                if requested_end is not None
                else None
            ),
            "variables": list(names),
            "reductions": list(reductions),
            "bounds": [west, south, east, north],
            "polygon": (
                [list(point) for point in poly] if poly is not None else None
            ),
            "min_count": min_count,
        }

        cache_enabled = (
            self._cache is not None if cache is None else bool(cache)
        )
        refresh = False if refresh is None else bool(refresh)
        key = AreaCacheKey(
            self.store_url,
            i0,
            i1,
            j0,
            j1,
            names,
            reductions,
            mask_digest,
            min_count,
        )
        cached: CachedPoint | None = None
        if cache_enabled and self._cache is not None and not refresh:
            cached = self._cache.load(key)

        if self.offline:
            if cached is None:
                raise CacheMissError(
                    "offline mode requested, but no cached area exists for "
                    f"bounds {bounds} and variables {names}"
                )
            frontiers = {
                name: pd.Timestamp(cached.metadata["frontiers"][name])
                for name in names
            }
            target_end = min(frontiers.values())
            if requested_end is not None:
                target_end = min(requested_end, target_end)
            if requested_start > target_end:
                raise NoDataError(
                    f"no data between {requested_start.date()} and "
                    f"{target_end.date()}"
                )
            cached_start = pd.Timestamp(cached.metadata["actual"]["start"])
            cached_end = pd.Timestamp(cached.metadata["actual"]["end"])
            if cached_start > requested_start or cached_end < target_end:
                raise CacheMissError(
                    "offline mode requested, but the cached area does not "
                    f"cover {requested_start.date()}..{target_end.date()}"
                )
            return self._area_from_cached(
                cached,
                bounds=(west, south, east, north),
                i0=i0,
                i1=i1,
                j0=j0,
                j1=j1,
                poly=poly,
                selected=selected,
                reductions=reductions,
                n_cells=n_cells,
                start=requested_start,
                end=target_end,
                warnings=warnings_list,
                frontiers=frontiers,
                strict=strict,
            )

        all_frontiers = self.frontiers()
        frontiers = {name: all_frontiers[name] for name in names}
        target_end = min(frontiers.values())
        if requested_end is not None:
            target_end = min(requested_end, target_end)
        if requested_start > target_end:
            raise NoDataError(
                f"no data between {requested_start.date()} and "
                f"{target_end.date()}"
            )
        if requested_end is not None and requested_end > target_end:
            warnings_list.append(
                f"end clamped from {requested_end.date()} to "
                f"{target_end.date()}"
            )

        if cached is not None:
            cached_start = pd.Timestamp(cached.metadata["actual"]["start"])
            cached_end = pd.Timestamp(cached.metadata["actual"]["end"])
            if cached_start <= requested_start and cached_end >= target_end:
                return self._area_from_cached(
                    cached,
                    bounds=(west, south, east, north),
                    i0=i0,
                    i1=i1,
                    j0=j0,
                    j1=j1,
                    poly=poly,
                    selected=selected,
                    reductions=reductions,
                    n_cells=n_cells,
                    start=requested_start,
                    end=target_end,
                    warnings=warnings_list,
                    frontiers=frontiers,
                    strict=strict,
                )

        group = self._open_group()
        segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if cached is None:
            segments.append((requested_start, target_end))
        else:
            cached_start = pd.Timestamp(cached.metadata["actual"]["start"])
            cached_end = pd.Timestamp(cached.metadata["actual"]["end"])
            if requested_start < cached_start:
                segment_end = min(
                    cached_start - pd.Timedelta(days=1), target_end
                )
                if requested_start <= segment_end:
                    segments.append((requested_start, segment_end))
            if target_end > cached_end:
                segment_start = max(
                    cached_end + pd.Timedelta(days=1), requested_start
                )
                if segment_start <= target_end:
                    segments.append((segment_start, target_end))

        frames: list[pd.DataFrame] = []
        new_coverage: dict[str, dict] = {}
        if cached is not None:
            frames.append(cached.data)
            new_coverage.update(cached.metadata.get("coverage", {}))
        for segment_start, segment_end in segments:
            frame, coverage = self._fetch_area_segment(
                group,
                selected,
                reductions,
                segment_start,
                segment_end,
                i0,
                i1,
                j0,
                j1,
                mask,
                n_cells,
                min_count,
                refresh,
            )
            frames.append(frame)
            for name, row in coverage.items():
                if name in new_coverage:
                    old_valid = new_coverage[name].get("n_valid")
                    if old_valid is None:
                        new_coverage[name]["n_valid"] = None
                    else:
                        new_coverage[name]["n_valid"] = (
                            old_valid + row["n_valid"]
                        )
                    new_coverage[name]["n_days"] = None
                    new_coverage[name]["n_cell_days"] = None
                    new_coverage[name]["valid_fraction"] = None
                else:
                    new_coverage[name] = dict(row)

        columns = [
            column_name(item.name, reduction)
            for item in selected
            for reduction in reductions
        ]
        full_data = pd.concat(frames).sort_index()
        full_data = full_data[~full_data.index.duplicated(keep="last")]
        full_data = full_data[columns]

        coverage = self._coverage_frame(
            new_coverage, selected, n_cells, len(full_data)
        )
        retrieved_at = datetime.now(timezone.utc).isoformat()
        full_area = RoClimateArea(
            bounds=(west, south, east, north),
            i0=i0,
            i1=i1,
            j0=j0,
            j1=j1,
            data=full_data,
            variables=selected,
            reductions=reductions,
            frontiers=frontiers,
            store_url=self.store_url,
            item_url=ITEM_URL,
            polygon=poly,
            n_cells=n_cells,
            coverage=coverage,
            warnings=_dedupe(
                list(cached.metadata.get("warnings", []))
                if cached is not None
                else []
            )
            + warnings_list,
            from_cache=False,
            retrieved_at=retrieved_at,
            requested=requested,
        )
        cache_path = None
        if cache_enabled and self._cache is not None:
            cache_path = self._cache.save(
                key, full_data, full_area.metadata()
            )

        sliced = full_data.loc[requested_start:target_end, columns]
        if sliced.empty:
            raise NoDataError(
                f"no data between {requested_start.date()} and "
                f"{target_end.date()}"
            )
        sliced_coverage = (
            coverage if sliced.index.equals(full_data.index) else None
        )
        area = RoClimateArea(
            bounds=(west, south, east, north),
            i0=i0,
            i1=i1,
            j0=j0,
            j1=j1,
            data=sliced.copy(),
            variables=selected,
            reductions=reductions,
            frontiers=frontiers,
            store_url=self.store_url,
            item_url=ITEM_URL,
            polygon=poly,
            n_cells=n_cells,
            coverage=sliced_coverage,
            warnings=full_area.warnings,
            from_cache=False,
            cache_path=cache_path,
            retrieved_at=retrieved_at,
            requested=requested,
        )
        if strict and not area.is_covered:
            raise NoDataError(
                f"the selected area has no valid data between "
                f"{requested_start.date()} and {target_end.date()}"
            )
        return area

    def _fetch_area_segment(
        self,
        group,
        selected: tuple[Variable, ...],
        reductions: tuple[str, ...],
        start: pd.Timestamp,
        end: pd.Timestamp,
        i0: int,
        i1: int,
        j0: int,
        j1: int,
        mask: np.ndarray | None,
        n_cells: int,
        min_count: int | None,
        refresh_chunks: bool,
    ) -> tuple[pd.DataFrame, dict[str, dict]]:
        """Fetch one date segment and return summaries plus coverage rows."""

        t0 = grid.date_to_index(start)
        t1 = grid.date_to_index(end)
        dates = pd.date_range(
            grid.index_to_date(t0), grid.index_to_date(t1), freq="D"
        )
        frame_columns: dict[str, np.ndarray] = {}
        coverage: dict[str, dict] = {}
        for variable in selected:
            if refresh_chunks and self._chunk_cache is not None:
                try:
                    chunk_size = int(group[variable.name].chunks[0])
                except (AttributeError, IndexError, TypeError):
                    chunk_size = t1 - t0 + 1
                self._chunk_cache.invalidate(
                    variable.name, t0, t1, chunk_size
                )
            summaries, counts = _read_area_variable(
                group,
                variable,
                t0,
                t1,
                i0,
                i1,
                j0,
                j1,
                mask,
                reductions,
                min_count,
            )
            for reduction in reductions:
                frame_columns[
                    column_name(variable.name, reduction)
                ] = summaries[reduction]
            n_valid = int(counts.sum())
            n_cell_days = n_cells * len(dates)
            coverage[variable.name] = {
                "variable": variable.name,
                "n_cells": n_cells,
                "n_days": len(dates),
                "n_cell_days": n_cell_days,
                "n_valid": n_valid,
                "valid_fraction": (
                    n_valid / n_cell_days if n_cell_days else np.nan
                ),
            }
        frame = pd.DataFrame(frame_columns, index=dates)
        return frame, coverage

    def _area_from_cached(
        self,
        cached: CachedPoint,
        *,
        bounds: tuple[float, float, float, float],
        i0: int,
        i1: int,
        j0: int,
        j1: int,
        poly: tuple[tuple[float, float], ...] | None,
        selected: tuple[Variable, ...],
        reductions: tuple[str, ...],
        n_cells: int,
        start: pd.Timestamp,
        end: pd.Timestamp,
        warnings: list[str],
        frontiers: dict[str, pd.Timestamp],
        strict: bool,
    ) -> RoClimateArea:
        """Build an area from a cached archive slice."""

        meta = cached.metadata
        columns = [
            column_name(item.name, reduction)
            for item in selected
            for reduction in reductions
        ]
        data = cached.data.loc[start:end, columns]
        if data.empty:
            raise NoDataError(
                f"no cached data between {start.date()} and {end.date()}"
            )
        coverage_rows = meta.get("coverage", {})
        if not data.index.equals(cached.data.index):
            coverage_rows = {}
        coverage = self._coverage_frame(
            coverage_rows, selected, n_cells, len(data)
        )
        area = RoClimateArea(
            bounds=bounds,
            i0=i0,
            i1=i1,
            j0=j0,
            j1=j1,
            data=data.copy(),
            variables=selected,
            reductions=reductions,
            frontiers=frontiers,
            store_url=str(meta.get("store_url", self.store_url)),
            item_url=meta.get("item_url"),
            polygon=poly,
            n_cells=n_cells,
            coverage=coverage,
            warnings=_dedupe(
                list(meta.get("warnings", [])) + list(warnings)
            ),
            from_cache=True,
            cache_path=cached.path,
            retrieved_at=meta.get("retrieved_at"),
            requested=meta.get("requested"),
        )
        if strict and not area.is_covered:
            raise NoDataError(
                f"the selected area has no valid data between "
                f"{start.date()} and {end.date()}"
            )
        return area

    @staticmethod
    def _coverage_frame(
        coverage: dict[str, dict],
        selected: tuple[Variable, ...],
        n_cells: int,
        n_days: int,
    ) -> pd.DataFrame:
        """Return a normalised coverage table for the selected variables."""

        rows = []
        for variable in selected:
            row = coverage.get(variable.name, {})
            n_valid = row.get("n_valid")
            if n_valid is None:
                n_valid = np.nan
            n_cell_days = n_cells * n_days
            rows.append(
                {
                    "variable": variable.name,
                    "n_cells": n_cells,
                    "n_days": n_days,
                    "n_cell_days": n_cell_days,
                    "n_valid": n_valid,
                    "valid_fraction": (
                        n_valid / n_cell_days
                        if n_cell_days and not pd.isna(n_valid)
                        else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows).set_index("variable")

    def _point_from_cached(
        self,
        cached: CachedPoint,
        *,
        lon: float,
        lat: float,
        selected: tuple[Variable, ...],
        start: pd.Timestamp,
        end: pd.Timestamp,
        warnings: list[str],
        strict: bool,
        frontiers: dict[str, pd.Timestamp] | None = None,
    ) -> RoClimatePoint:
        """Build an :class:`RoClimatePoint` from a cached archive slice."""
        meta = cached.metadata
        names = [v.name for v in selected]
        try:
            data = cached.data.loc[start:end, names]
        except KeyError as exc:
            raise CacheMissError(
                "cached point is missing requested variables"
            ) from exc
        if data.empty:
            raise NoDataError(
                f"no cached data between {start.date()} and {end.date()}"
            )
        if frontiers is None:
            frontiers = {
                name: pd.Timestamp(meta["frontiers"][name]) for name in names
            }
        all_warnings = _dedupe(list(meta.get("warnings", [])) + list(warnings))
        point = RoClimatePoint(
            lon=float(lon),
            lat=float(lat),
            i=int(meta["grid"]["i"]),
            j=int(meta["grid"]["j"]),
            grid_lon=float(meta["grid"]["lon"]),
            grid_lat=float(meta["grid"]["lat"]),
            data=data.copy(),
            variables=selected,
            frontiers=frontiers,
            store_url=str(meta.get("store_url", self.store_url)),
            item_url=meta.get("item_url"),
            warnings=all_warnings,
            from_cache=True,
            cache_path=cached.path,
            retrieved_at=meta.get("retrieved_at"),
        )
        if strict and not point.is_covered:
            raise NoDataError(
                f"the grid cell at ({lon}, {lat}) contains no valid data"
            )
        return point

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def cache_info(self) -> pd.DataFrame:
        """Return a table describing cached point and area archives."""
        if self._cache is None:
            return pd.DataFrame()
        return self._cache.info()

    def chunk_cache_info(self) -> pd.DataFrame:
        """Return a per-variable summary of the raw chunk cache."""
        if self._chunk_cache is None:
            return pd.DataFrame(
                columns=[
                    "variable",
                    "chunks",
                    "size_mb",
                    "oldest",
                    "newest",
                ]
            )
        return self._chunk_cache.info()

    def clear_cache(self) -> int:
        """Delete every cached archive and chunk, returning the count."""
        removed = self._cache.clear() if self._cache is not None else 0
        if self._chunk_cache is not None:
            removed += self._chunk_cache.clear()
        return removed

    def prune_cache(self, older_than_days: int = 30) -> int:
        """Delete cached archives and chunks older than ``older_than_days``."""
        removed = (
            self._cache.prune(older_than_days)
            if self._cache is not None
            else 0
        )
        if self._chunk_cache is not None:
            removed += self._chunk_cache.prune(older_than_days)
        return removed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._store.close()
        self._session.close()

    def __enter__(self) -> "RoClimate":
        """Return this client for use in a ``with`` statement."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the client when leaving a ``with`` statement."""
        self.close()
