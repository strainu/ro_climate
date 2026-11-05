"""High-level client for the ANM cube: discovery, Zarr access, and points."""

from __future__ import annotations

import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import zarr

from . import grid
from .cache import RoClimateCache, CacheKey, CachedPoint
from .exceptions import (
    CacheMissError,
    NoDataError,
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
        self._store = RequestsStore(
            self.store_url,
            self._session,
            timeout=self.timeout,
            retries=retries,
        )

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
        """Return a table describing cached point archives."""
        if self._cache is None:
            return pd.DataFrame()
        return self._cache.info()

    def clear_cache(self) -> int:
        """Delete every cached archive and return the number removed."""
        if self._cache is None:
            return 0
        return self._cache.clear()

    def prune_cache(self, older_than_days: int = 30) -> int:
        """Delete cached archives older than ``older_than_days``."""
        if self._cache is None:
            return 0
        return self._cache.prune(older_than_days)

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
