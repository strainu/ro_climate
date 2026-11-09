"""Official ANM station observations from the ODP daily archive.

The module exposes a packaged catalog of the 160 stations whose daily climate
series are published by the ANM Open Data Portal (ODP), helpers to search that
catalog, and a loader that returns a :class:`RoClimateStation` time series.

Quick start::

    from ro_climate import load_station_series

    series = load_station_series(
        "15014", start="2020-01-01", end="2020-12-31"
    )
    print(series.mean())
    print(series.sum("precipitation"))
    series.to_csv("baia_mare_2020.csv")

The ODP CSV files use ``-999`` for missing values and ``-1`` for trace
precipitation (less than 0.05 mm). ``-999`` is converted to ``NaN``; trace days
are stored as ``0.0`` so that sums stay correct and are flagged in
:attr:`RoClimateStation.precipitation_trace`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from . import grid
from ._chunk_cache import default_chunk_cache_dir
from ._reduce import (
    VALID_HOW as _VALID_HOW,
    extreme_dates as _extreme_dates,
    extreme_index as _extreme_index,
    reduce_grouped as _reduce_grouped,
    reduce_series as _reduce_series,
)
from .archive import write_archive
from .exceptions import (
    InvalidTimeRangeError,
    NoDataError,
    NonAdditiveVariableError,
    StationNotFoundError,
    StoreUnavailableError,
    UnknownVariableError,
)
from .indices import max_true_run
from .variables import (
    PRECIPITATION,
    STATION_ATTRIBUTION,
    STATION_COLUMNS,
    STATION_SOURCE_COLUMNS,
    Variable,
    resolve_station_variables,
)

ODP_BASE_URL = "https://odp.meteoromania.ro/station_data_series/climate/daily"
"""Root of the ODP daily station archive."""

DEFAULT_START = pd.Timestamp("1961-01-01")
"""Default first day of a station series."""

DEFAULT_STATION_CACHE_TTL = 3600.0
"""Default current-year station-archive refresh interval, in seconds."""

STATION_CACHE_TTL_ENV = "RO_CLIMATE_STATION_CACHE_TTL"
"""Environment variable overriding the current-year cache TTL."""

_MISSING_VALUE = -999.0
_TRACE_PRECIPITATION = -1.0
_EARTH_RADIUS_KM = 6371.0088
_CATALOG_PATH = Path(__file__).with_name("_stations.json")

__all__ = [
    "ODP_BASE_URL",
    "Station",
    "RoClimateStation",
    "clear_station_cache",
    "default_station_cache_dir",
    "get_station",
    "list_stations",
    "load_station_series",
    "nearest_station",
    "nearest_stations",
]


# ---------------------------------------------------------------------------
# Cache directory helpers
# ---------------------------------------------------------------------------
def default_station_cache_dir() -> Path:
    """Return the default directory for cached ODP station archives."""

    return default_chunk_cache_dir() / "stations"


def resolve_station_cache_ttl(value: float | None = None) -> float:
    """Resolve the current-year station-cache TTL in seconds.

    Precedence is an explicit ``value``, then
    ``RO_CLIMATE_STATION_CACHE_TTL``, then
    :data:`DEFAULT_STATION_CACHE_TTL`.
    """

    if value is None:
        raw = os.environ.get(STATION_CACHE_TTL_ENV)
        value = DEFAULT_STATION_CACHE_TTL if raw is None else float(raw)
    ttl = float(value)
    if ttl < 0:
        raise ValueError("station cache TTL must be non-negative")
    return ttl


def clear_station_cache(cache_dir: str | Path | None = None) -> int:
    """Delete cached ODP station archives and return the number removed."""

    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else default_station_cache_dir()
    )
    if not root.exists():
        return 0
    count = sum(1 for path in root.rglob("*") if path.is_file())
    shutil.rmtree(root, ignore_errors=True)
    return count


# ---------------------------------------------------------------------------
# Station catalog
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Station:
    """One ANM station in the packaged ODP catalog."""

    wmo: str
    name: str
    latitude: float
    longitude: float
    kind: str = "unknown"

    @property
    def id(self) -> str:
        """Return the WIGOS/WMO station identifier."""

        return self.wmo

    @property
    def label(self) -> str:
        """Return ``"Name (WMO)"`` for display."""

        return f"{self.name} ({self.wmo})"

    def distance_km(self, lon: float, lat: float) -> float:
        """Return the great-circle distance from the station to a point."""

        return _distance_km(self.longitude, self.latitude, lon, lat)

    def __repr__(self) -> str:
        """Return a compact representation of the station."""

        return (
            f"Station(wmo={self.wmo!r}, name={self.name!r}, "
            f"lat={self.latitude:.4f}, lon={self.longitude:.4f}, "
            f"kind={self.kind!r})"
        )


def _normalize_name(value: str) -> str:
    """Normalise a name for accent- and punctuation-insensitive lookup."""

    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_text = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).strip()


@lru_cache(maxsize=1)
def _load_catalog() -> tuple[Station, ...]:
    """Load and validate the packaged station catalog."""

    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    stations: list[Station] = []
    for item in raw:
        stations.append(
            Station(
                wmo=str(item["wmo"]).zfill(5),
                name=str(item["name"]),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                kind=str(item.get("kind", "unknown")),
            )
        )
    return tuple(stations)


@lru_cache(maxsize=1)
def _catalog_by_wmo() -> dict[str, Station]:
    """Return the station catalog keyed by WMO identifier."""

    return {station.wmo: station for station in _load_catalog()}


@lru_cache(maxsize=1)
def _catalog_by_name() -> dict[str, Station]:
    """Return the station catalog keyed by normalised station name."""

    return {
        _normalize_name(station.name): station for station in _load_catalog()
    }


def list_stations(kind: str | None = None) -> list[Station]:
    """Return the packaged station catalog, optionally filtered by kind."""

    stations = list(_load_catalog())
    if kind is not None:
        key = str(kind).strip().casefold()
        stations = [
            station for station in stations if station.kind.casefold() == key
        ]
    return stations


def get_station(identifier: str | int | Station) -> Station:
    """Return a station by WMO identifier or name.

    Numeric identifiers are zero-padded to five digits. Name lookup ignores
    case, accents, hyphens, and other punctuation.
    """

    if isinstance(identifier, Station):
        return identifier
    text = str(identifier).strip()
    if not text:
        raise StationNotFoundError("station identifier must not be empty")
    if text.isdigit():
        station = _catalog_by_wmo().get(text.zfill(5))
        if station is not None:
            return station
    else:
        station = _catalog_by_name().get(_normalize_name(text))
        if station is not None:
            return station
    raise StationNotFoundError(f"unknown ANM station {identifier!r}")


def _distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return the great-circle distance between two points in kilometres."""

    lon1r, lat1r, lon2r, lat2r = (
        np.radians(float(value)) for value in (lon1, lat1, lon2, lat2)
    )
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    haversine = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    return float(
        2.0
        * _EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    )


def nearest_stations(
    lon: float,
    lat: float,
    n: int = 5,
    *,
    max_distance_km: float | None = None,
    kind: str | None = None,
) -> list[tuple[Station, float]]:
    """Return the nearest stations to a coordinate.

    The result is a list of ``(station, distance_km)`` tuples sorted by
    distance. ``lon`` and ``lat`` are WGS84 decimal degrees.
    """

    lon = float(lon)
    lat = float(lat)
    if not np.isfinite(lon) or not np.isfinite(lat):
        raise ValueError("longitude and latitude must be finite numbers")
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    if max_distance_km is not None and float(max_distance_km) < 0:
        raise ValueError("max_distance_km must be non-negative")

    ranked: list[tuple[Station, float]] = []
    for station in list_stations(kind=kind):
        distance = station.distance_km(lon, lat)
        if max_distance_km is not None and distance > float(max_distance_km):
            continue
        ranked.append((station, distance))
    ranked.sort(key=lambda item: (item[1], item[0].wmo))
    return ranked[:n]


def nearest_station(
    lon: float,
    lat: float,
    *,
    max_distance_km: float | None = None,
    kind: str | None = None,
) -> Station:
    """Return the nearest station to a coordinate."""

    matches = nearest_stations(
        lon, lat, 1, max_distance_km=max_distance_km, kind=kind
    )
    if not matches:
        raise StationNotFoundError("no station matches the requested filters")
    return matches[0][0]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class _StationCache:
    """Local cache for ODP year archives and missing-year markers."""

    def __init__(self, directory: str | Path, *, ttl: float) -> None:
        """Create a cache rooted at ``directory``."""

        self.directory = Path(directory).expanduser()
        self.ttl = float(ttl)

    def zip_path(self, wmo: str, year: int) -> Path:
        """Return the cache path for one station-year archive."""

        return (
            self.directory / str(wmo).zfill(5) / f"{int(year)}.zip"
        )

    def missing_path(self, wmo: str, year: int) -> Path:
        """Return the path of the missing-year marker."""

        return (
            self.directory / str(wmo).zfill(5) / f"{int(year)}.missing"
        )

    @staticmethod
    def _age_seconds(path: Path) -> float:
        """Return the age of ``path`` in seconds."""

        now = datetime.now(timezone.utc).timestamp()
        return max(0.0, now - path.stat().st_mtime)

    def is_fresh(self, path: Path, year: int) -> bool:
        """Return whether a cached entry is still valid."""

        if int(year) < datetime.now(timezone.utc).year:
            return True
        try:
            return self._age_seconds(path) <= self.ttl
        except OSError:
            return False

    def read_zip(
        self, wmo: str, year: int, *, ignore_ttl: bool = False
    ) -> bytes | None:
        """Return cached ZIP bytes, or ``None`` when absent or stale."""

        path = self.zip_path(wmo, year)
        try:
            if not path.is_file():
                return None
            if not ignore_ttl and not self.is_fresh(path, year):
                return None
            return path.read_bytes()
        except OSError:
            return None

    def write_zip(self, wmo: str, year: int, data: bytes) -> Path:
        """Atomically store ZIP bytes and clear any missing marker."""

        path = self.zip_path(wmo, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        try:
            self.missing_path(wmo, year).unlink()
        except OSError:
            pass
        return path

    def is_missing(self, wmo: str, year: int) -> bool:
        """Return whether a year is known to be unavailable."""

        path = self.missing_path(wmo, year)
        try:
            if not path.is_file():
                return False
            if int(year) < datetime.now(timezone.utc).year - 1:
                return True
            return self._age_seconds(path) <= self.ttl
        except OSError:
            return False

    def mark_missing(self, wmo: str, year: int) -> None:
        """Record that a station-year archive is unavailable."""

        path = self.missing_path(wmo, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.touch(exist_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ODP downloader
# ---------------------------------------------------------------------------
class _ODPSource:
    """Download and cache ODP station-year archives."""

    def __init__(
        self,
        cache: _StationCache,
        *,
        session: requests.Session | None = None,
        timeout: float = 60.0,
        refresh: bool = False,
    ) -> None:
        """Create a source using ``cache`` and an optional HTTP session."""

        self.cache = cache
        self.timeout = float(timeout)
        self.refresh = bool(refresh)
        self._session = session
        self._owns_session = session is None
        self.warnings: list[str] = []

    def __enter__(self) -> "_ODPSource":
        """Return the source for use as a context manager."""

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the owned HTTP session."""

        self.close()

    def close(self) -> None:
        """Close the HTTP session if this object created it."""

        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def _session_obj(self) -> requests.Session:
        """Return the HTTP session, creating one on first use."""

        if self._session is None:
            self._session = requests.Session()
            self._session.headers.setdefault("User-Agent", "ro-climate/0.2")
            self._owns_session = True
        return self._session

    @staticmethod
    def station_url(wmo: str, year: int) -> str:
        """Return the ODP archive URL for one station-year."""

        wmo = str(wmo).zfill(5)
        return (
            f"{ODP_BASE_URL}/0-20000-0-{wmo}/"
            f"0-20000-0-{wmo}_{int(year)}.zip"
        )

    def fetch_year(self, wmo: str, year: int) -> bytes | None:
        """Return ZIP bytes for ``year``, or ``None`` when unavailable."""

        if not self.refresh:
            cached = self.cache.read_zip(wmo, year)
            if cached is not None:
                return cached
            if self.cache.is_missing(wmo, year):
                return None

        url = self.station_url(wmo, year)
        try:
            response = self._session_obj().get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            stale = self.cache.read_zip(wmo, year, ignore_ttl=True)
            if stale is not None:
                self.warnings.append(
                    f"using cached station archive for {wmo}/{year} "
                    f"after download error: {exc}"
                )
                return stale
            raise StoreUnavailableError(
                f"cannot download station archive {url}: {exc}"
            ) from exc

        if response.status_code == 404:
            stale = self.cache.read_zip(wmo, year, ignore_ttl=True)
            if stale is not None:
                self.warnings.append(
                    f"using cached station archive for {wmo}/{year} "
                    "after a 404 response"
                )
                return stale
            self.cache.mark_missing(wmo, year)
            return None

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            stale = self.cache.read_zip(wmo, year, ignore_ttl=True)
            if stale is not None:
                self.warnings.append(
                    f"using cached station archive for {wmo}/{year} "
                    f"after HTTP {response.status_code}"
                )
                return stale
            raise StoreUnavailableError(
                f"cannot download station archive {url}: HTTP "
                f"{response.status_code}"
            ) from exc

        payload = response.content
        if not zipfile.is_zipfile(io.BytesIO(payload)):
            stale = self.cache.read_zip(wmo, year, ignore_ttl=True)
            if stale is not None:
                self.warnings.append(
                    f"using cached station archive for {wmo}/{year} "
                    "after an invalid response"
                )
                return stale
            raise StoreUnavailableError(
                f"station archive {url} is not a valid ZIP file"
            )
        self.cache.write_zip(wmo, year, payload)
        return payload


# ---------------------------------------------------------------------------
# ODP CSV parsing
# ---------------------------------------------------------------------------
def _decode_csv(raw: bytes) -> str:
    """Decode an ODP CSV file, falling back to Windows-1250."""

    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1250")


def _csv_header_index(lines: list[str]) -> int | None:
    """Return the index of the ODP CSV header line, if present."""

    for index, line in enumerate(lines):
        if line.strip().casefold().startswith("wsi,"):
            return index
    return None


def _parse_month_csv(raw: bytes, columns: tuple[str, ...]) -> pd.DataFrame:
    """Parse one ODP monthly CSV into a date-indexed frame."""

    public_columns = [STATION_COLUMNS[column] for column in columns]
    empty = pd.DataFrame(columns=public_columns)
    lines = _decode_csv(raw).splitlines()
    header_index = _csv_header_index(lines)
    if header_index is None:
        return empty

    try:
        frame = pd.read_csv(
            io.StringIO("\n".join(lines[header_index:])),
            dtype=str,
            keep_default_na=False,
        )
    except (pd.errors.ParserError, ValueError):
        return empty
    frame.columns = [str(column).strip() for column in frame.columns]
    if not {"an", "luna", "zi"}.issubset(frame.columns):
        return empty

    dates = pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(frame["an"], errors="coerce"),
                "month": pd.to_numeric(frame["luna"], errors="coerce"),
                "day": pd.to_numeric(frame["zi"], errors="coerce"),
            }
        ),
        errors="coerce",
    )
    valid = dates.notna()
    if not valid.any():
        return empty

    frame = frame.loc[valid]
    dates = dates.loc[valid]
    out = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    for source in columns:
        if source in frame.columns:
            values = pd.to_numeric(frame[source], errors="coerce")
        else:
            values = pd.Series(np.nan, index=frame.index, dtype="float64")
        values = values.where(values != _MISSING_VALUE)
        out[STATION_COLUMNS[source]] = values.to_numpy(dtype="float32")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _parse_year_zip(payload: bytes, columns: tuple[str, ...]) -> pd.DataFrame:
    """Parse an ODP year ZIP into one date-indexed frame."""

    public_columns = [STATION_COLUMNS[column] for column in columns]
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, OSError) as exc:
        raise StoreUnavailableError(
            "station archive is not a valid ZIP file"
        ) from exc

    frames: list[pd.DataFrame] = []
    with archive:
        for name in sorted(archive.namelist()):
            if not name.casefold().endswith(".csv"):
                continue
            try:
                raw = archive.read(name)
            except (KeyError, OSError, zipfile.BadZipFile):
                continue
            frame = _parse_month_csv(raw, columns)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=public_columns)
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result


# ---------------------------------------------------------------------------
# Station time series
# ---------------------------------------------------------------------------
class RoClimateStation:
    """Daily observations for one ANM station.

    The object is created by :func:`load_station_series`; it is normally not
    instantiated directly.
    """

    def __init__(
        self,
        *,
        station: Station,
        data: pd.DataFrame,
        variables: Iterable[Variable],
        frontiers: dict[str, pd.Timestamp],
        requested_start: pd.Timestamp | None = None,
        requested_end: pd.Timestamp | None = None,
        warnings: Iterable[str] = (),
        retrieved_at: str | datetime | None = None,
        source_url: str = ODP_BASE_URL,
        trace: pd.Series | None = None,
    ) -> None:
        """Create a station series from date-indexed observation data."""

        if data.empty:
            raise NoDataError("station data are empty")
        self.station = station
        self.data = data
        self.variables = tuple(variables)
        self.frontiers = dict(frontiers)
        self.requested_start = requested_start
        self.requested_end = requested_end
        self.warnings = tuple(warnings)
        self.retrieved_at = retrieved_at
        self.source_url = str(source_url).rstrip("/")
        self._trace = trace
        self.start = pd.Timestamp(data.index.min())
        self.end = pd.Timestamp(data.index.max())

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------
    @property
    def wmo(self) -> str:
        """Return the WMO station identifier."""

        return self.station.wmo

    @property
    def name(self) -> str:
        """Return the station name."""

        return self.station.name

    @property
    def latitude(self) -> float:
        """Return the station latitude in WGS84 degrees."""

        return self.station.latitude

    @property
    def longitude(self) -> float:
        """Return the station longitude in WGS84 degrees."""

        return self.station.longitude

    @property
    def kind(self) -> str:
        """Return the station kind from the catalog."""

        return self.station.kind

    @property
    def lon(self) -> float:
        """Return the station longitude, mirroring ``RoClimatePoint``."""

        return self.station.longitude

    @property
    def lat(self) -> float:
        """Return the station latitude, mirroring ``RoClimatePoint``."""

        return self.station.latitude

    @property
    def units(self) -> dict[str, str]:
        """Return the unit of each loaded variable."""

        return {variable.name: variable.units for variable in self.variables}

    @property
    def attribution(self) -> str:
        """Return the data attribution text."""

        return STATION_ATTRIBUTION

    @property
    def is_covered(self) -> bool:
        """Return whether the series contains at least one valid value."""

        return bool(self.data.notna().any().any())

    @property
    def coverage(self) -> pd.DataFrame:
        """Return valid/missing counts and fractions per variable."""

        names = [variable.name for variable in self.variables]
        frame = self.data[names]
        return pd.DataFrame(
            {
                "n_total": len(frame),
                "n_valid": frame.notna().sum(),
                "n_missing": frame.isna().sum(),
                "valid_fraction": frame.notna().mean(),
            }
        )

    @property
    def precipitation_trace(self) -> pd.Series:
        """Return a boolean series marking days with trace precipitation.

        ODP encodes trace amounts (less than 0.05 mm) as ``-1``. The loaded
        precipitation column stores them as ``0.0`` so that sums remain
        correct; this mask preserves the distinction.
        """

        self._require("precipitation")
        if self._trace is None:
            result = pd.Series(False, index=self.data.index)
        else:
            result = self._trace.reindex(
                self.data.index, fill_value=False
            ).astype(bool)
        result.name = "precipitation_trace"
        return result

    def __len__(self) -> int:
        """Return the number of loaded days."""

        return len(self.data)

    def __getitem__(self, variable: str | Iterable[str]):
        """Return one variable as a Series or several as a DataFrame."""

        if isinstance(variable, str):
            self._require(variable)
            return self.data[variable]
        names = [
            item.name if isinstance(item, Variable) else str(item)
            for item in variable
        ]
        for name in names:
            self._require(name)
        return self.data[names]

    def __repr__(self) -> str:
        """Return a compact summary of the station and its date range."""

        names = ",".join(variable.name for variable in self.variables)
        return (
            f"RoClimateStation(wmo={self.wmo}, name={self.name!r}, "
            f"lat={self.latitude:.4f}, lon={self.longitude:.4f}, "
            f"{self.start.date()}..{self.end.date()}, "
            f"{len(self)} days, vars=[{names}])"
        )

    def at(self, date: grid.DateLike) -> pd.Series:
        """Return all loaded variables for one day."""

        ts = grid.normalize_date(date)
        try:
            return self.data.loc[ts]
        except KeyError as exc:
            raise NoDataError(
                f"{ts.date()} is not in the loaded range"
            ) from exc

    def subset(
        self,
        start: grid.DateLike | None = None,
        end: grid.DateLike | None = None,
        variables: Iterable[str] | str | None = None,
        months: int | Iterable[int] | None = None,
        season: str | None = None,
    ) -> "RoClimateStation":
        """Return a new series filtered by date, variable, month, or season.

        ``months`` and ``season`` are mutually exclusive; ``season`` accepts
        ``"DJF"``, ``"MAM"``, ``"JJA"``, or ``"SON"``.
        """

        selected = self._resolve(variables)
        names = [variable.name for variable in selected]
        start_ts = (
            grid.normalize_date(start) if start is not None else self.start
        )
        end_ts = grid.normalize_date(end) if end is not None else self.end
        if start_ts > end_ts:
            raise InvalidTimeRangeError(
                f"start {start_ts.date()} is after end {end_ts.date()}"
            )
        frame = self.data.loc[start_ts:end_ts, names]
        months_filter = grid.month_numbers(months=months, season=season)
        if months_filter is not None:
            frame = frame[frame.index.month.isin(months_filter)]
        if frame.empty:
            raise NoDataError(
                f"no data between {start_ts.date()} and {end_ts.date()}"
                + (
                    f" for months {months_filter}"
                    if months_filter
                    else ""
                )
            )
        return RoClimateStation(
            station=self.station,
            data=frame.copy(),
            variables=selected,
            frontiers=self.frontiers,
            requested_start=start_ts,
            requested_end=end_ts,
            warnings=self.warnings,
            retrieved_at=self.retrieved_at,
            source_url=self.source_url,
            trace=self._trace,
        )

    # ------------------------------------------------------------------
    # Core statistics
    # ------------------------------------------------------------------
    def mean(self, variables=None, *, skipna: bool = True):
        """Return the mean of each selected variable."""

        selected = self._resolve(variables)
        result = self.data[[v.name for v in selected]].mean(skipna=skipna)
        return self._result(result, selected)

    def avg(self, variables=None, *, skipna: bool = True):
        """Alias for :meth:`mean`."""

        return self.mean(variables, skipna=skipna)

    def max(self, variables=None, *, skipna: bool = True):
        """Return the maximum of each selected variable."""

        selected = self._resolve(variables)
        result = self.data[[v.name for v in selected]].max(skipna=skipna)
        return self._result(result, selected)

    def min(self, variables=None, *, skipna: bool = True):
        """Return the minimum of each selected variable."""

        selected = self._resolve(variables)
        result = self.data[[v.name for v in selected]].min(skipna=skipna)
        return self._result(result, selected)

    def median(self, variables=None, *, skipna: bool = True):
        """Return the median of each selected variable."""

        selected = self._resolve(variables)
        result = self.data[[v.name for v in selected]].median(
            skipna=skipna
        )
        return self._result(result, selected)

    def std(self, variables=None, *, skipna: bool = True, ddof: int = 1):
        """Return the standard deviation of each selected variable."""

        selected = self._resolve(variables)
        result = self.data[[v.name for v in selected]].std(
            skipna=skipna, ddof=ddof
        )
        return self._result(result, selected)

    def var(self, variables=None, *, skipna: bool = True, ddof: int = 1):
        """Return the variance of each selected variable."""

        selected = self._resolve(variables)
        result = self.data[[v.name for v in selected]].var(
            skipna=skipna, ddof=ddof
        )
        return self._result(result, selected)

    def quantile(self, q, variables=None):
        """Return the requested quantile of each selected variable."""

        selected = self._resolve(variables)
        result = self.data[[v.name for v in selected]].quantile(q)
        return self._result(result, selected)

    def count(self, variables=None):
        """Return the number of non-missing values per selected variable."""

        selected = self._resolve(variables)
        result = self.data[[v.name for v in selected]].count()
        return self._result(result, selected, integer=True)

    def sum(
        self,
        variables=None,
        *,
        skipna: bool = True,
        min_count: int = 1,
        allow_non_additive: bool = False,
    ):
        """Return the sum of additive variables.

        With no ``variables``, precipitation is used. Temperature variables
        raise :class:`NonAdditiveVariableError` unless
        ``allow_non_additive=True`` is passed.
        """

        if variables is None:
            self._require(PRECIPITATION.name)
            selected = (PRECIPITATION,)
        else:
            selected = self._resolve(variables)
        bad = [
            variable.name
            for variable in selected
            if not variable.additive
        ]
        if bad and not allow_non_additive:
            raise NonAdditiveVariableError(
                "sum is only meaningful for additive variables "
                f"({', '.join(bad)} cannot be summed); "
                "use mean/max/min or pass allow_non_additive=True"
            )
        result = self.data[[v.name for v in selected]].sum(
            skipna=skipna, min_count=int(min_count)
        )
        return self._result(result, selected)

    def describe(self, variables=None) -> pd.DataFrame:
        """Return a per-variable summary table."""

        selected = self._resolve(variables)
        names = [variable.name for variable in selected]
        desc = self.data[names].describe().T
        desc["missing"] = self.data[names].isna().sum()
        desc["sum"] = np.nan
        for variable in selected:
            if variable.additive:
                desc.loc[variable.name, "sum"] = self.data[
                    variable.name
                ].sum(min_count=1)
        desc.index.name = "variable"
        return desc

    # ------------------------------------------------------------------
    # Time aggregation
    # ------------------------------------------------------------------
    def resample(
        self,
        freq: str,
        how: str = "mean",
        variables=None,
        *,
        min_count: int | None = 1,
    ) -> pd.DataFrame:
        """Resample the loaded data to a new pandas frequency."""

        how = str(how).lower()
        if how not in _VALID_HOW:
            raise ValueError(
                f"unsupported resampling method {how!r}; expected one of "
                f"{sorted(_VALID_HOW)}"
            )
        selected = self._resolve(variables)
        bad = [
            variable.name
            for variable in selected
            if how == "sum" and not variable.additive
        ]
        if bad:
            raise NonAdditiveVariableError(
                f"cannot sum {', '.join(bad)}; use mean/max/min"
            )
        frame = self.data[[v.name for v in selected]]
        grouped = frame.resample(freq)
        if how == "count":
            return grouped.count()
        if how == "sum":
            return grouped.sum(
                min_count=0 if min_count is None else int(min_count)
            )
        result = grouped.agg(how)
        if min_count is not None:
            result = result.where(grouped.count() >= int(min_count))
        return result

    def monthly_mean(self, variables=None) -> pd.DataFrame:
        """Return monthly means with a month-start index."""

        return self.resample("MS", "mean", variables)

    def monthly_total(self, variables="precipitation") -> pd.DataFrame:
        """Return monthly totals, precipitation by default."""

        return self.resample("MS", "sum", variables)

    def annual_mean(self, variables=None) -> pd.DataFrame:
        """Return annual means with a year-start index."""

        return self.resample("YS", "mean", variables)

    def annual_total(self, variables="precipitation") -> pd.DataFrame:
        """Return annual totals, precipitation by default."""

        return self.resample("YS", "sum", variables)

    def annual_extremes(self, variables=None) -> pd.DataFrame:
        """Return yearly maxima and minima with ``_max``/``_min`` suffixes."""

        selected = self._resolve(variables)
        names = [variable.name for variable in selected]
        highest = self.resample("YS", "max", names).add_suffix("_max")
        lowest = self.resample("YS", "min", names).add_suffix("_min")
        return highest.join(lowest)

    def extreme_dates(self, variables=None) -> pd.DataFrame:
        """Return minima, maxima, and their dates for each variable."""

        selected = self._resolve(variables)
        frame = self.data[[variable.name for variable in selected]]
        result = _extreme_dates(frame)
        result.index.name = "variable"
        return result

    def annual_mean_temperature(self) -> pd.Series:
        """Return annual means of the daily mean temperature."""

        result = self.tmean.resample("YS").mean()
        result.name = "tmean"
        return result

    def monthly_mean_temperature(self) -> pd.Series:
        """Return climatological monthly means of mean temperature."""

        values = self.tmean
        result = values.groupby(values.index.month).mean()
        result.index.name = "month"
        result.name = "tmean"
        return result

    def warmest_month(self) -> tuple[int, float]:
        """Return the warmest calendar month and its mean temperature."""

        month, value = _extreme_index(
            self.monthly_mean_temperature(), "max"
        )
        return int(month), value

    def coldest_month(self) -> tuple[int, float]:
        """Return the coldest calendar month and its mean temperature."""

        month, value = _extreme_index(
            self.monthly_mean_temperature(), "min"
        )
        return int(month), value

    def wettest_year(self) -> tuple[int, float]:
        """Return the wettest year and its annual precipitation total."""

        year, value = _extreme_index(self._annual_precipitation(), "max")
        return int(pd.Timestamp(year).year), value

    def driest_year(self) -> tuple[int, float]:
        """Return the driest year and its annual precipitation total."""

        year, value = _extreme_index(self._annual_precipitation(), "min")
        return int(pd.Timestamp(year).year), value

    def precipitation_summary(self) -> dict:
        """Return mean annual precipitation and the wettest/driest years."""

        annual = self._annual_precipitation()
        wettest_year, wettest_value = _extreme_index(annual, "max")
        driest_year, driest_value = _extreme_index(annual, "min")
        return {
            "mean_annual": float(annual.mean()),
            "wettest": (
                int(pd.Timestamp(wettest_year).year),
                wettest_value,
            ),
            "driest": (
                int(pd.Timestamp(driest_year).year),
                driest_value,
            ),
        }

    def climatology(self, freq: str = "month") -> pd.DataFrame:
        """Return mean values grouped by calendar month or day of year."""

        selected = self._resolve()
        names = [variable.name for variable in selected]
        freq = freq.lower()
        if freq in {"month", "monthly"}:
            key = self.data.index.month
            index_name = "month"
        elif freq in {"dayofyear", "doy"}:
            key = self.data.index.day_of_year
            index_name = "dayofyear"
        else:
            raise ValueError("freq must be 'month' or 'dayofyear'")
        result = self.data[names].groupby(key).mean()
        result.index.name = index_name
        return result

    # ------------------------------------------------------------------
    # Derived variables
    # ------------------------------------------------------------------
    @property
    def tmean(self) -> pd.Series:
        """Return the daily mean temperature.

        The ODP ``ta`` observation is used when ``mean_temp`` is loaded;
        otherwise the value is ``(max_temp + min_temp) / 2``.
        """

        if "mean_temp" in self.data.columns:
            result = self.data["mean_temp"].copy()
        else:
            self._require("max_temp", "min_temp")
            result = (
                self.data["max_temp"] + self.data["min_temp"]
            ) / 2.0
        result.name = "tmean"
        result.attrs["units"] = "°C"
        return result

    @property
    def diurnal_range(self) -> pd.Series:
        """Return the daily temperature range, ``max_temp - min_temp``."""

        self._require("max_temp", "min_temp")
        result = self.data["max_temp"] - self.data["min_temp"]
        result.name = "diurnal_range"
        result.attrs["units"] = "°C"
        return result

    def mean_temperature(self) -> float:
        """Return the mean of the daily mean temperature series."""

        return float(self.tmean.mean())

    def mean_diurnal_range(self) -> float:
        """Return the mean daily temperature range."""

        return float(self.diurnal_range.mean())

    # ------------------------------------------------------------------
    # Climate indices
    # ------------------------------------------------------------------
    def _count_mask(self, mask: pd.Series, by: str | None):
        """Count true values overall or grouped by a time key."""

        if by is None:
            return int(mask.sum())
        key = str(by).lower()
        if key == "year":
            result = mask.groupby(mask.index.year).sum()
            result.index.name = "year"
        elif key == "month":
            result = mask.groupby(mask.index.month).sum()
            result.index.name = "month"
        elif key == "season_year":
            result = mask.groupby(
                grid.season_year_index(mask.index)
            ).sum()
        else:
            result = mask.astype(int).resample(by).sum()
        return result.astype(int)

    def frost_days(self, threshold: float = 0.0, *, by: str | None = None):
        """Return days with ``min_temp`` below ``threshold``."""

        self._require("min_temp")
        return self._count_mask(self.data["min_temp"] < threshold, by)

    def summer_days(
        self, threshold: float = 25.0, *, by: str | None = None
    ):
        """Return days with ``max_temp`` at or above ``threshold``."""

        self._require("max_temp")
        return self._count_mask(self.data["max_temp"] >= threshold, by)

    def hot_days(self, threshold: float = 30.0, *, by: str | None = None):
        """Return days with ``max_temp`` at or above ``threshold``."""

        self._require("max_temp")
        return self._count_mask(self.data["max_temp"] >= threshold, by)

    def tropical_nights(
        self, threshold: float = 20.0, *, by: str | None = None
    ):
        """Return nights with ``min_temp`` at or above ``threshold``."""

        self._require("min_temp")
        return self._count_mask(self.data["min_temp"] >= threshold, by)

    def rain_days(self, threshold: float = 1.0, *, by: str | None = None):
        """Return days with precipitation at or above ``threshold``."""

        self._require("precipitation")
        return self._count_mask(
            self.data["precipitation"] >= threshold, by
        )

    def heavy_rain_days(
        self, threshold: float = 10.0, *, by: str | None = None
    ):
        """Return days with precipitation at or above ``threshold``."""

        self._require("precipitation")
        return self._count_mask(
            self.data["precipitation"] >= threshold, by
        )

    def dry_days(self, threshold: float = 1.0, *, by: str | None = None):
        """Return days with precipitation below ``threshold``."""

        self._require("precipitation")
        return self._count_mask(
            self.data["precipitation"] < threshold, by
        )

    def max_daily_precip(self) -> float:
        """Return the largest daily precipitation value."""

        self._require("precipitation")
        return float(self.data["precipitation"].max())

    def max_n_day_precip(self, n: int = 5) -> float:
        """Return the largest rolling ``n``-day precipitation total."""

        self._require("precipitation")
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        return float(
            self.data["precipitation"]
            .rolling(n, min_periods=n)
            .sum()
            .max()
        )

    def growing_degree_days(
        self, base: float = 10.0, *, method: str = "mean"
    ) -> float:
        """Return accumulated degree days above ``base``."""

        method = method.lower()
        if method == "mean":
            series = self.tmean
        elif method == "max":
            self._require("max_temp")
            series = self.data["max_temp"]
        elif method == "min":
            self._require("min_temp")
            series = self.data["min_temp"]
        else:
            raise ValueError("method must be 'mean', 'max' or 'min'")
        return float(
            (series - float(base)).clip(lower=0).sum(min_count=1)
        )

    def consecutive_dry_days(self, threshold: float = 1.0) -> int:
        """Return the longest run of days below the precipitation threshold."""

        self._require("precipitation")
        return max_true_run(self.data["precipitation"] < threshold)

    def consecutive_wet_days(self, threshold: float = 1.0) -> int:
        """Return the longest run of days at or above the threshold."""

        self._require("precipitation")
        return max_true_run(self.data["precipitation"] >= threshold)

    # ------------------------------------------------------------------
    # Filtered aggregate API
    # ------------------------------------------------------------------
    def aggregate(
        self,
        variable: str,
        how: str = "mean",
        *,
        start: grid.DateLike | None = None,
        end: grid.DateLike | None = None,
        months: int | Iterable[int] | None = None,
        season: str | None = None,
        groupby: str | None = None,
        min_count: int | None = 1,
    ):
        """Filter and aggregate one variable in a single call."""

        how = str(how).lower()
        if how not in _VALID_HOW:
            raise ValueError(f"unsupported aggregation {how!r}")
        var = resolve_station_variables(variable)[0]
        if how == "sum" and not var.additive:
            raise NonAdditiveVariableError(
                f"cannot sum {var.name}; use mean/max/min"
            )
        subset = self.subset(
            start=start,
            end=end,
            variables=[var.name],
            months=months,
            season=season,
        )
        series = subset.data[var.name]
        if groupby is None:
            return _reduce_series(series, how, min_count)

        key = str(groupby).lower()
        if key == "year":
            groups = series.groupby(series.index.year)
        elif key == "month":
            groups = series.groupby(series.index.month)
        elif key == "season_year":
            groups = series.groupby(
                grid.season_year_index(series.index)
            )
        else:
            groups = series.resample(groupby)
        return _reduce_grouped(groups, how, min_count)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path, **kwargs) -> Path:
        """Write the loaded data to CSV and return the path."""

        path = Path(path)
        kwargs.setdefault("index_label", "date")
        kwargs.setdefault("float_format", "%.4f")
        self.data.to_csv(path, **kwargs)
        return path

    def to_json(self, path: str | Path, **kwargs) -> Path:
        """Write the loaded data to JSON and return the path."""

        path = Path(path)
        kwargs.setdefault("orient", "split")
        kwargs.setdefault("date_format", "iso")
        self.data.to_json(path, **kwargs)
        return path

    def to_npz(self, path: str | Path) -> Path:
        """Write dates, values, and column names to a compressed NPZ file."""

        path = Path(path)
        np.savez_compressed(
            path,
            dates=self.data.index.values.astype("datetime64[D]"),
            values=self.data.to_numpy(dtype="float32"),
            columns=np.asarray(
                [str(column) for column in self.data.columns], dtype="U"
            ),
        )
        return path

    def to_zip(self, path: str | Path) -> Path:
        """Write a zip archive and return its path."""

        return write_archive(Path(path), self.data, self.metadata())

    def metadata(self) -> dict:
        """Return JSON-serialisable provenance, station, and range metadata."""

        retrieved = self.retrieved_at
        if isinstance(retrieved, datetime):
            retrieved = retrieved.astimezone(timezone.utc).isoformat()
        elif isinstance(retrieved, pd.Timestamp):
            retrieved = retrieved.isoformat()
        if not retrieved:
            retrieved = datetime.now(timezone.utc).isoformat()
        return {
            "format_version": 1,
            "source": "ANM ODP station observations",
            "source_url": self.source_url,
            "station": {
                "wmo": self.station.wmo,
                "name": self.station.name,
                "latitude": self.station.latitude,
                "longitude": self.station.longitude,
                "kind": self.station.kind,
            },
            "requested": {
                "start": (
                    self.requested_start.strftime("%Y-%m-%d")
                    if self.requested_start is not None
                    else None
                ),
                "end": (
                    self.requested_end.strftime("%Y-%m-%d")
                    if self.requested_end is not None
                    else None
                ),
            },
            "actual": {
                "start": self.start.strftime("%Y-%m-%d"),
                "end": self.end.strftime("%Y-%m-%d"),
                "n_days": len(self.data),
            },
            "frontiers": {
                name: (
                    None
                    if pd.isna(value)
                    else pd.Timestamp(value).strftime("%Y-%m-%d")
                )
                for name, value in self.frontiers.items()
            },
            "units": self.units,
            "dtype": "float32",
            "column_order": [
                str(column) for column in self.data.columns
            ],
            "retrieved_at": retrieved,
            "license": "CC-BY-4.0",
            "attribution": STATION_ATTRIBUTION,
            "sentinel_values": {
                "missing": _MISSING_VALUE,
                "trace_precipitation": _TRACE_PRECIPITATION,
                "trace_precipitation_stored_as": 0.0,
            },
            "trace_precipitation_days": (
                int(self.precipitation_trace.sum())
                if "precipitation" in self.data.columns
                else None
            ),
            "warnings": list(self.warnings),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _annual_precipitation(self) -> pd.Series:
        """Return annual precipitation totals without empty years."""

        self._require("precipitation")
        annual = self.annual_total("precipitation")["precipitation"]
        return annual.dropna()

    def _resolve(self, variables=None) -> tuple[Variable, ...]:
        """Validate requested variables and ensure they are loaded."""

        selected = (
            resolve_station_variables(variables)
            if variables is not None
            else self.variables
        )
        missing = [
            variable.name
            for variable in selected
            if variable.name not in self.data.columns
        ]
        if missing:
            raise UnknownVariableError(
                "variable(s) not loaded for this station: "
                f"{', '.join(missing)}"
            )
        return tuple(selected)

    def _require(self, *names: str) -> None:
        """Raise :class:`UnknownVariableError` if a variable is missing."""

        missing = [
            name for name in names if name not in self.data.columns
        ]
        if missing:
            raise UnknownVariableError(
                "variable(s) not loaded for this station: "
                f"{', '.join(missing)}"
            )

    @staticmethod
    def _result(
        values: pd.Series,
        variables: tuple[Variable, ...],
        *,
        integer: bool = False,
    ):
        """Convert a pandas result to a scalar or Series with unit metadata."""

        if len(variables) == 1:
            value = values.iloc[0]
            if integer:
                return int(value)
            return float(value)
        if integer:
            values = values.astype(int)
        values.attrs["units"] = {
            variable.name: variable.units for variable in variables
        }
        return values


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_station_series(
    station: str | int | Station,
    start: grid.DateLike | None = None,
    end: grid.DateLike | None = None,
    variables: Iterable[str] | str | None = None,
    *,
    cache_dir: str | Path | None = None,
    cache_ttl: float | None = None,
    refresh: bool = False,
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> RoClimateStation:
    """Load daily observations for one ANM station from ODP.

    Parameters
    ----------
    station:
        WMO identifier, station name, or :class:`Station`.
    start:
        First requested day. Defaults to 1961-01-01.
    end:
        Last requested day. Defaults to today.
    variables:
        Public names (``max_temp``) or ODP codes (``tx``). Defaults to
        ``max_temp``, ``min_temp`` and ``precipitation``.
    cache_dir:
        Station-cache root. Defaults to ``<cache root>/stations``.
    cache_ttl:
        Seconds before a cached current-year archive is refreshed. Defaults
        to the ``RO_CLIMATE_STATION_CACHE_TTL`` environment variable, or one
        hour.
    refresh:
        Ignore cached archives and fetch them again.
    timeout:
        HTTP timeout in seconds.
    session:
        Optional ``requests.Session``; the caller retains ownership.
    """

    selected = resolve_station_variables(variables)
    station_obj = get_station(station)
    today = pd.Timestamp.now().normalize()
    start_ts = DEFAULT_START if start is None else grid.normalize_date(start)
    end_ts = today if end is None else grid.normalize_date(end)
    warnings_list: list[str] = []

    if start_ts > end_ts:
        raise InvalidTimeRangeError(
            f"start {start_ts.date()} is after end {end_ts.date()}"
        )
    if end_ts > today:
        warnings_list.append(
            f"end {end_ts.date()} is in the future; "
            f"clamped to {today.date()}"
        )
        end_ts = today
    if start_ts > today:
        raise NoDataError(
            f"requested start {start_ts.date()} is after today"
        )

    ttl = resolve_station_cache_ttl(cache_ttl)
    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else default_station_cache_dir()
    )
    cache = _StationCache(root, ttl=ttl)
    source_columns = tuple(
        STATION_SOURCE_COLUMNS[variable.name] for variable in selected
    )

    frames: list[pd.DataFrame] = []
    with _ODPSource(
        cache,
        session=session,
        timeout=timeout,
        refresh=refresh,
    ) as source:
        for year in range(start_ts.year, end_ts.year + 1):
            try:
                payload = source.fetch_year(station_obj.wmo, year)
            except StoreUnavailableError as exc:
                warnings_list.append(str(exc))
                continue
            if payload is None:
                continue
            try:
                frame = _parse_year_zip(payload, source_columns)
            except StoreUnavailableError as exc:
                warnings_list.append(
                    "cannot parse station archive for "
                    f"{station_obj.wmo}/{year}: {exc}"
                )
                continue
            if not frame.empty:
                frames.append(frame)
        warnings_list.extend(source.warnings)

    if not frames:
        detail = (
            f" ({'; '.join(warnings_list)})" if warnings_list else ""
        )
        raise NoDataError(
            f"no ODP data for station {station_obj.wmo} between "
            f"{start_ts.date()} and {end_ts.date()}{detail}"
        )

    data = pd.concat(frames).sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.loc[
        (data.index >= start_ts) & (data.index <= end_ts)
    ]
    if data.empty:
        raise NoDataError(
            f"no ODP data for station {station_obj.wmo} between "
            f"{start_ts.date()} and {end_ts.date()}"
        )

    valid_any = data.notna().any(axis=1)
    if not valid_any.any():
        raise NoDataError(
            f"no valid ODP observations for station {station_obj.wmo} "
            f"between {start_ts.date()} and {end_ts.date()}"
        )
    first = data.index[valid_any][0]
    last = data.index[valid_any][-1]
    data = data.loc[first:last]
    data = data.reindex(
        pd.date_range(first, last, freq="D", name="date")
    )
    data = data[[variable.name for variable in selected]].astype(
        "float32"
    )

    trace: pd.Series | None = None
    if "precipitation" in data.columns:
        trace = data["precipitation"].eq(_TRACE_PRECIPITATION)
        if bool(trace.any()):
            data.loc[trace, "precipitation"] = 0.0

    frontiers: dict[str, pd.Timestamp] = {}
    for variable in selected:
        last_valid = data[variable.name].last_valid_index()
        frontiers[variable.name] = (
            pd.Timestamp(last_valid)
            if last_valid is not None
            else pd.NaT
        )
        if last_valid is None:
            warnings_list.append(
                f"no valid {variable.name} observations in the "
                "requested range"
            )

    return RoClimateStation(
        station=station_obj,
        data=data,
        variables=selected,
        frontiers=frontiers,
        requested_start=start_ts,
        requested_end=end_ts,
        warnings=warnings_list,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_url=(
            f"{ODP_BASE_URL}/0-20000-0-{station_obj.wmo}/"
        ),
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def _station_dict(station: Station) -> dict[str, Any]:
    """Return a JSON-serialisable representation of a station."""

    return {
        "wmo": station.wmo,
        "name": station.name,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "kind": station.kind,
    }


def _format_station(
    station: Station, *, distance_km: float | None = None
) -> str:
    """Return one station as a fixed-width table row."""

    row = (
        f"{station.wmo:<6} {station.name:<32} "
        f"{station.latitude:>9.4f} {station.longitude:>10.4f} "
        f"{station.kind}"
    )
    if distance_km is not None:
        row += f" {distance_km:>9.3f}"
    return row


def main(argv: list[str] | None = None) -> int:
    """Run the station-catalog command-line interface."""

    parser = argparse.ArgumentParser(
        prog="python -m ro_climate.stations",
        description=(
            "List ANM station observations available through the ODP."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list", action="store_true", help="list all stations"
    )
    group.add_argument(
        "--near",
        nargs=2,
        metavar=("LON", "LAT"),
        type=float,
        help="list stations nearest to a longitude/latitude point",
    )
    parser.add_argument(
        "--kind", help="restrict results to a station kind"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="maximum number of nearby stations (default: 5)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON"
    )
    args = parser.parse_args(argv)

    if args.list:
        stations = list_stations(kind=args.kind)
        if args.json:
            print(
                json.dumps(
                    [_station_dict(station) for station in stations],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(
                f"{'WMO':<6} {'Name':<32} {'Latitude':>9} "
                f"{'Longitude':>10} Kind"
            )
            for station in stations:
                print(_format_station(station))
        return 0

    lon, lat = args.near
    ranked = nearest_stations(
        lon, lat, args.limit, kind=args.kind
    )
    if args.json:
        payload = [
            {
                **_station_dict(station),
                "distance_km": round(distance, 3),
            }
            for station, distance in ranked
        ]
        print(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
    else:
        print(
            f"{'WMO':<6} {'Name':<32} {'Latitude':>9} "
            f"{'Longitude':>10} Kind Distance"
        )
        for station, distance in ranked:
            print(
                _format_station(station, distance_km=distance)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
