"""Point time-series object for the ANM daily cube."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import grid
from .archive import write_archive
from .exceptions import (
    InvalidTimeRangeError,
    NoDataError,
    NonAdditiveVariableError,
    UnknownVariableError,
)
from ._reduce import (
    VALID_HOW as _VALID_HOW,
    extreme_dates as _extreme_dates,
    extreme_index as _extreme_index,
    reduce_grouped as _reduce_grouped,
    reduce_series as _reduce_series,
)
from .indices import max_true_run
from .variables import ATTRIBUTION, PRECIPITATION, Variable, resolve_variables


class RoClimatePoint:
    """Daily time series for one grid cell.

    The object is created by :meth:`ro_climate.RoClimate.point`; it is
    normally not instantiated directly.
    """

    def __init__(
        self,
        *,
        lon: float,
        lat: float,
        i: int,
        j: int,
        grid_lon: float,
        grid_lat: float,
        data: pd.DataFrame,
        variables: Iterable[Variable],
        frontiers: dict[str, pd.Timestamp],
        store_url: str,
        item_url: str | None = None,
        warnings: Iterable[str] = (),
        retrieved_at: str | datetime | None = None,
    ) -> None:
        """Create a point time series.

        This constructor is normally called by
        :meth:`ro_climate.RoClimate.point`; pass ``data`` indexed by date with
        one column per loaded variable.
        """
        if data.empty:
            raise NoDataError("point data are empty")
        self.lon = float(lon)
        self.lat = float(lat)
        self.i = int(i)
        self.j = int(j)
        self.grid_lon = float(grid_lon)
        self.grid_lat = float(grid_lat)
        self.data = data
        self.variables = tuple(variables)
        self.frontiers = dict(frontiers)
        self.store_url = store_url
        self.item_url = item_url
        self.warnings = tuple(warnings)
        self.retrieved_at = retrieved_at
        self.start = pd.Timestamp(data.index.min())
        self.end = pd.Timestamp(data.index.max())

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------
    @property
    def units(self) -> dict[str, str]:
        """Return the unit of each loaded variable."""
        return {v.name: v.units for v in self.variables}

    @property
    def cell_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(lon_min, lon_max, lat_min, lat_max)`` for the cell."""
        return grid.cell_bounds(self.i, self.j)

    @property
    def attribution(self) -> str:
        """Return the data attribution text."""
        return ATTRIBUTION

    @property
    def is_covered(self) -> bool:
        """Return whether the cell contains at least one valid value."""
        return bool(self.data.notna().any().any())

    @property
    def coverage(self) -> pd.DataFrame:
        """Return valid/missing counts and fractions per variable."""
        names = [v.name for v in self.variables]
        frame = self.data[names]
        return pd.DataFrame(
            {
                "n_total": len(frame),
                "n_valid": frame.notna().sum(),
                "n_missing": frame.isna().sum(),
                "valid_fraction": frame.notna().mean(),
            }
        )

    def __len__(self) -> int:
        """Return the number of loaded days."""
        return len(self.data)

    def __getitem__(self, variable: str | Iterable[str]):
        """Return one variable as a Series or several as a DataFrame."""
        if isinstance(variable, str):
            self._require(variable)
            return self.data[variable]
        names = [
            v.name if isinstance(v, Variable) else str(v)
            for v in variable
        ]
        for name in names:
            self._require(name)
        return self.data[names]

    def __repr__(self) -> str:
        """Return a compact summary of the point and its date range."""
        names = ",".join(v.name for v in self.variables)
        return (
            f"RoClimatePoint(lon={self.lon:.4f}, lat={self.lat:.4f}, "
            f"grid=({self.i},{self.j}), "
            f"{self.start.date()}..{self.end.date()}, "
            f"{len(self)} days, vars=[{names}])"
        )

    def at(self, date: grid.DateLike) -> pd.Series:
        """Return all loaded variables for one day.

        Raises :class:`NoDataError` if the date is outside the loaded range.
        """
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
    ) -> "RoClimatePoint":
        """Return a new point filtered by date, variable, month, or season.

        ``months`` and ``season`` are mutually exclusive; ``season`` accepts
        ``"DJF"``, ``"MAM"``, ``"JJA"``, or ``"SON"``. The result is a new
        :class:`RoClimatePoint` sharing the same cell metadata.
        """
        selected = self._resolve(variables)
        names = [v.name for v in selected]
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
                + (f" for months {months_filter}" if months_filter else "")
            )
        return RoClimatePoint(
            lon=self.lon,
            lat=self.lat,
            i=self.i,
            j=self.j,
            grid_lon=self.grid_lon,
            grid_lat=self.grid_lat,
            data=frame.copy(),
            variables=selected,
            frontiers=self.frontiers,
            store_url=self.store_url,
            item_url=self.item_url,
            warnings=self.warnings,
            retrieved_at=self.retrieved_at,
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
        result = self.data[[v.name for v in selected]].median(skipna=skipna)
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
        bad = [v.name for v in selected if not v.additive]
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
        """Return a per-variable summary table.

        The table adds ``missing`` and, for additive variables, ``sum`` to
        pandas' standard descriptive statistics.
        """
        selected = self._resolve(variables)
        names = [v.name for v in selected]
        desc = self.data[names].describe().T
        desc["missing"] = self.data[names].isna().sum()
        desc["sum"] = np.nan
        for var in selected:
            if var.additive:
                desc.loc[var.name, "sum"] = self.data[var.name].sum(
                    min_count=1
                )
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
        """Resample the loaded data to a new pandas frequency.

        ``how`` may be ``mean``, ``sum``, ``min``, ``max``, ``median``,
        ``std``, ``var``, or ``count``. Summing is only allowed for additive
        variables such as precipitation.
        """
        how = str(how).lower()
        if how not in _VALID_HOW:
            raise ValueError(
                f"unsupported resampling method {how!r}; expected one of "
                f"{sorted(_VALID_HOW)}"
            )
        selected = self._resolve(variables)
        bad = [v.name for v in selected if how == "sum" and not v.additive]
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
        names = [v.name for v in selected]
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
        month, value = _extreme_index(self.monthly_mean_temperature(), "max")
        return int(month), value

    def coldest_month(self) -> tuple[int, float]:
        """Return the coldest calendar month and its mean temperature."""
        month, value = _extreme_index(self.monthly_mean_temperature(), "min")
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
        names = [v.name for v in selected]
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
        """Return the daily mean temperature, ``(max_temp + min_temp) / 2``."""
        self._require("max_temp", "min_temp")
        result = (self.data["max_temp"] + self.data["min_temp"]) / 2.0
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
            result = mask.groupby(grid.season_year_index(mask.index)).sum()
        else:
            result = mask.astype(int).resample(by).sum()
        return result.astype(int)

    def frost_days(self, threshold: float = 0.0, *, by: str | None = None):
        """Return days with ``min_temp`` below ``threshold``."""
        self._require("min_temp")
        return self._count_mask(self.data["min_temp"] < threshold, by)

    def summer_days(self, threshold: float = 25.0, *, by: str | None = None):
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
        return self._count_mask(self.data["precipitation"] >= threshold, by)

    def heavy_rain_days(
        self, threshold: float = 10.0, *, by: str | None = None
    ):
        """Return days with precipitation at or above ``threshold``."""
        self._require("precipitation")
        return self._count_mask(self.data["precipitation"] >= threshold, by)

    def dry_days(self, threshold: float = 1.0, *, by: str | None = None):
        """Return days with precipitation below ``threshold``."""
        self._require("precipitation")
        return self._count_mask(self.data["precipitation"] < threshold, by)

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
            self.data["precipitation"].rolling(n, min_periods=n).sum().max()
        )

    def growing_degree_days(
        self, base: float = 10.0, *, method: str = "mean"
    ) -> float:
        """Return accumulated degree days above ``base``.

        ``method`` selects ``mean`` (default), ``max``, or ``min`` as the
        daily temperature series.
        """
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
        return float((series - float(base)).clip(lower=0).sum(min_count=1))

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
        """Filter and aggregate one variable in a single call.

        Date bounds, month/season filters, and ``groupby`` are applied before
        the reduction. ``groupby`` accepts ``"year"``, ``"month"``,
        ``"season_year"``, or a pandas frequency alias. With no ``groupby``,
        a scalar is returned.
        """
        how = str(how).lower()
        if how not in _VALID_HOW:
            raise ValueError(f"unsupported aggregation {how!r}")
        var = resolve_variables(variable)[0]
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
            groups = series.groupby(grid.season_year_index(series.index))
        else:
            groups = series.resample(groupby)
        return _reduce_grouped(groups, how, min_count)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path, **kwargs) -> Path:
        """Write the loaded data to CSV and return the path.

        The index is labelled ``date`` and floats default to four decimals;
        extra keyword arguments are passed to :meth:`pandas.DataFrame.to_csv`.
        """
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
            columns=np.asarray([str(c) for c in self.data.columns], dtype="U"),
        )
        return path

    def to_zip(self, path: str | Path) -> Path:
        """Write a zip archive and return its path."""
        return write_archive(Path(path), self.data, self.metadata())

    def metadata(self) -> dict:
        """Return JSON-serialisable provenance, grid, and range metadata."""
        retrieved = self.retrieved_at
        if isinstance(retrieved, datetime):
            retrieved = retrieved.astimezone(timezone.utc).isoformat()
        elif isinstance(retrieved, pd.Timestamp):
            retrieved = retrieved.isoformat()
        if not retrieved:
            retrieved = datetime.now(timezone.utc).isoformat()
        return {
            "format_version": 1,
            "store_url": self.store_url,
            "item_url": self.item_url,
            "grid": {
                "i": self.i,
                "j": self.j,
                "lon": self.grid_lon,
                "lat": self.grid_lat,
                "cell_bounds": list(self.cell_bounds),
            },
            "actual": {
                "start": self.start.strftime("%Y-%m-%d"),
                "end": self.end.strftime("%Y-%m-%d"),
                "n_days": len(self.data),
            },
            "frontiers": {
                name: pd.Timestamp(value).strftime("%Y-%m-%d")
                for name, value in self.frontiers.items()
            },
            "units": self.units,
            "dtype": "float32",
            "column_order": [str(c) for c in self.data.columns],
            "retrieved_at": retrieved,
            "license": "CC-BY-4.0",
            "attribution": ATTRIBUTION,
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
            resolve_variables(variables)
            if variables is not None
            else self.variables
        )
        missing = [v.name for v in selected if v.name not in self.data.columns]
        if missing:
            raise UnknownVariableError(
                f"variable(s) not loaded for this point: {', '.join(missing)}"
            )
        return tuple(selected)

    def _require(self, *names: str) -> None:
        """Raise :class:`UnknownVariableError` if a variable is missing."""
        missing = [n for n in names if n not in self.data.columns]
        if missing:
            raise UnknownVariableError(
                f"variable(s) not loaded for this point: {', '.join(missing)}"
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
        values.attrs["units"] = {v.name: v.units for v in variables}
        return values
