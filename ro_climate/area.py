"""Area summaries for the ANM daily gridded cube."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import grid
from ._reduce import (
    VALID_HOW,
    extreme_dates as _extreme_dates,
    extreme_index as _extreme_index,
    reduce_grouped,
    reduce_series,
)
from .archive import write_archive
from .exceptions import (
    InvalidTimeRangeError,
    NoDataError,
    NonAdditiveVariableError,
    UnknownVariableError,
)
from .variables import ATTRIBUTION, Variable, resolve_variables

SPATIAL_REDUCTIONS: tuple[str, ...] = (
    "mean",
    "min",
    "max",
    "sum",
    "count",
)
_SPATIAL_ORDER = {
    name: index for index, name in enumerate(SPATIAL_REDUCTIONS)
}


def resolve_reductions(
    reductions: Iterable[str] | str | None,
) -> tuple[str, ...]:
    """Validate spatial reductions and return them in canonical order."""

    if reductions is None:
        names: tuple[str, ...] = ("mean",)
    elif isinstance(reductions, str):
        names = (reductions,)
    else:
        names = tuple(str(name) for name in reductions)
    if not names:
        raise ValueError("at least one spatial reduction must be selected")
    seen: set[str] = set()
    for name in names:
        if name not in SPATIAL_REDUCTIONS:
            raise ValueError(
                f"unsupported spatial reduction {name!r}; expected one of "
                f"{list(SPATIAL_REDUCTIONS)}"
            )
        if name in seen:
            continue
        seen.add(name)
    return tuple(sorted(seen, key=_SPATIAL_ORDER.__getitem__))


def column_name(variable: str, reduction: str) -> str:
    """Return the output column name for a variable/reduction pair."""

    return f"{variable}_{reduction}"


class RoClimateArea:
    """Daily spatial summaries for a grid window or polygon.

    The object is created by :meth:`ro_climate.RoClimate.area`; it is normally
    not instantiated directly. Each data column is named
    ``{variable}_{reduction}``, for example ``max_temp_mean``.
    """

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        i0: int,
        i1: int,
        j0: int,
        j1: int,
        data: pd.DataFrame,
        variables: Iterable[Variable],
        reductions: Iterable[str],
        frontiers: dict[str, pd.Timestamp],
        store_url: str,
        item_url: str | None = None,
        polygon: Iterable[tuple[float, float]] | None = None,
        n_cells: int | None = None,
        coverage: pd.DataFrame | None = None,
        warnings: Iterable[str] = (),
        retrieved_at: str | datetime | None = None,
        requested: dict | None = None,
    ) -> None:
        """Create an area summary.

        Parameters
        ----------
        bounds:
            ``(west, south, east, north)`` bounds of the loaded window.
        i0, i1, j0, j1:
            Inclusive grid indices of the loaded window.
        data:
            Daily spatial summaries indexed by date.
        variables:
            Variables represented in ``data``.
        reductions:
            Spatial reductions represented in ``data``.
        frontiers:
            Last available day for each variable.
        store_url:
            Source Zarr store.
        item_url:
            Optional STAC item URL.
        polygon:
            Optional polygon used as a cell-centre mask.
        n_cells:
            Number of selected cells. Defaults to the full window size.
        coverage:
            Optional coverage table.
        warnings:
            Warnings recorded while loading the area.
        retrieved_at:
            Retrieval timestamp.
        requested:
            Optional JSON-serialisable request metadata.
        """
        if data.empty:
            raise NoDataError("area data are empty")
        self.bounds = tuple(float(value) for value in bounds)
        self.i0 = int(i0)
        self.i1 = int(i1)
        self.j0 = int(j0)
        self.j1 = int(j1)
        self.data = data
        self.variables = tuple(variables)
        self.reductions = tuple(reductions)
        self.frontiers = dict(frontiers)
        self.store_url = store_url
        self.item_url = item_url
        self.polygon = (
            tuple((float(lon), float(lat)) for lon, lat in polygon)
            if polygon is not None
            else None
        )
        self.n_cells = (
            int(n_cells)
            if n_cells is not None
            else (self.i1 - self.i0 + 1) * (self.j1 - self.j0 + 1)
        )
        self._coverage = (
            coverage.reindex(
                [variable.name for variable in self.variables]
            ).copy()
            if coverage is not None
            else None
        )
        self.warnings = tuple(warnings)
        self.retrieved_at = retrieved_at
        self.requested = dict(requested) if requested is not None else None
        self.start = pd.Timestamp(data.index.min())
        self.end = pd.Timestamp(data.index.max())

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """Return the underlying window shape as ``(rows, columns)``."""
        return self.j1 - self.j0 + 1, self.i1 - self.i0 + 1

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the loaded data column names."""
        return tuple(str(name) for name in self.data.columns)

    @property
    def units(self) -> dict[str, str]:
        """Return the unit of each loaded variable."""
        return {variable.name: variable.units for variable in self.variables}

    @property
    def attribution(self) -> str:
        """Return the data attribution text."""
        return ATTRIBUTION

    @property
    def is_covered(self) -> bool:
        """Return whether at least one daily summary is valid."""
        return bool(self.data.notna().any().any())

    @property
    def coverage(self) -> pd.DataFrame:
        """Return valid-cell coverage per variable.

        Uses a ``count`` reduction when present; otherwise returns the
        coverage recorded when the area was loaded, or ``NaN`` for derived
        subsets.
        """
        n_days = len(self.data)
        n_cell_days = self.n_cells * n_days
        exact = all(
            column_name(variable.name, "count") in self.data.columns
            for variable in self.variables
        )
        if exact:
            rows = []
            for variable in self.variables:
                n_valid = int(
                    self.data[column_name(variable.name, "count")].sum()
                )
                rows.append(
                    {
                        "variable": variable.name,
                        "n_cells": self.n_cells,
                        "n_days": n_days,
                        "n_cell_days": n_cell_days,
                        "n_valid": n_valid,
                        "valid_fraction": (
                            n_valid / n_cell_days if n_cell_days else np.nan
                        ),
                    }
                )
            return pd.DataFrame(rows).set_index("variable")
        if self._coverage is not None:
            return self._coverage.copy()
        rows = [
            {
                "variable": variable.name,
                "n_cells": self.n_cells,
                "n_days": n_days,
                "n_cell_days": n_cell_days,
                "n_valid": np.nan,
                "valid_fraction": np.nan,
            }
            for variable in self.variables
        ]
        return pd.DataFrame(rows).set_index("variable")

    def __len__(self) -> int:
        """Return the number of loaded days."""
        return len(self.data)

    def __getitem__(self, column: str | Iterable[str]):
        """Return one output column as a Series or several as a DataFrame."""
        if isinstance(column, str):
            self._require(column)
            return self.data[column]
        names = [str(name) for name in column]
        for name in names:
            self._require(name)
        return self.data[names]

    def __repr__(self) -> str:
        """Return a compact summary of the area and its date range."""
        names = ",".join(variable.name for variable in self.variables)
        reductions = ",".join(self.reductions)
        return (
            f"RoClimateArea(bounds={self.bounds}, "
            f"grid=({self.i0}:{self.i1},{self.j0}:{self.j1}), "
            f"{self.n_cells} cells, "
            f"{self.start.date()}..{self.end.date()}, "
            f"{len(self)} days, vars=[{names}], "
            f"reductions=[{reductions}])"
        )

    def at(self, date: grid.DateLike) -> pd.Series:
        """Return all daily summary columns for one day."""
        ts = grid.normalize_date(date)
        try:
            return self.data.loc[ts]
        except KeyError as exc:
            raise NoDataError(
                f"{ts.date()} is not in the loaded range"
            ) from exc

    def series(
        self, variable: str, reduction: str = "mean"
    ) -> pd.Series:
        """Return one ``variable``/``reduction`` daily series."""
        if variable not in {item.name for item in self.variables}:
            raise UnknownVariableError(
                f"variable {variable!r} is not loaded for this area"
            )
        if reduction not in self.reductions:
            raise ValueError(
                f"reduction {reduction!r} is not loaded for this area"
            )
        column = column_name(variable, reduction)
        self._require(column)
        return self.data[column]

    def subset(
        self,
        start: grid.DateLike | None = None,
        end: grid.DateLike | None = None,
        variables: Iterable[str] | str | None = None,
        reductions: Iterable[str] | str | None = None,
        months: int | Iterable[int] | None = None,
        season: str | None = None,
    ) -> "RoClimateArea":
        """Return a new area filtered by date, variable, month, or season."""
        specs = self._selected_specs(variables, reductions)
        columns = [column for _, _, column in specs]
        start_ts = (
            grid.normalize_date(start) if start is not None else self.start
        )
        end_ts = grid.normalize_date(end) if end is not None else self.end
        if start_ts > end_ts:
            raise InvalidTimeRangeError(
                f"start {start_ts.date()} is after end {end_ts.date()}"
            )
        frame = self.data.loc[start_ts:end_ts, columns]
        months_filter = grid.month_numbers(months=months, season=season)
        if months_filter is not None:
            frame = frame[frame.index.month.isin(months_filter)]
        if frame.empty:
            raise NoDataError(
                f"no data between {start_ts.date()} and {end_ts.date()}"
                + (f" for months {months_filter}" if months_filter else "")
            )
        selected_vars = tuple(
            {variable.name: variable for variable, _, _ in specs}.values()
        )
        selected_reductions = tuple(
            dict.fromkeys(reduction for _, reduction, _ in specs)
        )
        coverage = (
            self._coverage
            if frame.index.equals(self.data.index)
            else None
        )
        return RoClimateArea(
            bounds=self.bounds,
            i0=self.i0,
            i1=self.i1,
            j0=self.j0,
            j1=self.j1,
            data=frame.copy(),
            variables=selected_vars,
            reductions=selected_reductions,
            frontiers=self.frontiers,
            store_url=self.store_url,
            item_url=self.item_url,
            polygon=self.polygon,
            n_cells=self.n_cells,
            coverage=coverage,
            warnings=self.warnings,
            retrieved_at=self.retrieved_at,
            requested=self.requested,
        )

    # ------------------------------------------------------------------
    # Statistics over the daily spatial summaries
    # ------------------------------------------------------------------
    def mean(self, variables=None, reductions=None, *, skipna: bool = True):
        """Return the temporal mean of each selected summary series."""
        return self._temporal_stat(
            "mean", variables, reductions, skipna=skipna
        )

    def avg(self, variables=None, reductions=None, *, skipna: bool = True):
        """Alias for :meth:`mean`."""
        return self.mean(variables, reductions, skipna=skipna)

    def max(self, variables=None, reductions=None, *, skipna: bool = True):
        """Return the temporal maximum of each selected summary series."""
        return self._temporal_stat(
            "max", variables, reductions, skipna=skipna
        )

    def min(self, variables=None, reductions=None, *, skipna: bool = True):
        """Return the temporal minimum of each selected summary series."""
        return self._temporal_stat(
            "min", variables, reductions, skipna=skipna
        )

    def median(self, variables=None, reductions=None, *, skipna: bool = True):
        """Return the temporal median of each selected summary series."""
        return self._temporal_stat(
            "median", variables, reductions, skipna=skipna
        )

    def std(self, variables=None, reductions=None, *, skipna: bool = True):
        """Return the sample standard deviation of each summary series."""
        return self._temporal_stat(
            "std", variables, reductions, skipna=skipna
        )

    def var(self, variables=None, reductions=None, *, skipna: bool = True):
        """Return the sample variance of each summary series."""
        return self._temporal_stat(
            "var", variables, reductions, skipna=skipna
        )

    def quantile(self, q, variables=None, reductions=None):
        """Return the requested temporal quantile of each summary series."""
        specs = self._selected_specs(variables, reductions)
        frame = self.data[[column for _, _, column in specs]]
        return self._format_result(frame.quantile(q), specs)

    def count(self, variables=None, reductions=None):
        """Return the number of valid days in each summary series."""
        return self._temporal_stat(
            "count", variables, reductions, skipna=True, integer=True
        )

    def sum(
        self,
        variables=None,
        reductions=None,
        *,
        skipna: bool = True,
        min_count: int = 1,
        allow_non_additive: bool = False,
    ):
        """Return the temporal sum of additive summary series.

        With no ``variables``, precipitation is used when it is loaded.
        Temperature series raise :class:`NonAdditiveVariableError` unless
        ``allow_non_additive=True`` is passed.
        """
        if variables is None:
            if any(item.name == "precipitation" for item in self.variables):
                variables = "precipitation"
            else:
                variables = [item.name for item in self.variables]
        return self._temporal_stat(
            "sum",
            variables,
            reductions,
            skipna=skipna,
            min_count=min_count,
            allow_non_additive=allow_non_additive,
        )

    def describe(self, variables=None, reductions=None) -> pd.DataFrame:
        """Return a per-summary descriptive statistics table."""
        specs = self._selected_specs(variables, reductions)
        columns = [column for _, _, column in specs]
        frame = self.data[columns]
        desc = frame.describe().T
        desc["missing"] = frame.isna().sum()
        desc["sum"] = np.nan
        for variable, _, column in specs:
            if variable.additive:
                desc.loc[column, "sum"] = frame[column].sum(min_count=1)
        desc.index.name = "column"
        return desc

    # ------------------------------------------------------------------
    # Derived variables
    # ------------------------------------------------------------------
    @property
    def tmean(self) -> pd.Series:
        """Return the daily spatial mean temperature.

        The value is ``(max_temp_mean + min_temp_mean) / 2`` and therefore
        requires the ``mean`` reduction for both temperature variables.
        """
        self._require("max_temp_mean", "min_temp_mean")
        result = (
            self.data["max_temp_mean"] + self.data["min_temp_mean"]
        ) / 2.0
        result.name = "tmean"
        result.attrs["units"] = "°C"
        return result

    def mean_temperature(self) -> float:
        """Return the mean of the daily spatial mean temperature series."""
        return float(self.tmean.mean())

    # ------------------------------------------------------------------
    # Time aggregation
    # ------------------------------------------------------------------
    def resample(
        self,
        freq: str,
        how: str = "mean",
        variables=None,
        reductions=None,
        *,
        min_count: int | None = 1,
    ) -> pd.DataFrame:
        """Resample the daily spatial summaries to ``freq``."""
        how = str(how).lower()
        if how not in VALID_HOW:
            raise ValueError(
                f"unsupported resampling method {how!r}; expected one of "
                f"{sorted(VALID_HOW)}"
            )
        specs = self._selected_specs(variables, reductions)
        columns = [column for _, _, column in specs]
        frame = self.data[columns]
        if how == "sum":
            bad = [
                variable.name
                for variable, _, _ in specs
                if not variable.additive
            ]
            if bad:
                raise NonAdditiveVariableError(
                    f"cannot sum {', '.join(bad)}; use mean/max/min"
                )
        grouped = frame.resample(freq)
        if how == "count":
            result = grouped.count()
        elif how == "sum":
            result = grouped.sum(
                min_count=0 if min_count is None else int(min_count)
            )
        else:
            result = grouped.agg(how)
            if min_count is not None:
                result = result.where(grouped.count() >= int(min_count))
        result.attrs["units"] = {
            column: variable.units for variable, _, column in specs
        }
        return result

    def monthly_mean(self, variables=None, reductions=None) -> pd.DataFrame:
        """Return monthly means of the daily spatial summaries."""
        return self.resample("MS", "mean", variables, reductions)

    def monthly_total(
        self, variables="precipitation", reductions=None
    ) -> pd.DataFrame:
        """Return monthly totals, precipitation by default."""
        return self.resample("MS", "sum", variables, reductions)

    def annual_mean(self, variables=None, reductions=None) -> pd.DataFrame:
        """Return annual means of the daily spatial summaries."""
        return self.resample("YS", "mean", variables, reductions)

    def annual_total(
        self, variables="precipitation", reductions=None
    ) -> pd.DataFrame:
        """Return annual totals, precipitation by default."""
        return self.resample("YS", "sum", variables, reductions)

    def annual_extremes(self, variables=None, reductions=None) -> pd.DataFrame:
        """Return yearly maxima and minima with ``_max``/``_min`` suffixes."""
        highest = self.resample(
            "YS", "max", variables, reductions
        ).add_suffix("_max")
        lowest = self.resample(
            "YS", "min", variables, reductions
        ).add_suffix("_min")
        return highest.join(lowest)

    def extreme_dates(self, variables=None, reductions=None) -> pd.DataFrame:
        """Return minima, maxima, and their dates for each summary series."""
        specs = self._selected_specs(variables, reductions)
        frame = self.data[[column for _, _, column in specs]]
        result = _extreme_dates(frame)
        result.index.name = "column"
        return result

    def annual_mean_temperature(self) -> pd.Series:
        """Return annual means of the daily spatial mean temperature."""
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

    def wettest_year(self, reduction: str = "mean") -> tuple[int, float]:
        """Return the wettest year and its annual precipitation total."""
        year, value = _extreme_index(
            self._annual_precipitation(reduction), "max"
        )
        return int(pd.Timestamp(year).year), value

    def driest_year(self, reduction: str = "mean") -> tuple[int, float]:
        """Return the driest year and its annual precipitation total."""
        year, value = _extreme_index(
            self._annual_precipitation(reduction), "min"
        )
        return int(pd.Timestamp(year).year), value

    def precipitation_summary(self, reduction: str = "mean") -> dict:
        """Return mean annual precipitation and the wettest/driest years."""
        annual = self._annual_precipitation(reduction)
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
        """Return means grouped by calendar month or day of year."""
        specs = self._selected_specs()
        columns = [column for _, _, column in specs]
        freq = freq.lower()
        if freq in {"month", "monthly"}:
            key = self.data.index.month
            index_name = "month"
        elif freq in {"dayofyear", "doy"}:
            key = self.data.index.dayofyear
            index_name = "dayofyear"
        else:
            raise ValueError("freq must be 'month' or 'dayofyear'")
        result = self.data[columns].groupby(key).mean()
        result.index.name = index_name
        return result

    def aggregate(
        self,
        variable: str,
        reduction: str = "mean",
        how: str = "mean",
        *,
        start: grid.DateLike | None = None,
        end: grid.DateLike | None = None,
        months: int | Iterable[int] | None = None,
        season: str | None = None,
        groupby: str | None = None,
        min_count: int | None = 1,
    ):
        """Filter and temporally aggregate one summary series."""
        how = str(how).lower()
        if how not in VALID_HOW:
            raise ValueError(f"unsupported aggregation {how!r}")
        variable_obj = resolve_variables(variable)[0]
        subset = self.subset(
            start=start,
            end=end,
            variables=[variable_obj.name],
            reductions=[reduction],
            months=months,
            season=season,
        )
        column = column_name(variable_obj.name, reduction)
        series = subset.data[column]
        if how == "sum" and not variable_obj.additive:
            raise NonAdditiveVariableError(
                f"cannot sum {variable_obj.name}; use mean/max/min"
            )
        if groupby is None:
            return reduce_series(series, how, min_count)
        key = str(groupby).lower()
        if key == "year":
            groups = series.groupby(series.index.year)
        elif key == "month":
            groups = series.groupby(series.index.month)
        elif key == "season_year":
            groups = series.groupby(grid.season_year_index(series.index))
        else:
            groups = series.resample(groupby)
        return reduce_grouped(groups, how, min_count)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path, **kwargs) -> Path:
        """Write the daily summaries to CSV and return the path."""
        path = Path(path)
        kwargs.setdefault("index_label", "date")
        kwargs.setdefault("float_format", "%.4f")
        self.data.to_csv(path, **kwargs)
        return path

    def to_json(self, path: str | Path, **kwargs) -> Path:
        """Write the daily summaries to JSON and return the path."""
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
        """Return JSON-serialisable provenance, grid, and coverage metadata."""
        retrieved = self.retrieved_at
        if isinstance(retrieved, datetime):
            retrieved = retrieved.astimezone(timezone.utc).isoformat()
        elif isinstance(retrieved, pd.Timestamp):
            retrieved = retrieved.isoformat()
        if not retrieved:
            retrieved = datetime.now(timezone.utc).isoformat()
        coverage = {}
        for variable, row in self.coverage.iterrows():
            coverage[variable] = {
                key: (
                    None
                    if pd.isna(value)
                    else int(value)
                    if key
                    in {"n_cells", "n_days", "n_cell_days", "n_valid"}
                    else float(value)
                )
                for key, value in row.items()
            }
        return {
            "format_version": 1,
            "kind": "area",
            "store_url": self.store_url,
            "item_url": self.item_url,
            "requested": self.requested or {},
            "grid": {
                "i0": self.i0,
                "i1": self.i1,
                "j0": self.j0,
                "j1": self.j1,
                "n_rows": self.shape[0],
                "n_cols": self.shape[1],
                "n_cells": self.n_cells,
                "bounds": list(self.bounds),
                "polygon": (
                    [list(point) for point in self.polygon]
                    if self.polygon is not None
                    else None
                ),
            },
            "variables": [variable.name for variable in self.variables],
            "reductions": list(self.reductions),
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
            "coverage": coverage,
            "dtype": "float32",
            "column_order": [str(column) for column in self.data.columns],
            "retrieved_at": retrieved,
            "license": "CC-BY-4.0",
            "attribution": ATTRIBUTION,
            "warnings": list(self.warnings),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _annual_precipitation(self, reduction: str = "mean") -> pd.Series:
        """Return annual precipitation totals without empty years."""
        column = column_name("precipitation", reduction)
        self._require(column)
        annual = self.annual_total("precipitation", [reduction])
        return annual[column].dropna()

    def _selected_specs(
        self,
        variables=None,
        reductions=None,
    ) -> list[tuple[Variable, str, str]]:
        """Return validated ``(variable, reduction, column)`` triples."""
        if variables is None:
            selected_vars = self.variables
        else:
            selected_vars = resolve_variables(variables)
            loaded = {item.name for item in self.variables}
            missing = [
                item.name for item in selected_vars if item.name not in loaded
            ]
            if missing:
                raise UnknownVariableError(
                    "variable(s) not loaded for this area: "
                    + ", ".join(missing)
                )
        if reductions is None:
            selected_reductions = self.reductions
        else:
            selected_reductions = resolve_reductions(reductions)
            missing = [
                name for name in selected_reductions
                if name not in self.reductions
            ]
            if missing:
                raise ValueError(
                    "reduction(s) not loaded for this area: "
                    + ", ".join(missing)
                )
        specs: list[tuple[Variable, str, str]] = []
        for variable in selected_vars:
            for reduction in selected_reductions:
                column = column_name(variable.name, reduction)
                self._require(column)
                specs.append((variable, reduction, column))
        return specs

    def _temporal_stat(
        self,
        how: str,
        variables=None,
        reductions=None,
        *,
        skipna: bool = True,
        min_count: int | None = None,
        allow_non_additive: bool = False,
        integer: bool = False,
    ):
        """Apply a temporal reduction to the selected summary columns."""
        specs = self._selected_specs(variables, reductions)
        columns = [column for _, _, column in specs]
        frame = self.data[columns]
        if how == "sum":
            bad = [
                variable.name
                for variable, _, _ in specs
                if not variable.additive
            ]
            if bad and not allow_non_additive:
                raise NonAdditiveVariableError(
                    "sum is only meaningful for additive variables "
                    f"({', '.join(bad)} cannot be summed); "
                    "use mean/max/min or pass allow_non_additive=True"
                )
            result = frame.sum(
                skipna=skipna,
                min_count=1 if min_count is None else int(min_count),
            )
        elif how == "count":
            result = frame.count()
        elif how in {"mean", "min", "max", "median", "std", "var"}:
            result = getattr(frame, how)(skipna=skipna)
        else:
            raise ValueError(f"unsupported aggregation {how!r}")
        return self._format_result(result, specs, integer=integer)

    @staticmethod
    def _format_result(
        values: pd.Series,
        specs: list[tuple[Variable, str, str]],
        *,
        integer: bool = False,
    ):
        """Convert a pandas result to a scalar or annotated Series."""
        if len(specs) == 1:
            value = values.iloc[0]
            if integer:
                return int(value)
            return float(value)
        if integer:
            values = values.astype(int)
        values.attrs["units"] = {
            column: variable.units for variable, _, column in specs
        }
        return values

    def _require(self, *columns: str) -> None:
        """Raise :class:`UnknownVariableError` if a column is missing."""
        missing = [
            column for column in columns if column not in self.data.columns
        ]
        if missing:
            raise UnknownVariableError(
                f"column(s) not loaded for this area: {', '.join(missing)}"
            )
