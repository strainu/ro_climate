import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from ro_climate import RoClimate, RoClimateArea, DEFAULT_VARIABLES
from ro_climate import grid
from ro_climate.area import column_name, resolve_reductions
from ro_climate.exceptions import (
    InvalidTimeRangeError,
    NoDataError,
    NonAdditiveVariableError,
    OutsideGridError,
    UnknownVariableError,
)

START = "2020-01-01"
I0, I1 = 597, 600
J0, J1 = 392, 394
N_DAYS = 5
N_ROWS = J1 - J0 + 1
N_COLS = I1 - I0 + 1

FRONTIERS = {name: pd.Timestamp("2020-01-05") for name in DEFAULT_VARIABLES}
OFFSETS = {"max_temp": 0.0, "min_temp": 100.0, "precipitation": 200.0}

CELL_BOUNDS = (26.10, 44.43, 26.11, 44.44)
FULL_BOUNDS = (26.10, 44.41, 26.14, 44.44)
CELL_POLYGON = (
    (26.101, 44.431),
    (26.109, 44.431),
    (26.109, 44.439),
    (26.101, 44.439),
)


def base_data() -> np.ndarray:
    """Return a deterministic array with missing cells and days."""

    shape = (N_DAYS, N_ROWS, N_COLS)
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape) + 1.0
    data[0, 0, 0] = np.nan
    data[2, :, :] = np.nan
    data[3, :, :] = np.nan
    data[3, 0, 0] = 42.0
    return data


def variable_data(name: str, fill=None) -> np.ndarray:
    """Return fake data for one variable."""

    if fill is not None:
        return np.full(
            (N_DAYS, N_ROWS, N_COLS), fill, dtype="float32"
        )
    return base_data() + OFFSETS[name]


def expected_series(
    data: np.ndarray,
    reduction: str,
    *,
    mask: np.ndarray | None = None,
    min_count: int = 1,
) -> np.ndarray:
    """Compute the expected daily spatial reduction for fake data."""

    values = data[:, mask] if mask is not None else data.reshape(N_DAYS, -1)
    counts = np.count_nonzero(~np.isnan(values), axis=1)
    if reduction == "count":
        return counts
    valid = counts >= max(1, int(min_count))
    result = np.full(N_DAYS, np.nan, dtype="float64")
    for day in np.flatnonzero(valid):
        row = values[day]
        if reduction == "mean":
            result[day] = np.nanmean(row)
        elif reduction == "sum":
            result[day] = np.nansum(row)
        elif reduction == "min":
            result[day] = np.nanmin(row)
        elif reduction == "max":
            result[day] = np.nanmax(row)
    return result


class FakeArray:
    """Fake Zarr array exposing a small global grid window."""

    def __init__(self, data, chunks=(2, 2, 2)):
        self.data = data
        self.chunks = chunks
        self.t0 = grid.date_to_index(START)
        self.i0 = I0
        self.j0 = J0
        self.reads = []

    def __getitem__(self, key):
        t_key, j_key, i_key = key
        self.reads.append(
            (
                (t_key.start, t_key.stop),
                (j_key.start, j_key.stop),
                (i_key.start, i_key.stop),
            )
        )
        t = slice(t_key.start - self.t0, t_key.stop - self.t0)
        j = slice(j_key.start - self.j0, j_key.stop - self.j0)
        i = slice(i_key.start - self.i0, i_key.stop - self.i0)
        return self.data[t, j, i]


class FakeGroup:
    """Fake Zarr group with the three climate arrays."""

    def __init__(self, fill=None):
        self.arrays = {
            name: FakeArray(variable_data(name, fill=fill))
            for name in DEFAULT_VARIABLES
        }

    def __getitem__(self, name):
        return self.arrays[name]


def make_client(tmp_path, *, fill=None):
    """Create a fake-backed client that never touches the network."""

    client = RoClimate(
        store_url="https://example.invalid/store/",
        discover=False,
        chunk_cache_dir=tmp_path,
    )
    group = FakeGroup(fill=fill)
    client._open_group = lambda: group
    client.frontiers = lambda *, force=False: dict(FRONTIERS)
    client.test_group = group
    return client


def test_resolve_reductions_order_and_validation():
    assert resolve_reductions(None) == ("mean",)
    assert resolve_reductions("max") == ("max",)
    assert resolve_reductions(["count", "mean", "max"]) == (
        "mean",
        "max",
        "count",
    )
    assert resolve_reductions(["mean", "mean"]) == ("mean",)
    with pytest.raises(ValueError):
        resolve_reductions([])
    with pytest.raises(ValueError):
        resolve_reductions("median")
    assert column_name("max_temp", "mean") == "max_temp_mean"


def test_area_spatial_reductions_and_columns(tmp_path):
    client = make_client(tmp_path)
    reductions = ("mean", "min", "max", "sum", "count")
    area = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables="max_temp",
        reductions=reductions,
        allow_non_additive=True,
    )

    assert isinstance(area, RoClimateArea)
    assert area.columns == tuple(
        f"max_temp_{name}" for name in reductions
    )
    data = variable_data("max_temp")
    for name in reductions:
        expected = expected_series(data, name)
        np.testing.assert_allclose(
            area.data[f"max_temp_{name}"].to_numpy(),
            expected,
            equal_nan=True,
        )
    assert area.n_cells == N_ROWS * N_COLS
    assert (area.i0, area.i1, area.j0, area.j1) == (I0, I1, J0, J1)
    assert area.shape == (N_ROWS, N_COLS)
    assert area.units == {"max_temp": "°C"}
    assert area.attribution
    assert area.start == pd.Timestamp(START)
    assert area.end == pd.Timestamp("2020-01-05")
    assert area.is_covered
    assert "RoClimateArea" in repr(area)
    client.close()


def test_area_default_reduction_is_mean(tmp_path):
    client = make_client(tmp_path)
    area = client.area(
        bounds=FULL_BOUNDS, start=START, end="2020-01-05", variables="max_temp"
    )
    assert area.reductions == ("mean",)
    assert area.columns == ("max_temp_mean",)
    assert area.coverage.loc["max_temp", "n_cells"] == N_ROWS * N_COLS
    client.close()


def test_area_polygon_mask_and_bounds(tmp_path):
    client = make_client(tmp_path)
    area = client.area(
        polygon=CELL_POLYGON,
        start=START,
        end="2020-01-05",
        variables="max_temp",
    )

    assert area.bounds == pytest.approx(
        (26.101, 44.431, 26.109, 44.439)
    )
    assert (area.i0, area.i1, area.j0, area.j1) == (
        597,
        597,
        392,
        392,
    )
    assert area.n_cells == 1
    assert area.polygon == CELL_POLYGON
    data = variable_data("max_temp")
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    mask[0, 0] = True
    np.testing.assert_allclose(
        area.data["max_temp_mean"].to_numpy(),
        expected_series(data, "mean", mask=mask),
        equal_nan=True,
    )
    client.close()


def test_area_polygon_outside_bounds_warns(tmp_path):
    client = make_client(tmp_path)
    polygon = [(26.091, 44.421), (26.119, 44.421), (26.119, 44.449)]
    area = client.area(
        bounds=CELL_BOUNDS,
        polygon=polygon,
        start=START,
        end="2020-01-05",
        variables="max_temp",
    )
    assert area.n_cells == 1
    assert any("outside bounds" in warning for warning in area.warnings)
    client.close()


def test_area_polygon_without_cells_raises(tmp_path):
    client = make_client(tmp_path)
    tiny = [
        (26.108, 44.428),
        (26.112, 44.428),
        (26.112, 44.432),
        (26.108, 44.432),
    ]
    with pytest.raises(NoDataError):
        client.area(polygon=tiny, start=START, end="2020-01-05")
    client.close()


def test_area_reads_only_intersecting_chunks(tmp_path):
    client = make_client(tmp_path)
    array = client.test_group.arrays["max_temp"]

    client.area(
        bounds=CELL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables="max_temp",
    )
    single_cell_reads = len(array.reads)
    assert single_cell_reads == 3
    assert all(
        (i_stop - i_start, j_stop - j_start) == (1, 1)
        for (_, _), (j_start, j_stop), (i_start, i_stop) in array.reads
    )

    array.reads.clear()
    client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables="max_temp",
    )
    assert len(array.reads) == 18
    client.close()


def test_area_polygon_prunes_chunks(tmp_path):
    client = make_client(tmp_path)
    array = client.test_group.arrays["max_temp"]
    client.area(
        polygon=CELL_POLYGON,
        start=START,
        end="2020-01-05",
        variables="max_temp",
    )
    assert len(array.reads) == 3
    client.close()


def test_area_clamps_end_and_reports_warning(tmp_path):
    client = make_client(tmp_path)
    area = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-02-01",
        variables="max_temp",
    )
    assert area.end == FRONTIERS["max_temp"]
    assert any("clamped" in warning for warning in area.warnings)
    client.close()


def test_area_rejects_ranges_after_frontier(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(NoDataError):
        client.area(
            bounds=FULL_BOUNDS,
            start="2020-02-01",
            end="2020-02-10",
            variables="max_temp",
        )
    with pytest.raises(NoDataError):
        client.area(
            bounds=FULL_BOUNDS,
            start="2020-01-05",
            end="2020-01-01",
            variables="max_temp",
        )
    client.close()


def test_area_strict_with_all_missing(tmp_path):
    client = make_client(tmp_path, fill=np.nan)
    area = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables="max_temp",
    )
    assert not area.is_covered
    with pytest.raises(NoDataError):
        client.area(
            bounds=FULL_BOUNDS,
            start=START,
            end="2020-01-05",
            variables="max_temp",
            strict=True,
        )
    client.close()


def test_area_min_count(tmp_path):
    client = make_client(tmp_path)
    area = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables="max_temp",
        reductions=["mean", "count"],
        min_count=2,
    )
    data = variable_data("max_temp")
    np.testing.assert_allclose(
        area.data["max_temp_mean"].to_numpy(),
        expected_series(data, "mean", min_count=2),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        area.data["max_temp_count"].to_numpy(),
        expected_series(data, "count"),
    )
    assert np.isnan(area.data["max_temp_mean"].iloc[3])
    assert area.data["max_temp_count"].iloc[3] == 1
    assert area.coverage.loc["max_temp", "n_valid"] == int(
        area.data["max_temp_count"].sum()
    )
    client.close()


def test_area_min_count_bounds(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(ValueError):
        client.area(
            polygon=CELL_POLYGON,
            start=START,
            end="2020-01-05",
            min_count=2,
        )
    with pytest.raises(ValueError):
        client.area(
            bounds=FULL_BOUNDS,
            start=START,
            end="2020-01-05",
            min_count=0,
        )
    client.close()


def test_area_non_additive_sum_requires_override(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(NonAdditiveVariableError):
        client.area(
            bounds=FULL_BOUNDS,
            start=START,
            end="2020-01-05",
            variables="max_temp",
            reductions="sum",
        )
    area = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables="max_temp",
        reductions="sum",
        allow_non_additive=True,
    )
    assert area.columns == ("max_temp_sum",)

    precip = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables="precipitation",
        reductions="sum",
    )
    assert precip.columns == ("precipitation_sum",)
    client.close()


def test_area_requires_selection_and_valid_inputs(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(ValueError):
        client.area(start=START, end="2020-01-05")
    with pytest.raises(OutsideGridError):
        client.area(
            bounds=(10.0, 40.0, 11.0, 41.0),
            start=START,
            end="2020-01-05",
        )
    with pytest.raises(ValueError):
        client.area(
            bounds=(26.10, 44.43),
            start=START,
            end="2020-01-05",
        )
    with pytest.raises(ValueError):
        client.area(
            bounds=FULL_BOUNDS,
            start=START,
            end="2020-01-05",
            variables=[],
        )
    client.close()


def test_area_variable_order_is_stable(tmp_path):
    client = make_client(tmp_path)
    area = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables=["precipitation", "max_temp"],
        reductions=["count", "mean"],
    )
    assert area.columns == (
        "max_temp_mean",
        "max_temp_count",
        "precipitation_mean",
        "precipitation_count",
    )
    assert [item.name for item in area.variables] == [
        "max_temp",
        "precipitation",
    ]
    client.close()


def test_area_object_access_and_statistics(tmp_path):
    client = make_client(tmp_path)
    area = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables=["max_temp", "precipitation"],
        reductions=["mean", "max", "count"],
    )

    assert len(area) == N_DAYS
    assert area.series("max_temp", "mean") is area.data["max_temp_mean"]
    assert area["max_temp_mean"] is area.data["max_temp_mean"]
    selected = area[["max_temp_mean", "precipitation_mean"]]
    assert list(selected.columns) == [
        "max_temp_mean",
        "precipitation_mean",
    ]
    with pytest.raises(UnknownVariableError):
        area["missing"]
    with pytest.raises(ValueError):
        area.series("max_temp", "min")
    with pytest.raises(UnknownVariableError):
        area.series("min_temp", "mean")

    day = area.at("2020-01-02")
    assert len(day) == 6
    with pytest.raises(NoDataError):
        area.at("2021-01-01")

    subset = area.subset(
        start="2020-01-02",
        end="2020-01-04",
        variables="max_temp",
        reductions="mean",
    )
    assert len(subset) == 3
    assert subset.columns == ("max_temp_mean",)
    assert [item.name for item in subset.variables] == ["max_temp"]
    assert subset.coverage.loc["max_temp", "n_days"] == 3
    assert pd.isna(subset.coverage.loc["max_temp", "n_valid"])

    variable_subset = area.subset(variables="max_temp")
    assert list(variable_subset.coverage.index) == ["max_temp"]

    seasonal = area.subset(months=1, season=None)
    assert len(seasonal) == N_DAYS
    with pytest.raises(NoDataError):
        area.subset(months=2)
    with pytest.raises(InvalidTimeRangeError):
        area.subset(start="2020-01-05", end="2020-01-01")

    mean = area.mean()
    np.testing.assert_allclose(
        mean.to_numpy(),
        area.data.mean().to_numpy(),
        equal_nan=True,
    )
    assert mean.attrs["units"]["max_temp_mean"] == "°C"
    scalar = area.mean(variables="max_temp", reductions="mean")
    assert scalar == pytest.approx(area.data["max_temp_mean"].mean())
    assert area.avg(variables="max_temp", reductions="mean") == scalar
    assert area.max(variables="max_temp", reductions="max") == pytest.approx(
        area.data["max_temp_max"].max()
    )
    assert area.min(variables="max_temp", reductions="mean") == pytest.approx(
        area.data["max_temp_mean"].min()
    )
    assert area.median(
        variables="max_temp", reductions="mean"
    ) == pytest.approx(area.data["max_temp_mean"].median())
    assert area.std(
        variables="max_temp", reductions="mean"
    ) == pytest.approx(area.data["max_temp_mean"].std())
    assert area.var(
        variables="max_temp", reductions="mean"
    ) == pytest.approx(area.data["max_temp_mean"].var())

    counts = area.count()
    assert isinstance(counts, pd.Series)
    assert counts["max_temp_mean"] == area.data["max_temp_mean"].count()

    total = area.sum(variables="precipitation", reductions="mean")
    assert total == pytest.approx(area.data["precipitation_mean"].sum())
    added = area.sum(
        variables="max_temp",
        reductions="mean",
        allow_non_additive=True,
    )
    assert added == pytest.approx(area.data["max_temp_mean"].sum())
    with pytest.raises(NonAdditiveVariableError):
        area.sum(variables="max_temp", reductions="mean")

    quantile = area.quantile(0.5, variables="max_temp", reductions="mean")
    assert quantile == pytest.approx(area.data["max_temp_mean"].median())

    description = area.describe(variables="max_temp")
    assert description.loc["max_temp_mean", "missing"] == 1
    assert np.isnan(description.loc["max_temp_mean", "sum"])

    coverage = area.coverage
    assert coverage.loc["max_temp", "n_valid"] == int(
        area.data["max_temp_count"].sum()
    )
    assert coverage.loc["max_temp", "valid_fraction"] == pytest.approx(
        area.data["max_temp_count"].sum()
        / (area.n_cells * len(area))
    )
    assert area.coverage.loc["precipitation", "n_cells"] == area.n_cells
    client.close()


def test_area_object_time_aggregation(tmp_path):
    client = make_client(tmp_path)
    area = client.area(
        bounds=FULL_BOUNDS,
        start=START,
        end="2020-01-05",
        variables=["max_temp", "precipitation"],
        reductions=["mean"],
    )

    resampled = area.resample("2D", "mean")
    assert len(resampled) == 3
    assert "max_temp_mean" in resampled.columns
    with pytest.raises(ValueError):
        area.resample("2D", "bogus")
    with pytest.raises(NonAdditiveVariableError):
        area.resample("2D", "sum", variables="max_temp")

    assert len(area.monthly_mean()) == 1
    assert len(area.monthly_total()) == 1
    assert len(area.annual_mean()) == 1
    assert len(area.annual_total()) == 1

    climate = area.climatology()
    assert list(climate.index) == [1]
    assert climate.index.name == "month"
    daily = area.climatology("dayofyear")
    assert len(daily) == N_DAYS
    with pytest.raises(ValueError):
        area.climatology("year")

    aggregated = area.aggregate("max_temp", how="mean")
    assert aggregated == pytest.approx(area.data["max_temp_mean"].mean())
    by_year = area.aggregate("max_temp", how="mean", groupby="year")
    assert list(by_year.index) == [2020]
    by_month = area.aggregate("max_temp", how="mean", groupby="month")
    assert list(by_month.index) == [1]
    by_period = area.aggregate("max_temp", how="mean", groupby="2D")
    assert len(by_period) == 3
    precip_total = area.aggregate(
        "precipitation", how="sum", start=START, end="2020-01-05"
    )
    assert precip_total == pytest.approx(
        area.data["precipitation_mean"].sum()
    )
    with pytest.raises(NonAdditiveVariableError):
        area.aggregate("max_temp", how="sum")
    client.close()


def test_area_exports_and_metadata(tmp_path):
    client = make_client(tmp_path)
    area = client.area(
        polygon=CELL_POLYGON,
        start=START,
        end="2020-01-05",
        variables="max_temp",
        reductions=["mean", "count"],
    )

    csv_path = area.to_csv(tmp_path / "area.csv")
    assert csv_path.is_file()
    assert len(pd.read_csv(csv_path)) == N_DAYS

    json_path = area.to_json(tmp_path / "area.json")
    payload = json.loads(json_path.read_text())
    assert payload["columns"] == ["max_temp_mean", "max_temp_count"]
    assert len(payload["data"]) == N_DAYS

    npz_path = area.to_npz(tmp_path / "area.npz")
    with np.load(npz_path) as npz:
        assert npz["values"].shape == (N_DAYS, 2)
        assert list(npz["columns"]) == ["max_temp_mean", "max_temp_count"]

    zip_path = area.to_zip(tmp_path / "area.zip")
    with zipfile.ZipFile(zip_path) as zf:
        metadata = json.loads(zf.read("metadata.json"))
        assert set(zf.namelist()) == {
            "data.npy",
            "dates.npy",
            "metadata.json",
            "README.txt",
        }
    assert metadata["kind"] == "area"
    assert metadata["grid"]["n_cells"] == 1
    assert metadata["grid"]["polygon"] == [
        list(point) for point in CELL_POLYGON
    ]

    meta = area.metadata()
    assert meta["kind"] == "area"
    assert meta["reductions"] == ["mean", "count"]
    assert meta["grid"]["i0"] == 597
    assert meta["grid"]["bounds"] == pytest.approx(
        (26.101, 44.431, 26.109, 44.439)
    )
    assert meta["coverage"]["max_temp"]["n_cells"] == 1
    assert meta["license"] == "CC-BY-4.0"
    client.close()
