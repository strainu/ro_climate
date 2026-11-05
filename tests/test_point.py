import zipfile

import numpy as np
import pandas as pd
import pytest

from ro_climate import RoClimatePoint, MAX_TEMP, MIN_TEMP, PRECIPITATION
from ro_climate.exceptions import (
    NoDataError,
    NonAdditiveVariableError,
    UnknownVariableError,
)


@pytest.fixture
def point() -> RoClimatePoint:
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    values = np.zeros((len(idx), 3), dtype="float32")
    values[:, 0] = 10.0 + np.arange(len(idx)) % 10
    values[:, 1] = 0.0 + np.arange(len(idx)) % 5
    values[:, 2] = 1.0
    values[0:5, 2] = np.nan
    data = pd.DataFrame(
        values,
        index=idx,
        columns=["max_temp", "min_temp", "precipitation"],
    )
    return RoClimatePoint(
        lon=26.1025,
        lat=44.4325,
        i=597,
        j=392,
        grid_lon=26.105,
        grid_lat=44.435,
        data=data,
        variables=(MAX_TEMP, MIN_TEMP, PRECIPITATION),
        frontiers={
            "max_temp": pd.Timestamp("2026-11-03"),
            "min_temp": pd.Timestamp("2026-11-04"),
            "precipitation": pd.Timestamp("2026-11-03"),
        },
        store_url="https://example.invalid/store/",
    )


def test_basic_properties(point):
    assert len(point) == 366
    assert point.start == pd.Timestamp("2020-01-01")
    assert point.end == pd.Timestamp("2020-12-31")
    assert point.units == {
        "max_temp": "°C",
        "min_temp": "°C",
        "precipitation": "mm",
    }
    assert point.cell_bounds == pytest.approx((26.10, 26.11, 44.43, 44.44))
    assert point.is_covered


def test_stats(point):
    assert point.mean("max_temp") == pytest.approx(
        point.data["max_temp"].mean()
    )
    assert point.max("max_temp") == pytest.approx(point.data["max_temp"].max())
    assert point.min("min_temp") == pytest.approx(point.data["min_temp"].min())
    assert point.median("max_temp") == pytest.approx(
        point.data["max_temp"].median()
    )
    assert point.count("precipitation") == 361
    assert point.sum("precipitation") == pytest.approx(361.0)
    assert point.sum() == pytest.approx(361.0)
    assert "max_temp" in point.mean()
    assert point.mean().attrs["units"]["max_temp"] == "°C"


def test_sum_rejects_temperature(point):
    with pytest.raises(NonAdditiveVariableError):
        point.sum("max_temp")
    with pytest.raises(NonAdditiveVariableError):
        point.resample("MS", "sum", variables="max_temp")


def test_describe_and_coverage(point):
    desc = point.describe()
    assert "missing" in desc.columns
    assert desc.loc["precipitation", "missing"] == 5
    cov = point.coverage
    assert cov.loc["precipitation", "n_missing"] == 5
    assert cov.loc["max_temp", "valid_fraction"] == 1.0


def test_resample_and_climatology(point):
    monthly = point.monthly_mean()
    assert len(monthly) == 12
    assert "max_temp" in monthly.columns
    totals = point.monthly_total()
    assert totals["precipitation"].iloc[0] == pytest.approx(26.0)
    annual = point.annual_total()
    assert annual["precipitation"].iloc[0] == pytest.approx(361.0)
    clim = point.climatology()
    assert len(clim) == 12


def test_derived(point):
    assert point.tmean.iloc[0] == pytest.approx((10.0 + 0.0) / 2)
    assert point.diurnal_range.iloc[0] == pytest.approx(10.0)
    assert point.mean_temperature() == pytest.approx(point.tmean.mean())
    assert point.mean_diurnal_range() == pytest.approx(
        point.diurnal_range.mean()
    )


def test_aggregate_month(point):
    jan = point.aggregate("max_temp", "max", months=1)
    assert jan == pytest.approx(point.data.loc["2020-01", "max_temp"].max())
    by_year = point.aggregate("max_temp", "max", months=1, groupby="year")
    assert by_year.index.tolist() == [2020]
    mean_jan = point.aggregate("max_temp", "mean", months=1)
    assert mean_jan == pytest.approx(
        point.data.loc["2020-01", "max_temp"].mean()
    )


def test_indices(point):
    assert point.frost_days() >= 0
    assert point.hot_days(10) == 366
    assert point.rain_days(1.0) == 361
    assert point.dry_days(1.0) == 0
    assert point.dry_days(1.5) == 361
    assert point.consecutive_dry_days(1.0) == 0
    assert point.max_daily_precip() == pytest.approx(1.0)
    assert point.max_n_day_precip(5) == pytest.approx(5.0)
    assert point.growing_degree_days(0) > 0
    assert point.consecutive_wet_days() >= 1
    by_year = point.hot_days(10, by="year")
    assert by_year.index.tolist() == [2020]


def test_subset_months(point):
    jan = point.subset(months=1, variables="max_temp")
    assert len(jan) == 31
    assert jan.data.columns.tolist() == ["max_temp"]
    with pytest.raises(NoDataError):
        point.subset(start="2021-01-01", end="2021-01-02")
    with pytest.raises(UnknownVariableError):
        jan["precipitation"]


def test_missing_variable(point):
    single = point.subset(variables="precipitation")
    with pytest.raises(UnknownVariableError):
        single.mean("max_temp")
    assert single.sum() == pytest.approx(361.0)
    temperature = point.subset(variables="max_temp")
    with pytest.raises(UnknownVariableError):
        temperature.sum()


def test_exports(point, tmp_path):
    csv = point.to_csv(tmp_path / "p.csv")
    assert csv.is_file()
    frame = pd.read_csv(csv, index_col="date", parse_dates=True)
    assert list(frame.columns) == ["max_temp", "min_temp", "precipitation"]
    assert frame["max_temp"].iloc[0] == pytest.approx(10.0)
    assert frame["precipitation"].isna().sum() == 5

    npz = point.to_npz(tmp_path / "p.npz")
    assert npz.is_file()
    archive = point.to_zip(tmp_path / "p.zip")
    assert archive.is_file()
    with zipfile.ZipFile(archive) as zf:
        assert "data.npy" in zf.namelist()
        assert not any(name.endswith(".csv") for name in zf.namelist())

    js = point.to_json(tmp_path / "p.json")
    assert js.is_file()
