import pandas as pd
import pytest

from ro_climate import grid
from ro_climate.exceptions import InvalidTimeRangeError, OutsideGridError


def test_lonlat_to_index_known_points():
    assert grid.lonlat_to_index(26.1025, 44.4325) == (597, 392)
    assert grid.lonlat_to_index(grid.LON_MIN, grid.LAT_MAX) == (0, 0)
    assert grid.lonlat_to_index(grid.LON_MAX, grid.LAT_MIN) == (971, 482)


def test_pixel_centre_and_bounds():
    lon, lat = grid.index_to_lonlat(597, 392)
    assert lon == pytest.approx(26.105)
    assert lat == pytest.approx(44.435)
    lon0, lon1, lat0, lat1 = grid.cell_bounds(597, 392)
    assert (lon0, lon1, lat0, lat1) == pytest.approx(
        (26.10, 26.11, 44.43, 44.44)
    )


def test_outside_grid():
    with pytest.raises(OutsideGridError):
        grid.lonlat_to_index(19.0, 45.0)
    with pytest.raises(OutsideGridError):
        grid.lonlat_to_index(25.0, 50.0)


def test_time_index():
    assert grid.date_to_index("1961-01-01") == 0
    assert grid.date_to_index("2026-11-03") == 24047
    assert grid.index_to_date(24047) == pd.Timestamp("2026-11-03")
    assert grid.date_to_index(grid.TIME_END) == grid.TIME_STEPS - 1


def test_months_and_seasons():
    assert grid.month_numbers(months=1) == (1,)
    assert grid.month_numbers(months=[1, 1, 2]) == (1, 2)
    assert grid.month_numbers(season="djf") == (12, 1, 2)
    with pytest.raises(InvalidTimeRangeError):
        grid.month_numbers(months=1, season="JJA")
    with pytest.raises(InvalidTimeRangeError):
        grid.month_numbers(months=13)


def test_season_year_december():
    idx = pd.DatetimeIndex(["2020-12-31", "2021-01-01", "2021-02-01"])
    sy = grid.season_year_index(idx)
    assert list(sy) == [2021, 2021, 2021]
