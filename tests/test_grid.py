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


def test_normalize_polygon_and_bounds():
    polygon = [
        (26.10, 44.43),
        (26.11, 44.43),
        (26.11, 44.44),
        (26.10, 44.43),
    ]
    assert grid.normalize_polygon(polygon) == (
        (26.10, 44.43),
        (26.11, 44.43),
        (26.11, 44.44),
    )
    assert grid.polygon_bounds(polygon) == pytest.approx(
        (26.10, 44.43, 26.11, 44.44)
    )
    with pytest.raises(ValueError):
        grid.normalize_polygon([(26.1, 44.4), (26.2, 44.4)])
    with pytest.raises(ValueError):
        grid.normalize_polygon(
            [(26.1, 44.4), (26.2, 44.4), (float("nan"), 44.5)]
        )


def test_bbox_to_index_matches_cell_centres():
    assert grid.bbox_to_index(26.10, 44.43, 26.11, 44.44) == (
        597,
        597,
        392,
        392,
    )
    assert grid.bbox_to_index(26.10, 44.41, 26.14, 44.44) == (
        597,
        600,
        392,
        394,
    )
    # A window narrower than one cell snaps to the containing cell centre.
    assert grid.bbox_to_index(26.102, 44.432, 26.104, 44.434) == (
        597,
        597,
        392,
        392,
    )


def test_bbox_to_index_clips_to_grid():
    i0, i1, j0, j1 = grid.bbox_to_index(20.0, 43.0, 20.2, 43.6)
    assert 0 <= i0 <= i1 < grid.N_X
    assert 0 <= j0 <= j1 < grid.N_Y
    assert i0 == 0
    assert j1 == grid.N_Y - 1

    i0, i1, j0, j1 = grid.bbox_to_index(29.7, 48.3, 30.0, 49.0)
    assert i1 == grid.N_X - 1
    assert j0 == 0


def test_bbox_to_index_rejects_invalid_windows():
    with pytest.raises(OutsideGridError):
        grid.bbox_to_index(10.0, 40.0, 11.0, 41.0)
    with pytest.raises(OutsideGridError):
        grid.bbox_to_index(float("nan"), 44.0, 26.0, 45.0)
    with pytest.raises(ValueError):
        grid.bbox_to_index(26.1, 44.4, 26.1, 44.5)
    with pytest.raises(ValueError):
        grid.bbox_to_index(26.1, 44.5, 26.2, 44.4)


def test_polygon_mask_selects_cell_centres():
    polygon = [
        (26.101, 44.431),
        (26.109, 44.431),
        (26.109, 44.439),
        (26.101, 44.439),
    ]
    mask = grid.polygon_mask(polygon, 597, 597, 392, 392)
    assert mask.shape == (1, 1)
    assert mask[0, 0]

    full = grid.polygon_mask(polygon, 596, 598, 391, 393)
    assert full.shape == (3, 3)
    assert full.sum() == 1
    assert full[1, 1]

    outside = grid.polygon_mask(
        [(20.5, 48.0), (20.6, 48.0), (20.6, 48.1)],
        597,
        597,
        392,
        392,
    )
    assert not outside.any()

    with pytest.raises(ValueError):
        grid.polygon_mask(polygon, 2, 1, 0, 0)
