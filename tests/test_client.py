import struct

import numpy as np
import pandas as pd
import pytest

from ro_climate import RoClimate, DEFAULT_VARIABLES
from ro_climate import grid
from ro_climate.exceptions import (
    NoDataError,
    OutsideGridError,
    UnknownVariableError,
)

FRONTIERS = {
    "max_temp": pd.Timestamp("2020-01-10"),
    "min_temp": pd.Timestamp("2020-01-12"),
    "precipitation": pd.Timestamp("2020-01-11"),
}


class FakeSeries:
    def __init__(self, offset=0.0, fill=None):
        self.offset = offset
        self.fill = fill

    def __getitem__(self, key):
        t = key[0]
        if not isinstance(t, slice):
            raise TypeError("expected a time slice")
        start = 0 if t.start is None else int(t.start)
        stop = int(t.stop)
        n = stop - start
        if self.fill is not None:
            return np.full(n, self.fill, dtype="float32")
        return (np.arange(start, stop, dtype="float32") + self.offset)


class FakeCoord:
    def __init__(self, origin, step):
        self.origin = float(origin)
        self.step = float(step)

    def __getitem__(self, index):
        return np.float32(self.origin + self.step * int(index))


class FakeGroup:
    def __init__(self, fill=None):
        self.arrays = {
            "max_temp": FakeSeries(10.0, fill=fill),
            "min_temp": FakeSeries(-5.0, fill=fill),
            "precipitation": FakeSeries(1.0, fill=fill),
            "x": FakeCoord(grid.X_MIN, grid.STEP),
            "y": FakeCoord(grid.Y_MAX, -grid.STEP),
        }

    def __getitem__(self, name):
        return self.arrays[name]


class FakeKV:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def close(self):
        pass


def make_client(tmp_path, *, fill=None):
    client = RoClimate(
        store_url="https://example.invalid/store/",
        discover=False,
        chunk_cache_dir=tmp_path,
    )
    group = FakeGroup(fill=fill)
    client._open_group = lambda: group
    client.frontiers = lambda *, force=False: dict(FRONTIERS)
    return client


def test_point_fetch_and_metadata(tmp_path):
    client = make_client(tmp_path)
    point = client.point(
        26.1025, 44.4325, start="2020-01-01", end="2020-01-07"
    )

    assert point.i == 597 and point.j == 392
    assert point.grid_lon == pytest.approx(grid.X_MIN + grid.STEP * 597)
    assert point.grid_lat == pytest.approx(grid.Y_MAX - grid.STEP * 392)
    assert list(point.data.columns) == list(DEFAULT_VARIABLES)
    assert point.start == pd.Timestamp("2020-01-01")
    assert point.end == pd.Timestamp("2020-01-07")
    assert point.data["max_temp"].iloc[0] == pytest.approx(
        grid.date_to_index("2020-01-01") + 10.0
    )
    assert list(tmp_path.glob("*.zip")) == []
    client.close()


def test_point_is_not_archived(tmp_path):
    client = make_client(tmp_path)
    first = client.point(
        26.1025, 44.4325, start="2020-01-01", end="2020-01-07"
    )
    second = client.point(
        26.1025, 44.4325, start="2020-01-01", end="2020-01-07"
    )

    pd.testing.assert_frame_equal(first.data, second.data)
    assert list(tmp_path.glob("*.zip")) == []
    client.close()


def test_point_clamps_end_to_intersection(tmp_path):
    client = make_client(tmp_path)
    point = client.point(
        26.1025, 44.4325, start="2020-01-01", end="2020-02-01"
    )
    assert point.end == FRONTIERS["max_temp"]
    assert any("clamped" in warning for warning in point.warnings)
    client.close()


def test_point_rejects_range_after_frontiers(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(NoDataError):
        client.point(
            26.1025, 44.4325, start="2020-02-01", end="2020-02-10"
        )
    client.close()


def test_point_variable_subset(tmp_path):
    client = make_client(tmp_path)
    point = client.point(
        26.1025,
        44.4325,
        start="2020-01-01",
        end="2020-01-07",
        variables="precipitation",
    )
    assert list(point.data.columns) == ["precipitation"]
    assert point.units == {"precipitation": "mm"}
    client.close()


def test_point_variable_order_is_stable(tmp_path):
    client = make_client(tmp_path)
    reversed_point = client.point(
        26.1025,
        44.4325,
        start="2020-01-01",
        end="2020-01-07",
        variables=["precipitation", "max_temp"],
    )
    assert list(reversed_point.data.columns) == ["max_temp", "precipitation"]

    normal = client.point(
        26.1025,
        44.4325,
        start="2020-01-01",
        end="2020-01-07",
        variables=["max_temp", "precipitation"],
    )
    pd.testing.assert_frame_equal(normal.data, reversed_point.data)
    client.close()


def test_point_outside_grid(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(OutsideGridError):
        client.point(10.0, 44.0, start="2020-01-01", end="2020-01-02")
    client.close()


def test_point_unknown_variable(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(UnknownVariableError):
        client.point(
            26.1025,
            44.4325,
            start="2020-01-01",
            end="2020-01-02",
            variables="wind",
        )
    client.close()


def test_point_strict_with_missing_values(tmp_path):
    client = make_client(tmp_path, fill=np.nan)
    with pytest.raises(NoDataError):
        client.point(
            26.1025,
            44.4325,
            start="2020-01-01",
            end="2020-01-07",
            strict=True,
        )
    client.close()


def test_frontiers_reads_scalar_arrays(tmp_path):
    client = RoClimate(
        store_url="https://example.invalid/store/",
        discover=False,
        chunk_cache=False,
    )
    values = {}
    for name in DEFAULT_VARIABLES:
        day = grid.date_to_index(FRONTIERS[name])
        values[f"last_{name}_time/0"] = struct.pack("<q", day)
    client._store = FakeKV(values)
    client._frontier_cache = None

    result = client.frontiers(force=True)
    assert result == FRONTIERS
    client.close()


def test_chunk_cache_wiring_and_clear(tmp_path):
    client = RoClimate(
        store_url="https://example.invalid/store/",
        discover=False,
        chunk_cache_dir=tmp_path,
    )
    assert client.chunk_cache_info().empty
    client._chunk_cache.put("max_temp/0.0.0", b"x" * 2000)
    info = client.chunk_cache_info()
    assert len(info) == 1
    assert info.iloc[0]["variable"] == "max_temp"
    assert client.clear_cache() == 1
    assert client.chunk_cache_info().empty
    client.close()


def test_chunk_cache_disabled_info_is_empty(tmp_path):
    client = RoClimate(
        store_url="https://example.invalid/store/",
        discover=False,
        chunk_cache=False,
    )
    assert client.chunk_cache_info().empty
    assert list(client.chunk_cache_info().columns) == [
        "variable",
        "chunks",
        "size_mb",
        "oldest",
        "newest",
    ]
    client.close()
