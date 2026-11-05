import json
import os
import time
import zipfile
import numpy as np
import pandas as pd
import pytest

from ro_climate.cache import (
    RoClimateCache,
    CacheKey,
    FORMAT_VERSION,
    default_cache_dir,
    write_archive,
)


def make_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "max_temp": np.array([1.0, 2.0, 3.0, 4.0], dtype="float32"),
            "precipitation": np.array([0.0, 0.1, 0.2, 0.3], dtype="float32"),
        },
        index=index,
    )


def make_metadata() -> dict:
    return {
        "grid": {"i": 597, "j": 392, "lon": 26.105, "lat": 44.435},
        "actual": {"start": "2020-01-01", "end": "2020-01-04"},
        "column_order": ["max_temp", "precipitation"],
        "store_url": "https://example.invalid/store/",
    }


def make_key() -> CacheKey:
    return CacheKey(
        store_url="https://example.invalid/store/",
        i=597,
        j=392,
        variables=("max_temp", "precipitation"),
        lon=26.105,
        lat=44.435,
    )


def test_write_archive_roundtrip(tmp_path):
    path = write_archive(tmp_path / "point.zip", make_frame(), make_metadata())
    assert path.is_file()

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        assert names == {
            "data.npy",
            "dates.npy",
            "metadata.json",
            "README.txt",
        }
        meta = json.loads(zf.read("metadata.json"))
        assert meta["format_version"] == FORMAT_VERSION
        assert meta["column_order"] == ["max_temp", "precipitation"]
        assert meta["attribution"]
        assert "ANM daily gridded" in zf.read("README.txt").decode()


def test_cache_save_and_load(tmp_path):
    cache = RoClimateCache(tmp_path)
    key = make_key()
    frame = make_frame()

    path = cache.save(key, frame, make_metadata())
    assert path == cache.path_for(key)

    cached = cache.load(key)
    assert cached is not None
    assert cached.path == path
    pd.testing.assert_frame_equal(cached.data, frame)
    assert cached.metadata["grid"]["i"] == 597


def test_cache_miss_returns_none(tmp_path):
    cache = RoClimateCache(tmp_path)
    assert cache.load(make_key()) is None


def test_cache_rejects_corrupt_zip(tmp_path):
    cache = RoClimateCache(tmp_path)
    path = cache.path_for(make_key())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a zip file")

    assert cache.load(make_key()) is None
    assert not path.exists()


def test_cache_rejects_wrong_format_version(tmp_path):
    cache = RoClimateCache(tmp_path)
    key = make_key()
    metadata = make_metadata()
    metadata["format_version"] = FORMAT_VERSION + 1
    metadata["column_order"] = list(key.variables)
    path = write_archive(cache.path_for(key), make_frame(), metadata)

    assert cache.load(key) is None
    assert not path.exists()


def test_cache_rejects_wrong_columns(tmp_path):
    cache = RoClimateCache(tmp_path)
    key = make_key()
    metadata = make_metadata()
    metadata["column_order"] = ["max_temp"]
    path = write_archive(cache.path_for(key), make_frame(), metadata)

    assert cache.load(key) is None
    assert not path.exists()


def test_cache_info(tmp_path):
    cache = RoClimateCache(tmp_path)
    key = make_key()
    cache.save(key, make_frame(), make_metadata())

    info = cache.info()
    assert len(info) == 1
    row = info.iloc[0]
    assert row["i"] == 597
    assert row["j"] == 392
    assert row["variables"] == "max_temp,precipitation"
    assert row["start"] == "2020-01-01"
    assert row["end"] == "2020-01-04"
    assert row["size_mb"] > 0


def test_cache_info_empty(tmp_path):
    info = RoClimateCache(tmp_path).info()
    assert info.empty
    assert "path" in info.columns


def test_cache_clear(tmp_path):
    cache = RoClimateCache(tmp_path)
    key = make_key()
    cache.save(key, make_frame(), make_metadata())
    (tmp_path / "not-a-cache.txt").write_text("keep me")

    assert cache.clear() == 1
    assert cache.load(key) is None
    assert (tmp_path / "not-a-cache.txt").is_file()


def test_cache_prune(tmp_path):
    cache = RoClimateCache(tmp_path)
    old = cache.save(make_key(), make_frame(), make_metadata())

    other = CacheKey(
        store_url="https://example.invalid/store/",
        i=0,
        j=0,
        variables=("max_temp",),
        lon=20.135,
        lat=48.355,
    )
    new = write_archive(
        cache.path_for(other),
        make_frame()[["max_temp"]],
        {
            "grid": {"i": 0, "j": 0},
            "actual": {"start": "2020-01-01", "end": "2020-01-04"},
            "column_order": ["max_temp"],
            "store_url": other.store_url,
        },
    )

    stale = time.time() - 10 * 86400
    os.utime(old, (stale, stale))

    assert cache.prune(older_than_days=5) == 1
    assert not old.exists()
    assert new.exists()
    with pytest.raises(ValueError):
        cache.prune(older_than_days=-1)


def test_default_cache_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RO_CLIMATE_CACHE_DIR", str(tmp_path / "custom"))
    assert default_cache_dir() == tmp_path / "custom"


def test_cache_key_digest_depends_on_variables():
    a = make_key()
    b = CacheKey(
        store_url=a.store_url,
        i=a.i,
        j=a.j,
        variables=("max_temp",),
        lon=a.lon,
        lat=a.lat,
    )
    assert a.digest != b.digest


def test_cache_path_uses_grid_centre(tmp_path):
    cache = RoClimateCache(tmp_path)
    key = make_key()
    path = cache.path_for(key)
    assert path.name.startswith(
        "ro_climate_lat44.4350_lon26.1050_max-temp-precipitation_"
    )
    assert path.name.endswith(f"{key.digest[:12]}.zip")
