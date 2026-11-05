import os
import time

import pandas as pd
import pytest

from ro_climate._chunk_cache import CachingRequestsStore, ChunkCache

STORE_URL = "https://example.invalid/store/"


class FakeRemote:
    """Minimal read-only store used to exercise the caching wrapper."""

    def __init__(self, values):
        self.values = dict(values)
        self.calls = []
        self.closed = False

    def __getitem__(self, key):
        self.calls.append(key)
        return self.values[key]

    def getitems(self, keys, *, contexts=None):
        keys = list(keys)
        self.calls.extend(keys)
        return {key: self.values[key] for key in keys}

    def __contains__(self, key):
        return key in self.values

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def close(self):
        self.closed = True


def make_cache(tmp_path, *, max_bytes=None, variables=("max_temp",)):
    return ChunkCache(
        tmp_path,
        STORE_URL,
        variables,
        max_bytes=max_bytes,
    )


def test_chunk_key_detection(tmp_path):
    cache = make_cache(tmp_path, variables=("max_temp", "min_temp"))
    assert cache.is_chunk_key("max_temp/0.0.0")
    assert cache.is_chunk_key("min_temp/1500.12.3")
    assert not cache.is_chunk_key("max_temp/.zarray")
    assert not cache.is_chunk_key("max_temp/0.0")
    assert not cache.is_chunk_key("max_temp/a.b.c")
    assert not cache.is_chunk_key("precipitation/0.0.0")
    assert not cache.is_chunk_key("last_max_temp_time/0")
    with pytest.raises(ValueError):
        cache.path_for("max_temp/.zarray")


def test_chunk_cache_put_get_and_info(tmp_path):
    cache = make_cache(tmp_path)
    key = "max_temp/1500.2.3"
    payload = b"x" * 2000
    path = cache.put(key, payload)

    assert path.is_file()
    assert path.read_bytes() == payload
    assert cache.get(key) == payload
    assert cache.contains(key)
    assert cache.get("max_temp/1501.2.3") is None
    assert not cache.contains("max_temp/1501.2.3")

    info = cache.info()
    assert len(info) == 1
    row = info.iloc[0]
    assert row["variable"] == "max_temp"
    assert row["chunks"] == 1
    assert row["size_mb"] > 0
    assert row["oldest"] and row["newest"]


def test_chunk_cache_overwrites_and_clear(tmp_path):
    cache = make_cache(tmp_path)
    key = "max_temp/10.0.0"
    cache.put(key, b"first")
    cache.put(key, b"second")
    assert cache.get(key) == b"second"
    assert cache.info().iloc[0]["chunks"] == 1

    (tmp_path / "unrelated.txt").write_text("keep me")
    assert cache.clear() == 1
    assert cache.get(key) is None
    assert cache.info().empty
    assert (tmp_path / "unrelated.txt").is_file()


def test_chunk_cache_invalidate_aligned_range(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("max_temp/90.1.1", b"a")
    cache.put("max_temp/180.1.1", b"b")
    cache.put("max_temp/1500.2.3", b"c")

    # 0..100 covers the aligned chunk starting at 90 but not the one at 180.
    assert cache.invalidate("max_temp", 0, 100, 90) == 1
    assert cache.get("max_temp/90.1.1") is None
    assert cache.get("max_temp/180.1.1") == b"b"
    assert cache.get("max_temp/1500.2.3") == b"c"

    # 1500..1600 covers the aligned chunk starting at 1440.
    assert cache.invalidate("max_temp", 1500, 1600, 90) == 1
    assert cache.get("max_temp/1500.2.3") is None
    assert cache.invalidate("max_temp", 0, 100, 0) == 0


def test_chunk_cache_invalidate_other_variable(tmp_path):
    cache = make_cache(tmp_path, variables=("max_temp", "min_temp"))
    cache.put("max_temp/0.0.0", b"a")
    cache.put("min_temp/0.0.0", b"b")
    assert cache.invalidate("min_temp", 0, 10, 90) == 1
    assert cache.get("max_temp/0.0.0") == b"a"
    assert cache.get("min_temp/0.0.0") is None


def test_chunk_cache_evicts_oldest(tmp_path):
    cache = make_cache(tmp_path, max_bytes=15)
    cache.put("max_temp/0.0.0", b"0123456789")
    cache.put("max_temp/1.0.0", b"abcdefghij")

    total = sum(
        path.stat().st_size
        for path in cache.directory.rglob("*")
        if path.is_file()
    )
    assert total <= 15
    assert cache.info()["chunks"].sum() >= 1


def test_chunk_cache_prune(tmp_path):
    cache = make_cache(tmp_path)
    old = cache.put("max_temp/0.0.0", b"old")
    new = cache.put("max_temp/1.0.0", b"new")
    stale = time.time() - 10 * 86400
    os.utime(old, (stale, stale))

    assert cache.prune(older_than_days=5) == 1
    assert not old.exists()
    assert new.exists()
    with pytest.raises(ValueError):
        cache.prune(older_than_days=-1)


def test_caching_store_reuses_chunks(tmp_path):
    cache = make_cache(tmp_path)
    remote = FakeRemote(
        {
            "max_temp/0.0.0": b"chunk-a",
            "max_temp/0.0.1": b"chunk-b",
            "max_temp/.zarray": b"metadata",
        }
    )
    store = CachingRequestsStore(remote, cache)

    assert store["max_temp/0.0.0"] == b"chunk-a"
    assert store["max_temp/.zarray"] == b"metadata"
    assert remote.calls == ["max_temp/0.0.0", "max_temp/.zarray"]

    assert store["max_temp/0.0.0"] == b"chunk-a"
    assert remote.calls == ["max_temp/0.0.0", "max_temp/.zarray"]
    assert cache.contains("max_temp/0.0.0")
    assert not cache.contains("max_temp/.zarray")


def test_caching_store_getitems(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("max_temp/0.0.1", b"cached")
    remote = FakeRemote(
        {
            "max_temp/0.0.0": b"remote-a",
            "max_temp/0.0.1": b"remote-b",
            "max_temp/.zarray": b"metadata",
        }
    )
    store = CachingRequestsStore(remote, cache)

    result = store.getitems(
        ["max_temp/0.0.0", "max_temp/0.0.1", "max_temp/.zarray"]
    )
    assert result == {
        "max_temp/0.0.0": b"remote-a",
        "max_temp/0.0.1": b"cached",
        "max_temp/.zarray": b"metadata",
    }
    assert remote.calls == ["max_temp/.zarray", "max_temp/0.0.0"]
    assert cache.get("max_temp/0.0.0") == b"remote-a"
    assert cache.get("max_temp/0.0.1") == b"cached"


def test_caching_store_delegates_and_is_read_only(tmp_path):
    cache = make_cache(tmp_path)
    remote = FakeRemote({"max_temp/0.0.0": b"a"})
    store = CachingRequestsStore(remote, cache)

    assert "max_temp/0.0.0" in store
    assert len(store) == 1
    assert list(store) == ["max_temp/0.0.0"]
    with pytest.raises(TypeError):
        store["max_temp/0.0.0"] = b"b"
    with pytest.raises(TypeError):
        del store["max_temp/0.0.0"]

    store.close()
    assert remote.closed


def test_chunk_cache_info_empty(tmp_path):
    info = make_cache(tmp_path).info()
    assert info.empty
    assert list(info.columns) == [
        "variable",
        "chunks",
        "size_mb",
        "oldest",
        "newest",
    ]
    assert isinstance(info, pd.DataFrame)
