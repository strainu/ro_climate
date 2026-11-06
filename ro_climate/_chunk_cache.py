"""Private cache for raw Zarr chunks and a caching Zarr store wrapper."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from zarr.storage import Store

from ._store import RequestsStore

DEFAULT_CHUNK_CACHE_MAX_MB = 20_000.0
CHUNK_CACHE_MAX_MB_ENV = "RO_CLIMATE_CHUNK_CACHE_MAX_MB"


def default_chunk_cache_dir() -> Path:
    """Return the platform-appropriate default chunk-cache root."""

    override = os.environ.get("RO_CLIMATE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "ro_climate" / "cache"
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "ro_climate"
    return Path.home() / ".cache" / "ro_climate"


def resolve_chunk_cache_max_mb(value: float | None = None) -> float:
    """Resolve the chunk-cache limit in megabytes.

    Precedence is explicit ``value``, then ``RO_CLIMATE_CHUNK_CACHE_MAX_MB``,
    then :data:`DEFAULT_CHUNK_CACHE_MAX_MB`. A value of ``0`` means unlimited.
    """

    if value is None:
        raw = os.environ.get(CHUNK_CACHE_MAX_MB_ENV)
        value = DEFAULT_CHUNK_CACHE_MAX_MB if raw is None else float(raw)
    limit = float(value)
    if limit < 0:
        raise ValueError("chunk cache size must be non-negative")
    return limit


class ChunkCache:
    """Store compressed Zarr chunk bytes as individual cache files.

    Only keys of the form ``variable/t.y.x`` are considered chunks. Small
    metadata and frontier keys are always passed through to the remote store.
    """

    def __init__(
        self,
        directory: str | Path,
        store_url: str,
        variables: tuple[str, ...],
        *,
        max_bytes: int | None = None,
    ) -> None:
        """Create a chunk cache rooted at ``directory``.

        Parameters
        ----------
        directory:
            Parent directory for chunk files. A store-specific subdirectory is
            created below it.
        store_url:
            Remote store identity; it is included in the cache path.
        variables:
            Variable names whose chunk keys may be cached.
        max_bytes:
            Optional maximum cache size. Oldest files are evicted when the
            limit is exceeded.
        """
        digest = hashlib.sha256(store_url.encode("utf-8")).hexdigest()[:16]
        self.directory = Path(directory).expanduser() / digest
        self.variables = tuple(variables)
        self.max_bytes = int(max_bytes) if max_bytes is not None else None
        self._size = self._scan_size()

    def _scan_size(self) -> int:
        """Return the total size of cached chunk files."""
        if not self.directory.exists():
            return 0
        total = 0
        for path in self.directory.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total

    def is_chunk_key(self, key: str) -> bool:
        """Return whether ``key`` is a cacheable Zarr chunk key."""
        if key.count("/") != 1:
            return False
        variable, coordinates = key.split("/", 1)
        if variable not in self.variables:
            return False
        parts = coordinates.split(".")
        return len(parts) == 3 and all(part.isdigit() for part in parts)

    def path_for(self, key: str) -> Path:
        """Return the local file path for chunk ``key``."""
        if not self.is_chunk_key(key):
            raise ValueError(f"not a cacheable chunk key: {key!r}")
        variable, coordinates = key.split("/", 1)
        time_part = coordinates.split(".", 1)[0]
        bucket = f"{int(time_part) // 1000:04d}"
        return self.directory / variable / bucket / coordinates

    def get(self, key: str) -> bytes | None:
        """Return cached bytes for ``key``, or ``None`` on a miss."""
        path = self.path_for(key)
        try:
            return path.read_bytes()
        except OSError:
            return None

    def put(self, key: str, data: bytes) -> Path:
        """Store chunk bytes atomically and return the cache path."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        old_size = 0
        try:
            old_size = path.stat().st_size
        except OSError:
            pass
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
        self._size += len(data) - old_size
        self._trim()
        return path

    def contains(self, key: str) -> bool:
        """Return whether ``key`` is present in the cache."""
        if not self.is_chunk_key(key):
            return False
        return self.path_for(key).is_file()

    def invalidate(
        self, variable: str, t0: int, t1: int, chunk_size: int
    ) -> int:
        """Delete cached time chunks overlapping ``t0..t1`` for a variable.

        Chunk files are named with their first time index, so the requested
        time range is expanded to the full aligned chunk range before files
        are removed.
        """
        if chunk_size <= 0:
            return 0
        size = int(chunk_size)
        first_chunk = int(t0) // size
        last_chunk = int(t1) // size
        first_time = first_chunk * size
        last_time = (last_chunk + 1) * size - 1
        removed = 0
        for bucket_index in range(first_time // 1000, last_time // 1000 + 1):
            bucket = self.directory / variable / f"{bucket_index:04d}"
            if not bucket.is_dir():
                continue
            for path in bucket.iterdir():
                if not path.is_file():
                    continue
                parts = path.name.split(".")
                if len(parts) != 3 or not parts[0].isdigit():
                    continue
                time_index = int(parts[0])
                if not first_time <= time_index <= last_time:
                    continue
                try:
                    file_size = path.stat().st_size
                    path.unlink()
                except OSError:
                    continue
                self._size = max(0, self._size - file_size)
                removed += 1
        return removed

    def _trim(self) -> None:
        """Evict oldest chunks until the configured size limit is met."""
        if self.max_bytes is None or self._size <= self.max_bytes:
            return
        files: list[tuple[float, int, Path]] = []
        for path in self.directory.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
        files.sort()
        for _, size, path in files:
            if self._size <= self.max_bytes:
                break
            try:
                path.unlink()
            except OSError:
                continue
            self._size = max(0, self._size - size)

    def clear(self) -> int:
        """Delete every cached chunk and return the number removed."""
        if not self.directory.exists():
            return 0
        count = sum(
            1 for path in self.directory.rglob("*") if path.is_file()
        )
        shutil.rmtree(self.directory, ignore_errors=True)
        self._size = 0
        return count

    def prune(self, older_than_days: int = 30) -> int:
        """Delete cached chunks older than ``older_than_days``."""
        if older_than_days < 0:
            raise ValueError("older_than_days must be non-negative")
        cutoff = time.time() - older_than_days * 86400
        removed = 0
        if not self.directory.exists():
            return 0
        for path in self.directory.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
                if stat.st_mtime < cutoff:
                    path.unlink()
                    self._size = max(0, self._size - stat.st_size)
                    removed += 1
            except OSError:
                pass
        return removed

    def info(self) -> pd.DataFrame:
        """Return a per-variable summary of the chunk cache."""
        rows: list[dict] = []
        if self.directory.exists():
            for variable_dir in sorted(self.directory.iterdir()):
                if not variable_dir.is_dir():
                    continue
                files = [
                    path for path in variable_dir.rglob("*") if path.is_file()
                ]
                if not files:
                    continue
                sizes = []
                mtimes = []
                for path in files:
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    sizes.append(stat.st_size)
                    mtimes.append(stat.st_mtime)
                rows.append(
                    {
                        "variable": variable_dir.name,
                        "chunks": len(sizes),
                        "size_mb": round(sum(sizes) / 1_000_000, 3),
                        "oldest": datetime.fromtimestamp(
                            min(mtimes), timezone.utc
                        ).isoformat()
                        if mtimes
                        else None,
                        "newest": datetime.fromtimestamp(
                            max(mtimes), timezone.utc
                        ).isoformat()
                        if mtimes
                        else None,
                    }
                )
        return pd.DataFrame(
            rows, columns=["variable", "chunks", "size_mb", "oldest", "newest"]
        )


class CachingRequestsStore(Store):
    """Zarr store that caches raw chunks from a :class:`RequestsStore`."""

    def __init__(self, remote: RequestsStore, cache: ChunkCache) -> None:
        """Wrap ``remote`` with the local ``cache``."""
        self.remote = remote
        self.cache = cache

    def __getitem__(self, key: str) -> bytes:
        """Return bytes for ``key``, using or filling the chunk cache."""
        if self.cache.is_chunk_key(key):
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            data = self.remote[key]
            self.cache.put(key, data)
            return data
        return self.remote[key]

    def getitems(self, keys, *, contexts=None):
        """Read several keys, using the chunk cache where possible."""
        result: dict[str, bytes] = {}
        chunk_keys: list[str] = []
        other_keys: list[str] = []
        for key in keys:
            if self.cache.is_chunk_key(key):
                cached = self.cache.get(key)
                if cached is not None:
                    result[key] = cached
                else:
                    chunk_keys.append(key)
            else:
                other_keys.append(key)
        if other_keys:
            result.update(self.remote.getitems(other_keys, contexts=contexts))
        if chunk_keys:
            fetched = self.remote.getitems(chunk_keys, contexts=contexts)
            for key, data in fetched.items():
                self.cache.put(key, data)
                result[key] = data
        return result

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` exists locally or remotely."""
        if isinstance(key, str) and self.cache.is_chunk_key(key):
            if self.cache.contains(key):
                return True
        return key in self.remote

    def __setitem__(self, key: str, value: bytes) -> None:
        """Reject writes through the Zarr store interface."""
        raise TypeError("CachingRequestsStore is read-only")

    def __delitem__(self, key: str) -> None:
        """Reject deletions through the Zarr store interface."""
        raise TypeError("CachingRequestsStore is read-only")

    def __iter__(self):
        """Delegate key iteration to the remote store."""
        return iter(self.remote)

    def __len__(self) -> int:
        """Delegate key count to the remote store."""
        return len(self.remote)

    def close(self) -> None:
        """Close the remote store if it supports closing."""
        close = getattr(self.remote, "close", None)
        if close is not None:
            close()
