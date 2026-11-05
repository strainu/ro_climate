"""Local zip cache for downloaded ANM point time series."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .variables import ATTRIBUTION

FORMAT_VERSION = 1


def default_cache_dir() -> Path:
    """Return the platform-appropriate default cache directory."""

    override = os.environ.get("RO_CLIMATE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "ro_climate" / "cache" / "points"
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "ro_climate" / "points"
    return Path.home() / ".cache" / "ro_climate" / "points"


@dataclass(frozen=True)
class CacheKey:
    """Identity of a cached point/variable set."""

    store_url: str
    i: int
    j: int
    variables: tuple[str, ...]
    lon: float
    lat: float

    @property
    def expected_columns(self) -> tuple[str, ...]:
        """Return the data columns stored for this key."""
        return self.variables

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 digest for this cache identity."""
        payload = json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "store_url": self.store_url,
                "i": self.i,
                "j": self.j,
                "variables": list(self.variables),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AreaCacheKey:
    """Identity of a cached area summary."""

    store_url: str
    i0: int
    i1: int
    j0: int
    j1: int
    variables: tuple[str, ...]
    reductions: tuple[str, ...]
    mask_digest: str = ""
    min_count: int | None = 1

    @property
    def expected_columns(self) -> tuple[str, ...]:
        """Return the ``{variable}_{reduction}`` columns for this key."""
        return tuple(
            f"{variable}_{reduction}"
            for variable in self.variables
            for reduction in self.reductions
        )

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 digest for this area cache identity."""
        payload = json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "kind": "area",
                "store_url": self.store_url,
                "i0": self.i0,
                "i1": self.i1,
                "j0": self.j0,
                "j1": self.j1,
                "variables": list(self.variables),
                "reductions": list(self.reductions),
                "mask_digest": self.mask_digest,
                "min_count": self.min_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CachedPoint:
    """A point loaded from a local archive."""

    data: pd.DataFrame
    metadata: dict
    path: Path


def write_archive(
    path: str | Path, data: pd.DataFrame, metadata: dict
) -> Path:
    """Write a point archive atomically.

    The archive contains ``data.npy`` (float32), ``dates.npy`` (datetime64[D]),
    ``metadata.json`` and ``README.txt``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.setdefault("format_version", FORMAT_VERSION)
    metadata.setdefault("column_order", list(data.columns))
    metadata.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
    metadata.setdefault("attribution", ATTRIBUTION)

    values = data.to_numpy(dtype="float32", copy=True)
    dates = data.index.values.astype("datetime64[D]")

    values_buf = io.BytesIO()
    np.save(values_buf, values, allow_pickle=False)
    dates_buf = io.BytesIO()
    np.save(dates_buf, dates, allow_pickle=False)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.npy", values_buf.getvalue())
            zf.writestr("dates.npy", dates_buf.getvalue())
            zf.writestr(
                "metadata.json",
                json.dumps(metadata, indent=2, ensure_ascii=False),
            )
            zf.writestr(
                "README.txt",
                "ANM daily gridded meteorological data\n"
                "=====================================\n\n"
                + str(metadata.get("attribution", ATTRIBUTION))
                + "\n\n"
                + str(metadata.get("store_url", ""))
                + "\n",
            )
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


class RoClimateCache:
    """Read/write manager for the local zip cache."""

    def __init__(self, directory: str | Path | None = None) -> None:
        """Create the cache manager and ensure its directory exists.

        Parameters
        ----------
        directory:
            Cache directory. Defaults to :func:`default_cache_dir`.
        """
        self.directory = (
            Path(directory).expanduser() if directory else default_cache_dir()
        )
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: CacheKey | AreaCacheKey) -> Path:
        """Return the archive path for ``key``.

        Point filenames contain the grid centre; area filenames contain their
        grid window. Both end in a short digest so they remain unique and
        human-readable.
        """
        var_part = "-".join(v.replace("_", "-") for v in key.variables)
        if isinstance(key, AreaCacheKey):
            red_part = "-".join(key.reductions)
            return self.directory / (
                f"ro_climate_area_i{key.i0}-{key.i1}_j{key.j0}-{key.j1}_"
                f"{var_part}_{red_part}_{key.digest[:12]}.zip"
            )
        return self.directory / (
            f"ro_climate_lat{key.lat:.4f}_lon{key.lon:.4f}_{var_part}_"
            f"{key.digest[:12]}.zip"
        )

    def load(self, key: CacheKey | AreaCacheKey) -> CachedPoint | None:
        """Load a cached archive, returning ``None`` on a miss or bad archive.

        Invalid, corrupt, or incompatible archives are deleted so the next
        request can replace them.
        """
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            with zipfile.ZipFile(path, "r") as zf:
                metadata = json.loads(zf.read("metadata.json"))
                if int(metadata.get("format_version", 0)) != FORMAT_VERSION:
                    raise ValueError("unsupported cache format")
                kind = metadata.get("kind", "point")
                if isinstance(key, AreaCacheKey):
                    if kind != "area":
                        raise ValueError("not an area archive")
                elif kind != "point":
                    raise ValueError("not a point archive")
                columns = list(metadata.get("column_order", []))
                if columns != list(key.expected_columns):
                    raise ValueError("cached columns do not match key")
                values = np.load(
                    io.BytesIO(zf.read("data.npy")), allow_pickle=False
                )
                dates = np.load(
                    io.BytesIO(zf.read("dates.npy")), allow_pickle=False
                )
            if values.ndim != 2 or values.shape[1] != len(columns):
                raise ValueError("invalid data shape")
            if dates.ndim != 1 or len(dates) != values.shape[0]:
                raise ValueError("invalid dates shape")
            index = pd.DatetimeIndex(dates).as_unit("ns")
            if len(index) > 2:
                freq = pd.infer_freq(index)
                if freq is not None:
                    index.freq = freq
            data = pd.DataFrame(
                values.astype("float32", copy=False),
                index=index,
                columns=columns,
            )
            return CachedPoint(data=data, metadata=metadata, path=path)
        except (
            OSError,
            ValueError,
            KeyError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ):
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def save(
        self,
        key: CacheKey | AreaCacheKey,
        data: pd.DataFrame,
        metadata: dict,
    ) -> Path:
        """Write ``data`` and ``metadata`` to the archive for ``key``."""
        return write_archive(self.path_for(key), data, metadata)

    def info(self) -> pd.DataFrame:
        """Return a table describing the archives in the cache directory.

        Corrupt archives are included with ``variables`` set to
        ``"<corrupt>"``.
        """
        rows: list[dict] = []
        for path in sorted(self.directory.glob("*.zip")):
            row = {
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1_000_000, 3),
                "kind": "point",
                "i": None,
                "j": None,
                "i0": None,
                "i1": None,
                "j0": None,
                "j1": None,
                "variables": None,
                "reductions": None,
                "start": None,
                "end": None,
                "retrieved_at": None,
            }
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    meta = json.loads(zf.read("metadata.json"))
                kind = meta.get("kind", "point")
                grid = meta.get("grid", {})
                actual = meta.get("actual", {})
                row.update(
                    {
                        "kind": kind,
                        "variables": ",".join(
                            meta.get("variables", meta.get("column_order", []))
                        ),
                        "start": actual.get("start"),
                        "end": actual.get("end"),
                        "retrieved_at": meta.get("retrieved_at"),
                    }
                )
                if kind == "area":
                    row.update(
                        {
                            "i0": grid.get("i0"),
                            "i1": grid.get("i1"),
                            "j0": grid.get("j0"),
                            "j1": grid.get("j1"),
                            "reductions": ",".join(
                                meta.get("reductions", [])
                            ),
                        }
                    )
                else:
                    row.update({"i": grid.get("i"), "j": grid.get("j")})
            except (OSError, KeyError, ValueError, zipfile.BadZipFile):
                row["variables"] = "<corrupt>"
            rows.append(row)
        if not rows:
            return pd.DataFrame(
                columns=[
                    "path",
                    "size_mb",
                    "kind",
                    "i",
                    "j",
                    "i0",
                    "i1",
                    "j0",
                    "j1",
                    "variables",
                    "reductions",
                    "start",
                    "end",
                    "retrieved_at",
                ]
            )
        return pd.DataFrame(rows)

    def clear(self) -> int:
        """Delete every cached archive and return the number removed."""
        paths = list(self.directory.glob("*.zip"))
        for path in paths:
            try:
                path.unlink()
            except OSError:
                pass
        return len(paths)

    def prune(self, older_than_days: int = 30) -> int:
        """Delete archives older than ``older_than_days``.

        Return the number of archives removed.
        """
        if older_than_days < 0:
            raise ValueError("older_than_days must be non-negative")
        cutoff = (
            datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        )
        removed = 0
        for path in self.directory.glob("*.zip"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                pass
        return removed
