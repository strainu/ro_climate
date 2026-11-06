"""Zip archive helpers for point and area exports."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .variables import ATTRIBUTION

FORMAT_VERSION = 1


def write_archive(
    path: str | Path, data: pd.DataFrame, metadata: dict
) -> Path:
    """Write a data archive atomically.

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
