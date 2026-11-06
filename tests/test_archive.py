import json
import zipfile

import numpy as np
import pandas as pd

from ro_climate.archive import FORMAT_VERSION, write_archive


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


def test_write_archive_roundtrip(tmp_path):
    path = write_archive(tmp_path / "archive.zip", make_frame(), make_metadata())
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


