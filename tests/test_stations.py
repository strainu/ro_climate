"""Offline tests for the ODP station catalog and loader."""

from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
import requests

from ro_climate import (
    RoClimateStation,
    Station,
    clear_station_cache,
    default_station_cache_dir,
    get_station,
    list_stations,
    load_station_series,
    nearest_station,
    nearest_stations,
)
from ro_climate.exceptions import (
    InvalidTimeRangeError,
    NoDataError,
    NonAdditiveVariableError,
    StationNotFoundError,
    UnknownVariableError,
)
from ro_climate.stations import (
    ODP_BASE_URL,
    main,
    resolve_station_cache_ttl,
)

WMO = "15014"
PREAMBLE = [
    "Acest set de date oferă date climatice zilnice.",
    "Furnizor: Administrația Națională de Meteorologie (ANM, România)",
    "URL Oficial: odp.meteoromania.ro/edrapi",
    "E-mail de Contact: odp@meteoromania.ro",
    "WSI  - Identificator statie.",
    "an,luna,zi - ziua climatologica.",
    "ff   - Viteza medie a vântului.",
    "p    - Presiunea medie la nivelul mării.",
    "r    - Cantitatea de precipitații.",
    "ta   - Temperatura medie.",
    "tn   - Temperatura minimă.",
    "tx   - Temperatura maximă.",
]
HEADER = "wsi,an,luna,zi,ff,p,r,ta,tn,tx"


class FakeResponse:
    """Minimal stand-in for a ``requests.Response``."""

    def __init__(self, status_code: int, content: bytes = b"") -> None:
        """Store the status code and body."""

        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        """Raise ``requests.HTTPError`` for error status codes."""

        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """HTTP session that serves in-memory ZIP payloads."""

    def __init__(self, payloads=None, *, error=None) -> None:
        """Create a session from URL-to-bytes mappings."""

        self.payloads = dict(payloads or {})
        self.error = error
        self.calls: list[str] = []
        self.closed = False

    def get(self, url, timeout=None):
        """Return a synthetic response for ``url``."""

        self.calls.append(url)
        if self.error is not None:
            raise self.error
        payload = self.payloads.get(url)
        if payload is None:
            return FakeResponse(404)
        return FakeResponse(200, payload)

    def close(self) -> None:
        """Record that the session was closed."""

        self.closed = True


def _csv(wmo: str, year: int, month: int, rows: dict) -> bytes:
    """Build one synthetic ODP monthly CSV file."""

    lines = list(PREAMBLE) + [HEADER]
    for day, values in rows.items():
        fields = [f"0-20000-0-{wmo}", str(year), str(month), str(day)]
        fields.extend(
            values.get(key, "")
            for key in ("ff", "p", "r", "ta", "tn", "tx")
        )
        lines.append(",".join(fields))
    return ("\r\n".join(lines) + "\r\n").encode("utf-8-sig")


def _year_zip(wmo: str, year: int, months: dict) -> bytes:
    """Build a synthetic ODP year ZIP with one CSV per month."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for month, rows in months.items():
            name = f"0-20000-0-{wmo}_{year}_{month:02d}.csv"
            zf.writestr(name, _csv(wmo, year, month, rows))
    return buffer.getvalue()


def _url(wmo: str, year: int) -> str:
    """Return the ODP archive URL used by the loader."""

    return (
        f"{ODP_BASE_URL}/0-20000-0-{wmo}/"
        f"0-20000-0-{wmo}_{year}.zip"
    )


@pytest.fixture
def payload() -> bytes:
    """Return a synthetic 2020 archive for Baia Mare."""

    return _year_zip(
        WMO,
        2020,
        {
            1: {
                1: {
                    "ff": "2.5",
                    "p": "1010.0",
                    "r": "-1",
                    "ta": "-1.0",
                    "tn": "-2.5",
                    "tx": "1.0",
                },
                2: {
                    "ff": "-999",
                    "p": "-999",
                    "r": "-999",
                    "ta": "-999",
                    "tn": "-999",
                    "tx": "-999",
                },
                3: {
                    "ff": "3.0",
                    "p": "1000.0",
                    "r": "",
                    "ta": "0.5",
                    "tn": "-1.0",
                    "tx": "2.0",
                },
                # January 4 is intentionally absent from the CSV.
                5: {
                    "ff": "4.0",
                    "p": "999.0",
                    "r": "0.0",
                    "ta": "1.0",
                    "tn": "0.0",
                    "tx": "2.0",
                },
            },
            2: {
                1: {
                    "ff": "1.0",
                    "p": "1005.0",
                    "r": "2.0",
                    "ta": "3.0",
                    "tn": "2.0",
                    "tx": "4.0",
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# Catalog and geographic lookup
# ---------------------------------------------------------------------------
def test_catalog_lookup() -> None:
    """The packaged catalog contains all ODP stations and names resolve."""

    stations = list_stations()
    assert len(stations) == 160
    assert all(isinstance(station, Station) for station in stations)
    assert get_station("15014") == get_station(15014)
    assert get_station("baia mare").name == "Baia Mare"
    assert get_station("BAIA MARE").wmo == "15014"
    assert get_station("București Filaret").wmo == "15422"
    assert get_station(get_station("15014")).wmo == "15014"
    assert list_stations(kind="meteorological") == stations
    assert list_stations(kind="pluviometric") == []
    with pytest.raises(StationNotFoundError):
        get_station("does-not-exist")
    with pytest.raises(StationNotFoundError):
        get_station("")


def test_nearest_stations() -> None:
    """Nearby-station lookup is sorted and understands distances."""

    ranked = nearest_stations(26.1025, 44.4325, 3)
    assert [station.wmo for station, _ in ranked] == [
        "15422",
        "15420",
        "15421",
    ]
    assert ranked[0][1] < ranked[1][1] < ranked[2][1]
    assert nearest_station(26.1025, 44.4325).wmo == "15422"
    filaret = get_station("15422")
    assert filaret.distance_km(26.0938, 44.4121) < 0.01
    assert (
        len(nearest_stations(26.1025, 44.4325, 10, max_distance_km=10))
        == 2
    )
    with pytest.raises(ValueError):
        nearest_stations(float("nan"), 44.0)
    with pytest.raises(ValueError):
        nearest_stations(26.0, 44.0, 0)
    with pytest.raises(ValueError):
        nearest_stations(26.0, 44.0, max_distance_km=-1)
    with pytest.raises(StationNotFoundError):
        nearest_station(26.0, 44.0, max_distance_km=0)
    with pytest.raises(StationNotFoundError):
        nearest_station(26.0, 44.0, kind="pluviometric")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def test_load_station_series(tmp_path, payload) -> None:
    """The loader parses ODP CSVs, filters sentinels, and fills date gaps."""

    session = FakeSession({_url(WMO, 2020): payload})
    series = load_station_series(
        WMO,
        start="2020-01-01",
        end="2020-02-02",
        cache_dir=tmp_path,
        session=session,
    )

    assert isinstance(series, RoClimateStation)
    assert series.wmo == WMO
    assert series.name == "Baia Mare"
    assert series.data.columns.tolist() == [
        "max_temp",
        "min_temp",
        "precipitation",
    ]
    assert len(series) == 32
    assert series.data.index.equals(
        pd.date_range(
            "2020-01-01", "2020-02-01", freq="D", name="date"
        )
    )
    assert series.data.loc["2020-01-01", "precipitation"] == pytest.approx(
        0.0
    )
    assert bool(series.precipitation_trace.loc["2020-01-01"]) is True
    assert int(series.precipitation_trace.sum()) == 1
    assert np.isnan(series.data.loc["2020-01-02", "max_temp"])
    assert np.isnan(series.data.loc["2020-01-03", "precipitation"])
    assert np.isnan(series.data.loc["2020-01-04", "max_temp"])
    assert series.data.loc["2020-02-01", "max_temp"] == pytest.approx(4.0)
    assert series.frontiers["max_temp"] == pd.Timestamp("2020-02-01")
    assert series.warnings == ()
    assert session.calls == [_url(WMO, 2020)]


def test_cache_reuses_archive(tmp_path, payload) -> None:
    """A second request is served entirely from the local cache."""

    first = FakeSession({_url(WMO, 2020): payload})
    original = load_station_series(
        WMO,
        start="2020-01-01",
        end="2020-02-02",
        cache_dir=tmp_path,
        session=first,
    )
    assert len(first.calls) == 1
    assert (tmp_path / WMO / "2020.zip").is_file()

    second = FakeSession()
    cached = load_station_series(
        WMO,
        start="2020-01-01",
        end="2020-02-02",
        cache_dir=tmp_path,
        session=second,
    )
    assert second.calls == []
    pd.testing.assert_frame_equal(original.data, cached.data)


def test_missing_year_is_cached(tmp_path) -> None:
    """A 404 is remembered so repeated requests do not retry it."""

    first = FakeSession()
    with pytest.raises(NoDataError):
        load_station_series(
            WMO,
            start="2020-01-01",
            end="2020-01-02",
            cache_dir=tmp_path,
            session=first,
        )
    assert first.calls == [_url(WMO, 2020)]
    assert (tmp_path / WMO / "2020.missing").is_file()

    second = FakeSession()
    with pytest.raises(NoDataError):
        load_station_series(
            WMO,
            start="2020-01-01",
            end="2020-01-02",
            cache_dir=tmp_path,
            session=second,
        )
    assert second.calls == []


def test_stale_cache_is_used_after_download_error(tmp_path) -> None:
    """An expired current-year archive still serves reads after an error."""

    year = pd.Timestamp.now().year
    payload = _year_zip(
        WMO,
        year,
        {1: {1: {"tx": "1.0", "tn": "0.0", "r": "0.0"}}},
    )
    original = load_station_series(
        WMO,
        start=f"{year}-01-01",
        end=f"{year}-01-02",
        cache_dir=tmp_path,
        session=FakeSession({_url(WMO, year): payload}),
    )
    assert original.start == pd.Timestamp(f"{year}-01-01")

    archive = tmp_path / WMO / f"{year}.zip"
    old = archive.stat().st_mtime - 600
    os.utime(archive, (old, old))

    failing = FakeSession(error=requests.ConnectionError("offline"))
    series = load_station_series(
        WMO,
        start=f"{year}-01-01",
        end=f"{year}-01-02",
        cache_dir=tmp_path,
        cache_ttl=1,
        session=failing,
    )
    assert failing.calls == [_url(WMO, year)]
    assert any("using cached" in warning for warning in series.warnings)


def test_variable_aliases_and_tmean(tmp_path, payload) -> None:
    """ODP short codes are accepted and map to public column names."""

    session = FakeSession({_url(WMO, 2020): payload})
    series = load_station_series(
        WMO,
        start="2020-01-01",
        end="2020-01-05",
        variables=("ta", "ff", "p"),
        cache_dir=tmp_path,
        session=session,
    )
    assert series.data.columns.tolist() == [
        "mean_temp",
        "wind_speed",
        "pressure",
    ]
    assert series.data.loc["2020-01-01", "mean_temp"] == pytest.approx(
        -1.0
    )
    assert series.tmean.iloc[0] == pytest.approx(-1.0)
    assert series.mean("ta") == pytest.approx(0.5 / 3.0)
    with pytest.raises(UnknownVariableError):
        series.diurnal_range
    with pytest.raises(UnknownVariableError):
        series.precipitation_trace
    with pytest.raises(UnknownVariableError):
        load_station_series(
            WMO,
            start="2020-01-01",
            end="2020-01-05",
            variables=("bogus",),
            cache_dir=tmp_path,
            session=session,
        )
    with pytest.raises(ValueError):
        load_station_series(
            WMO,
            start="2020-01-01",
            end="2020-01-05",
            variables=(),
            cache_dir=tmp_path,
            session=session,
        )


def test_statistics_and_exports(tmp_path, payload) -> None:
    """The station object mirrors the point reduction and export API."""

    series = load_station_series(
        WMO,
        start="2020-01-01",
        end="2020-02-02",
        cache_dir=tmp_path,
        session=FakeSession({_url(WMO, 2020): payload}),
    )

    assert series.sum() == pytest.approx(2.0)
    assert series.max_daily_precip() == pytest.approx(2.0)
    assert series.max("tx") == pytest.approx(4.0)
    assert series.max("max_temp") == pytest.approx(4.0)
    with pytest.raises(NonAdditiveVariableError):
        series.sum("max_temp")
    assert series.sum(
        "max_temp", allow_non_additive=True
    ) == pytest.approx(9.0)
    assert series.count("max_temp") == 4
    assert series.coverage.loc["precipitation", "n_missing"] == 29
    assert "sum" in series.describe().columns

    monthly = series.monthly_total()
    assert monthly.index[0] == pd.Timestamp("2020-01-01")
    assert monthly.iloc[0, 0] == pytest.approx(0.0)
    annual = series.annual_total()
    assert annual.iloc[0, 0] == pytest.approx(2.0)
    assert series.extreme_dates().loc["max_temp", "max"] == pytest.approx(
        4.0
    )
    assert series.precipitation_summary()["wettest"][0] == 2020
    assert np.isfinite(series.mean_temperature())
    assert series.frost_days() == 2
    assert series.rain_days(0.0) == 3
    assert series.hot_days(3.0) == 1

    subset = series.subset(start="2020-01-01", end="2020-01-31")
    assert isinstance(subset, RoClimateStation)
    assert len(subset) == 31
    assert bool(subset.precipitation_trace.loc["2020-01-01"]) is True
    with pytest.raises(NoDataError):
        series.subset(start="2021-01-01", end="2021-01-02")
    with pytest.raises(UnknownVariableError):
        series["pressure"]

    csv_path = series.to_csv(tmp_path / "station.csv")
    assert csv_path.is_file()
    frame = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    assert frame.columns.tolist() == [
        "max_temp",
        "min_temp",
        "precipitation",
    ]
    assert series.to_json(tmp_path / "station.json").is_file()
    assert series.to_npz(tmp_path / "station.npz").is_file()

    archive = series.to_zip(tmp_path / "station.zip")
    with zipfile.ZipFile(archive) as zf:
        metadata = json.loads(zf.read("metadata.json"))
        assert metadata["station"]["wmo"] == WMO
        assert metadata["source"] == "ANM ODP station observations"
        assert metadata["sentinel_values"]["missing"] == -999.0
        assert metadata["sentinel_values"]["trace_precipitation"] == -1.0
        assert (
            metadata["sentinel_values"]["trace_precipitation_stored_as"]
            == 0.0
        )
        assert metadata["trace_precipitation_days"] == 1
        assert b"odp.meteoromania.ro" in zf.read("README.txt")
        assert not any(name.endswith(".csv") for name in zf.namelist())


def test_future_end_is_clamped(tmp_path) -> None:
    """An end beyond today is clamped and reported, and years stop there."""

    year = pd.Timestamp.now().year
    payload = _year_zip(
        WMO,
        year,
        {
            1: {
                1: {
                    "tx": "1.0",
                    "tn": "0.0",
                    "r": "0.0",
                }
            }
        },
    )
    session = FakeSession({_url(WMO, year): payload})
    series = load_station_series(
        WMO,
        start=f"{year}-01-01",
        end="2999-12-31",
        cache_dir=tmp_path,
        session=session,
    )
    assert any("future" in warning for warning in series.warnings)
    assert series.end == pd.Timestamp(f"{year}-01-01")
    assert session.calls == [_url(WMO, year)]


def test_invalid_ranges_and_unknown_station(tmp_path) -> None:
    """Invalid date ranges and unknown stations fail before any download."""

    with pytest.raises(InvalidTimeRangeError):
        load_station_series(
            WMO,
            start="2020-02-01",
            end="2020-01-01",
            cache_dir=tmp_path,
            session=FakeSession(),
        )
    with pytest.raises(NoDataError):
        load_station_series(
            WMO,
            start="2999-01-01",
            end="2999-12-31",
            cache_dir=tmp_path,
            session=FakeSession(),
        )
    with pytest.raises(StationNotFoundError):
        load_station_series(
            "99999",
            start="2020-01-01",
            end="2020-12-31",
            cache_dir=tmp_path,
            session=FakeSession(),
        )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def test_cache_helpers(tmp_path, monkeypatch) -> None:
    """Default paths, TTL overrides, and cache clearing work offline."""

    monkeypatch.setenv("RO_CLIMATE_CACHE_DIR", str(tmp_path))
    assert default_station_cache_dir() == tmp_path / "stations"
    monkeypatch.setenv("RO_CLIMATE_STATION_CACHE_TTL", "42")
    assert resolve_station_cache_ttl() == 42.0
    assert resolve_station_cache_ttl(10) == 10.0
    with pytest.raises(ValueError):
        resolve_station_cache_ttl(-1)

    station_dir = tmp_path / "15014"
    station_dir.mkdir()
    (station_dir / "2020.zip").write_bytes(b"x")
    (station_dir / "2021.missing").write_bytes(b"")
    assert clear_station_cache(tmp_path) == 2
    assert not tmp_path.exists()
    assert clear_station_cache(tmp_path) == 0


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def test_cli_list_json(capsys) -> None:
    """``--list --json`` emits the complete packaged catalog."""

    assert main(["--list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 160
    assert payload[0]["wmo"] == "15000"
    assert payload[0]["kind"] == "meteorological"


def test_cli_list_table(capsys) -> None:
    """``--list`` renders a human-readable table."""

    assert main(["--list", "--kind", "meteorological"]) == 0
    output = capsys.readouterr().out
    assert "WMO" in output
    assert "15014" in output
    assert "Baia Mare" in output


def test_cli_near_json(capsys) -> None:
    """``--near --json`` returns distance-sorted stations."""

    assert (
        main(
            [
                "--near",
                "26.1025",
                "44.4325",
                "--limit",
                "3",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert [item["wmo"] for item in payload] == [
        "15422",
        "15420",
        "15421",
    ]
    assert payload[0]["distance_km"] > 0
