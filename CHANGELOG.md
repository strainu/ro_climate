# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-09

### Added

- Official ANM daily station observations from the Open Data Portal (ODP):
  `Station`, `list_stations`, `get_station`, `nearest_stations`,
  `nearest_station`, `load_station_series`, and `RoClimateStation`.
- A packaged catalog of the 160 ODP stations, with WMO identifier, name,
  latitude, longitude, and kind.
- `mean_temp`, `wind_speed`, and `pressure` station variables, plus ODP code
  aliases (`ta`, `ff`, `p`) alongside `tx`, `tn`, and `r`.
- Persistent per-station year-archive cache with immutable past years, a
  current-year TTL, stale-on-error fallback, and `clear_station_cache()`.
- `python -m ro_climate.stations --list|--near [--json]` command-line
  interface.
- Offline tests with synthetic ODP ZIP archives and a fake HTTP session.

### Changed

- `RoClimateStation` mirrors the point reduction, aggregation, index, and
  export API. `.tmean` uses the observed `ta` value when available.

### Fixed

- ODP `-999` sentinels are treated as missing; `r=-1` is recognized as valid
  trace precipitation, stored as `0.0` for correct sums, and exposed through
  `RoClimateStation.precipitation_trace`.

## [0.1.0] - 2026-11-07

### Added

- Point time series and area summaries from the ANM daily gridded
  meteorological cube for Romania.
- `RoClimate`, `RoClimatePoint`, and `RoClimateArea` public API.
- Statistics, resampling, climatologies, climate indices, and narrative
  climate-description helpers.
- Chunk-granular area reads and a persistent raw Zarr chunk cache.
- CSV, JSON, NPZ, and zip exports.
- Offline synthetic test suite.

[Unreleased]: https://github.com/strainu/ro_climate/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/strainu/ro_climate/releases/tag/v0.2.0
[0.1.0]: https://github.com/strainu/ro_climate/releases/tag/v0.1.0
