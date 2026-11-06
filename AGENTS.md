# AGENTS.md

## Project scope

`ro-climate` (import name `ro_climate`) is a standalone Python library for
point time series and area summaries from the public **ANM daily gridded
meteorological cube** for Romania.

Work only inside this `meteo/` directory. Do not navigate to, import from, or
depend on the parent `pywikibot` checkout.

## Naming

- Distribution: `ro-climate`
- Package/import: `ro_climate`
- Public classes: `RoClimate`, `RoClimatePoint`, `RoClimateArea`,
  `RoClimateError`

## Setup and verification

Requires Python >= 3.10.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Focused run while iterating:

```bash
python -m pytest tests/test_point.py -q
```

Packaging check after changing `pyproject.toml`:

```bash
python -m pip install --dry-run --no-deps .
```

Tests must pass without network access.

## Git workflow

- This `meteo/` directory is the library's own Git repository. Do not treat
  the parent `pywikibot` checkout as the library repository.
- Make every commit with `git commit -s` so it carries the DCO sign-off.
- Do not push to any remote without explicit confirmation from the user.

## Layout

| Path | Responsibility |
| --- | --- |
| `ro_climate/client.py` | `RoClimate`: STAC discovery, Zarr access, frontiers, point/area loading |
| `ro_climate/point.py` | `RoClimatePoint`: data access, statistics, resampling, indices, exports |
| `ro_climate/area.py` | `RoClimateArea`: spatial-reduction summaries, statistics, aggregation, exports |
| `ro_climate/archive.py` | Atomic zip archive writer used by exports |
| `ro_climate/_chunk_cache.py` | Private raw Zarr chunk cache and caching store wrapper |
| `ro_climate/_reduce.py` | Shared temporal reduction helpers |
| `ro_climate/_store.py` | Private read-only `RequestsStore` for Zarr v2 over HTTP |
| `ro_climate/grid.py` | 0.01° grid mapping, bbox/polygon helpers, date conversion, season helpers |
| `ro_climate/variables.py` | Variable metadata, default order, attribution |
| `ro_climate/indices.py` | Pure helpers for climate-index calculations |
| `ro_climate/exceptions.py` | Exception hierarchy rooted at `RoClimateError` |
| `ro_climate/__init__.py` | Public API exports and `__all__` |
| `tests/` | Offline synthetic tests |

## Hard constraints

- Runtime dependencies are only `zarr>=2.18,<3`, `numpy`, `pandas`, and
  `requests`. Do not add `xarray`, `fsspec`, `aiohttp`, or plotting libraries.
- Zarr v2 only. Keep `RequestsStore` private, small, and read-only.
- Every function and method needs a docstring, including private helpers,
  properties, constructors, and dunder methods.
- No plotting. Exports are CSV, JSON, NPZ, and zip only.
- Keep public names exported from `ro_climate/__init__.py` and listed in
  `__all__` when public.
- Do not write CSV files into export archives.

## Behavior contract

- `point(lon, lat, start=None, end=None, variables=None, ...)` snaps the
  coordinate to the containing 0.01° cell and returns `RoClimatePoint`.
- Default `start` is 1961-01-01. Default `end` is the common frontier of the
  requested variables.
- Frontiers come from the authoritative `last_<var>_time` scalars, not STAC
  `end_datetime`, which can be stale.
- An `end` beyond the frontier is clamped and reported in `point.warnings`.
- Statistics (`mean`/`avg`/`min`/`max`/`median`/`std`/`var`/`quantile`/
  `count`/`describe`) apply to all variables.
- `sum()` and `resample(..., how="sum")` are only valid for additive variables
  (precipitation). Temperature raises `NonAdditiveVariableError` unless
  `allow_non_additive=True` is passed to `sum()`.
- `area(bounds=(west, south, east, north), ..., polygon=...)` returns a
  `RoClimateArea` with daily spatial summaries, not raw grids. When `bounds` is
  omitted, the polygon bounding box is used; a supplied polygon masks cells
  inside that window.
- Spatial reductions are `mean` (default), `min`, `max`, `sum`, and `count`.
  Columns are `{variable}_{reduction}`. `min_count` defaults to 1 and applies to
  non-count reductions. Spatial `sum` is limited to additive variables unless
  `allow_non_additive=True`.
- Polygon selection uses cell centres and one simple ring; holes,
  multipolygons, and exact area weighting are not supported.
- Area downloads are chunk-granular: only chunks containing at least one
  selected cell are read. Time reads are aligned to Zarr time chunks.
- Bounds outside the grid are clipped with an `RoClimateArea.warnings` entry;
  an `end` beyond the frontier is clamped with a warning.
- Point series and area summaries are not archived; both are rebuilt from
  raw Zarr chunks on every request. The only cache is the raw chunk cache.
- The raw chunk cache is enabled by default and lives under
  `<cache root>/chunks/<store digest>/`, keyed by store/variable/time/y/x.
  Its limit defaults to 20 GB, overridable with `chunk_cache_max_mb` or
  `RO_CLIMATE_CHUNK_CACHE_MAX_MB`; `0` disables the limit. Oldest chunks are
  evicted first. `chunk_cache_info()` lists chunks, and
  `clear_cache()`/`prune_cache()` manage them.
- `point(refresh=True)` and `area(refresh=True)` invalidate cached chunks
  covering the requested range.
- Export archives (`to_zip`) contain `data.npy`, `dates.npy`,
  `metadata.json`, `README.txt`. Writes must remain atomic.
- Cache directory override: `RO_CLIMATE_CACHE_DIR`.

## Data source facts

- Zarr store: `https://storage.svc.uvt-01.eocube.ro/meteo/cube/open-v1/`
- STAC item:
  `https://stac.eocube.ro/collections/anm-daily-grids/items/anm-daily-grids-open-v1`
- Variables: `max_temp` (°C), `min_temp` (°C), `precipitation` (mm)
- Grid: 0.01° cells; x centres 20.135°E–29.845°E; y centres 48.355°N–43.535°N
- Licence: CC BY 4.0. Attribution: Administrația Națională de Meteorologie
  (MeteoRomania), via data.gov.ro; cube assembly and hosting by West University
  of Timișoara (ROCS / EOCube.RO).

## Testing rules

- Tests use synthetic in-memory data and fake HTTP sessions; they never contact
  the network.
- Add or update tests for every behavior change.
- Use `tmp_path` for chunk-cache and export tests.
- Area tests use a fake 3-D array with small chunks to verify chunk pruning and
  spatial reductions without network access.
- Chunk-cache tests exercise path naming, invalidation, eviction, and the
  caching store wrapper.
- The store tests monkeypatch `ro_climate._store.time.sleep`; keep retry tests
  fast and deterministic.
- Do not replace tests with live downloads. Live smoke tests are optional and
  manual only.

## Change checklist

1. Update docstrings, README examples, and the README API summary if the public
   API changes.
2. Run `python -m pytest -q`.
3. Verify every function/method has a docstring (AST check).
4. Remove generated `__pycache__`, `.pytest_cache`, `build/`, and `*.egg-info`.
5. If packaging metadata changed, run the `pip install --dry-run` check.

## Known pitfalls

- Do not infer data availability from STAC `end_datetime`.
- Do not mix variable frontiers when multiple variables are requested.
- Select polygon cells by their centres; do not treat polygon vertices as grid
  indices or add exact-area weighting without changing the documented contract.
- Keep chunk-cache file names (`<time>.<y>.<x>` bucketed by `time // 1000`) and
  `ChunkCache.invalidate` aligned.
- Keep the distinction: `ro_climate` is the library; `ANM` is the dataset.
