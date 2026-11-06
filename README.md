# ro-climate — ANM daily gridded meteorology

`ro-climate` is a small, object-oriented Python client for the public **ANM
daily gridded meteorological cube** for Romania. It reads single grid cells or
rectangular/polygon areas for an optional date range, returns pandas time
series or daily spatial summaries, and provides statistics, time aggregation,
climate indices, exports, and a local chunk cache.

- Store: `https://storage.svc.uvt-01.eocube.ro/meteo/cube/open-v1/`
- STAC item: `https://stac.eocube.ro/collections/anm-daily-grids/items/anm-daily-grids-open-v1`
- Variables: `max_temp` (°C), `min_temp` (°C), `precipitation` (mm)
- Data licence: CC BY 4.0

Data © Administrația Națională de Meteorologie (MeteoRomania), via
data.gov.ro; cube assembly and hosting by West University of Timișoara
(ROCS / EOCube.RO).

## Installation

From this directory:

```bash
python -m pip install -e .
# with test tools
python -m pip install -e ".[test]"
```

Runtime dependencies are deliberately small: `zarr` (v2), `numpy`, `pandas`,
and `requests`. No `xarray`, `fsspec`, `aiohttp`, or plotting stack is needed.

## Quick start

```python
from ro_climate import RoClimate

with RoClimate() as client:
    point = client.point(
        26.1025,
        44.4325,
        start="2020-01-01",
        end="2020-12-31",
    )
    print(point)
    print(point.mean())                 # mean of each variable
    print(point.max("max_temp"))        # hottest day in the period
    print(point.sum("precipitation"))   # total precipitation
```

> **Performance note.** The cube stores 90-day × 161-row × 324-column chunks,
> so a single point's full 1961–present history requires downloading the
> chunks covering that point for every 90-day period. The first uncached
> full-history request can therefore transfer several gigabytes and take
> minutes per variable. For interactive work, pass a narrower `start`/`end`
> and select only the variables you need. Area requests are chunk-aligned and
> read each intersecting chunk once instead of repeating a point download for
> every cell. Point series and area summaries are not archived; the raw Zarr
> chunk cache (enabled by default) makes later overlapping requests local.

The requested coordinate is snapped to the containing 0.01° grid cell. The
result records the cell indices, the cell centre, and the cell bounds:

```python
print(point.i, point.j)
print(point.grid_lon, point.grid_lat)
print(point.cell_bounds)   # (lon_min, lon_max, lat_min, lat_max)
```

The cube uses a 0.01° grid with cell centres from 20.135°E to 29.845°E and
48.355°N to 43.535°N. Coordinates outside the grid raise `OutsideGridError`.

## Date ranges and frontiers

- Without `start`, a point starts at the first cube day, **1961-01-01**.
- Without `end`, it ends at the latest day that is available for **all**
  requested variables.
- Variable frontiers are read from the authoritative `last_*_time` arrays,
  not from the STAC `end_datetime` field, which can lag behind the data.
- An `end` later than the common frontier is clamped and reported in
  `point.warnings`. Intersecting frontiers avoids returning the zero-filled
  preallocated days beyond each variable's last written value.

```python
point = client.point(26.1025, 44.4325, start="2015-06-01")
print(point.start, point.end, point.warnings)
```

## Statistics

```python
point.mean()                 # pandas Series, one value per loaded variable
point.avg()                  # alias of mean()
point.max()
point.min()
point.median()
point.std()
point.var()
point.quantile(0.9)
point.count()
point.describe()             # count/mean/std/min/... plus missing and sum
point.coverage               # valid/missing counts and fraction per variable
```

`sum()` is only defined for additive variables. It defaults to precipitation
and raises `NonAdditiveVariableError` for temperature:

```python
point.sum()                       # precipitation total
point.sum("precipitation")
point.sum("max_temp")             # raises NonAdditiveVariableError
point.sum("max_temp", allow_non_additive=True)  # explicit override
```

All statistics skip missing values by default (`skipna=True`).

## Time aggregation

```python
point.monthly_mean()              # month-start index
point.monthly_total()             # precipitation totals
point.annual_mean()
point.annual_total()
point.annual_extremes()           # yearly max and min for each variable
point.climatology("month")        # mean by calendar month
point.climatology("dayofyear")    # mean by day of year

point.resample("MS", "mean")      # any pandas offset alias
point.resample("YS", "max")
```

`resample(..., how="sum")` is rejected for temperature, just like `sum()`.

### Filtered queries in one line

`RoClimatePoint.aggregate()` combines date filtering, month/season filtering, and a
reduction. For example, the maximum temperature in January over a date range:

```python
point = client.point(26.1025, 44.4325, start="2010-01-01", end="2020-12-31")
jan_max = point.aggregate("max_temp", "max", months=1)
print(jan_max)

# Same query, one value per year
per_year = point.aggregate("max_temp", "max", months=1, groupby="year")

# Seasons: DJF, MAM, JJA, SON
jja_mean = point.aggregate("max_temp", "mean", season="JJA")
winter = point.aggregate(
    "precipitation", "sum", season="DJF", groupby="season_year"
)
```

`groupby` accepts `"year"`, `"month"`, `"season_year"`, or a pandas offset
alias such as `"YS"` and `"MS"`. `season_year` assigns December to the
following winter.

To keep several variables or apply custom pandas logic, use `subset()`:

```python
jan = point.subset(start="2020-01-01", end="2020-01-31", months=1)
print(jan.data)
```

## Derived values and climate indices

```python
point.tmean                    # (max_temp + min_temp) / 2
point.diurnal_range            # max_temp - min_temp
point.mean_temperature()
point.mean_diurnal_range()

point.frost_days()             # min_temp < 0 °C
point.summer_days()            # max_temp >= 25 °C
point.hot_days()               # max_temp >= 30 °C
point.tropical_nights()        # min_temp >= 20 °C
point.rain_days()              # precipitation >= 1 mm
point.heavy_rain_days()        # precipitation >= 10 mm
point.dry_days()               # precipitation < 1 mm
point.max_daily_precip()
point.max_n_day_precip(5)      # max rolling 5-day precipitation total
point.growing_degree_days(10)  # base 10 °C, mean-temperature method
point.consecutive_dry_days()
point.consecutive_wet_days()
```

Most threshold indices accept a custom `threshold` and a `by` argument:

```python
point.hot_days(30, by="year")
point.rain_days(1.0, by="month")
point.frost_days(0.0, by="season_year")
point.hot_days(30, by="YS")     # any pandas offset alias
```

## Area and grid summaries

`area()` reads a rectangular window and returns **daily spatial summaries**,
not raw grids. Bounds are `(west, south, east, north)` in WGS84 degrees:

```python
area = client.area(
    bounds=(25.90, 44.30, 26.30, 44.60),
    start="2020-01-01",
    end="2020-12-31",
    variables=["max_temp", "precipitation"],
    reductions=["mean", "max"],
)
print(area.data.head())
print(area.mean())            # temporal mean of each daily summary
print(area.series("max_temp", "max"))
```

Supported spatial reductions are `mean` (the default), `min`, `max`, `sum`,
and `count`. Data columns are named `{variable}_{reduction}`, for example
`max_temp_mean` or `precipitation_count`.

Pass `polygon=` to keep only cells whose centres fall inside a simple polygon.
The polygon's bounding box is used as the search window unless `bounds=` is
also given, in which case the polygon acts as a mask within the window:

```python
area = client.area(
    polygon=[
        (26.00, 44.30),
        (26.30, 44.30),
        (26.30, 44.60),
        (26.00, 44.60),
    ],
    start="2020-06-01",
    end="2020-08-31",
)
```

Holes and multipart polygons are not supported. Selection is by cell centre;
there is no exact area weighting.

`min_count` is the minimum number of valid cells required for a non-count
reduction (default `1`). Spatial `sum` is limited to additive variables
(precipitation) unless `allow_non_additive=True` is passed:

```python
dry_cells = client.area(
    bounds=(25.90, 44.30, 26.30, 44.60),
    start="2020-01-01",
    reductions=["mean", "count"],
    min_count=10,
)

rain = client.area(
    bounds=(25.90, 44.30, 26.30, 44.60),
    start="2020-01-01",
    variables="precipitation",
    reductions="sum",
)
```

The returned `RoClimateArea` exposes:

```python
area.data                 # daily summaries, one row per day
area.bounds               # requested west/south/east/north
area.i0, area.i1          # inclusive column indices
area.j0, area.j1          # inclusive row indices
area.n_cells              # selected cells (after polygon masking)
area.shape                # (rows, columns) of the grid window
area.variables, area.reductions, area.columns
area.coverage             # valid cell-days per variable
area.warnings             # clipping and clamping notices
```

`.coverage` covers the requested date range and the selected cells.

It offers the same style of access and aggregation as a point:

```python
area.at("2020-07-01")
area.subset(start="2020-06-01", end="2020-08-31", months=[6, 7, 8])
area["max_temp_mean"]
area[["max_temp_mean", "precipitation_mean"]]
area.mean(), area.max(), area.min(), area.median()
area.std(), area.var(), area.count(), area.quantile(0.9)
area.sum(), area.describe()

area.resample("MS", "mean")
area.monthly_mean(), area.monthly_total()
area.annual_mean(), area.annual_total()
area.climatology("month"), area.climatology("dayofyear")
area.aggregate("max_temp", "max", how="mean", months=1, groupby="year")
```

### Area reads

Area summaries are not stored. Each call reads the chunks intersecting the
selected cells and computes the requested reductions.

## Local chunk cache

The only cache is the raw compressed Zarr chunk cache, enabled by default:

```python
client = RoClimate(
    chunk_cache_dir="ro-climate-cache",
    chunk_cache_max_mb=2048,
)
```

Compressed Zarr chunks are stored under
`<chunk_cache_dir>/chunks/<store digest>/`. They are shared by point and area
requests. The cache is capped at **20 GB** by default. Set
`chunk_cache_max_mb=` or the `RO_CLIMATE_CHUNK_CACHE_MAX_MB` environment
variable to change it; `0` disables the limit. The oldest chunks are evicted
when the cap is exceeded.

```python
from ro_climate import RoClimate

client = RoClimate(chunk_cache_dir="ro-climate-cache")
point = client.point(26.1025, 44.4325, start="2020-01-01", end="2020-12-31")
area = client.area(
    bounds=(25.0, 44.0, 27.0, 45.0), start="2020-01-01", end="2020-12-31"
)

print(client.chunk_cache_info())
client.prune_cache(older_than_days=30)
client.clear_cache()
client.close()
```

Set `RO_CLIMATE_CACHE_DIR` to override the platform cache root and
`RO_CLIMATE_CHUNK_CACHE_MAX_MB` to override the size cap. Pass
`chunk_cache=False` to disable local storage, or `refresh=True` to a `point()`
or `area()` call to invalidate the chunks covering the requested range.

## Exports

```python
point.to_csv("bucharest_2020.csv")
point.to_json("bucharest_2020.json")
point.to_npz("bucharest_2020.npz")
point.to_zip("bucharest_2020.zip")   # same archive layout
print(point.metadata())

area.to_csv("muntenia_2020.csv")
area.to_json("muntenia_2020.json")
area.to_npz("muntenia_2020.npz")
area.to_zip("muntenia_2020.zip")
print(area.metadata())
```

`to_csv()` writes the date index with the column name `date` and formats
floats to four decimals by default. Extra keyword arguments are passed to
pandas, so override `float_format` or use `sep=";"` as needed.

## API summary

`RoClimate(...)`

- `RoClimate(store_url=None, *, discover=True, chunk_cache=True,
  chunk_cache_dir=None, chunk_cache_max_mb=None, timeout=30.0, retries=3,
  frontier_ttl=3600)`
- `.point(lon, lat, start=None, end=None, variables=None, *, refresh=None,
  strict=False) -> RoClimatePoint`
- `.area(bounds=None, start=None, end=None, variables=None, reductions=None,
  *, polygon=None, min_count=1, allow_non_additive=False, refresh=None,
  strict=False) -> RoClimateArea`
- `.frontiers(force=False)` — last written day per variable
- `.chunk_cache_info()`, `.clear_cache()`, `.prune_cache(older_than_days=30)`
- Context manager and `.close()`

`RoClimatePoint`

- Access: `.data`, `.variables`, `.units`, `.start`, `.end`, `.i`, `.j`,
  `.grid_lon`, `.grid_lat`, `.cell_bounds`, `.coverage`, `.warnings`
- Selection: `.at(date)`, `.subset(...)`, `point["max_temp"]`,
  `point[["max_temp", "precipitation"]]`
- Statistics: `.mean/.avg/.max/.min/.median/.std/.var/.quantile/.count/
  .sum/.describe`
- Aggregation: `.resample`, `.monthly_mean`, `.monthly_total`,
  `.annual_mean`, `.annual_total`, `.annual_extremes`, `.climatology`,
  `.aggregate`
- Indices and derived values as listed above
- Exports: `.to_csv`, `.to_json`, `.to_npz`, `.to_zip`, `.metadata`

`RoClimateArea`

- Access: `.data`, `.bounds`, `.i0/.i1/.j0/.j1`, `.shape`, `.n_cells`,
  `.variables`, `.reductions`, `.columns`, `.polygon`, `.coverage`,
  `.warnings`
- Selection: `.at(date)`, `.subset(...)`, `.series(variable, reduction)`,
  `area["max_temp_mean"]`, `area[["max_temp_mean", "max_temp_max"]]`
- Statistics over daily spatial summaries: `.mean/.avg/.max/.min/.median/
  .std/.var/.quantile/.count/.sum/.describe`
- Aggregation: `.resample`, `.monthly_mean`, `.monthly_total`,
  `.annual_mean`, `.annual_total`, `.climatology`, `.aggregate`
- Exports: `.to_csv`, `.to_json`, `.to_npz`, `.to_zip`, `.metadata`

## Tests

From this directory:

```bash
python -m pytest
```

The tests use a synthetic in-memory group and never contact the network.
They cover grid mapping, statistics, resampling, indices, area summaries and
polygon masks, chunk-aligned reads, the raw chunk cache, the requests-backed
store, client fetching, and export archives.
