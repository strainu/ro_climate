# ro-climate — ANM daily gridded meteorology

`ro-climate` is a small, object-oriented Python client for the public **ANM
daily gridded meteorological cube** for Romania. It reads a single grid cell for an
optional date range, returns a pandas time series, and provides statistics,
time aggregation, climate indices, exports, and a local zip cache.

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
> and select only the variables you need. The local zip cache makes later
> reads of the same point local.

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

## Local zip cache

Caching is enabled by default. Each cached point/variable set is a single zip
archive named after the grid centre, for example:

```text
ro_climate_lat44.4350_lon26.1050_max-temp-min-temp-precipitation_a787356e6d8f.zip
```

The archive contains:

```text
data.npy       float32 values, shape (days, variables)
dates.npy      datetime64[D] dates
metadata.json  grid cell, actual/requested range, frontiers, units, licence,
               attribution, warnings, retrieval time
README.txt     human-readable provenance note
```

No CSV is written inside the cache. The cache key includes the store URL,
cell indices, and variable set; the requested date range is stored in the
metadata. A later request for a wider range updates the archive, and a request
covered by the archive is served without re-downloading the data chunks (the
small frontier scalars may still be refreshed).

```python
from ro_climate import RoClimate

client = RoClimate(cache_dir="ro-climate-cache")
point = client.point(26.1025, 44.4325, start="2020-01-01", end="2020-12-31")
print(point.from_cache, point.cache_path)

print(client.cache_info())       # one row per archive
client.prune_cache(older_than_days=30)
client.clear_cache()
client.close()
```

Offline use:

```python
offline = RoClimate(cache_dir="ro-climate-cache", offline=True)
point = offline.point(26.1025, 44.4325, start="2020-01-01", end="2020-12-31")
# A missing or non-covering archive raises CacheMissError.
```

Set `RO_CLIMATE_CACHE_DIR` to override the platform cache directory. Pass
`cache=False` to bypass the cache for a client, or `refresh=True` to a
`point()` call to re-download and replace the archive.

## Exports

```python
point.to_csv("bucharest_2020.csv")
point.to_json("bucharest_2020.json")
point.to_npz("bucharest_2020.npz")
point.to_zip("bucharest_2020.zip")   # same layout as the cache archive
print(point.metadata())
```

`to_csv()` writes the date index with the column name `date` and formats
floats to four decimals by default. Extra keyword arguments are passed to
pandas, so override `float_format` or use `sep=";"` as needed.

## API summary

`RoClimate(...)`

- `RoClimate(store_url=None, *, discover=True, cache=True, cache_dir=None,
  offline=False, timeout=30.0, retries=3, frontier_ttl=3600)`
- `.point(lon, lat, start=None, end=None, variables=None, *, cache=None,
  refresh=None, strict=False) -> RoClimatePoint`
- `.frontiers(force=False)` — last written day per variable
- `.cache_info()`, `.clear_cache()`, `.prune_cache(older_than_days=30)`
- Context manager and `.close()`

`RoClimatePoint`

- Access: `.data`, `.variables`, `.units`, `.start`, `.end`, `.i`, `.j`,
  `.grid_lon`, `.grid_lat`, `.cell_bounds`, `.coverage`, `.warnings`,
  `.from_cache`, `.cache_path`
- Selection: `.at(date)`, `.subset(...)`, `point["max_temp"]`,
  `point[["max_temp", "precipitation"]]`
- Statistics: `.mean/.avg/.max/.min/.median/.std/.var/.quantile/.count/
  .sum/.describe`
- Aggregation: `.resample`, `.monthly_mean`, `.monthly_total`,
  `.annual_mean`, `.annual_total`, `.annual_extremes`, `.climatology`,
  `.aggregate`
- Indices and derived values as listed above
- Exports: `.to_csv`, `.to_json`, `.to_npz`, `.to_zip`, `.metadata`

## Tests

From this directory:

```bash
python -m pytest
```

The tests use a synthetic in-memory group and never contact the network.
They cover grid mapping, statistics, resampling, indices, the zip cache,
the requests-backed store, client fetching, cache hits, and offline mode.
