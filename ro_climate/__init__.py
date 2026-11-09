"""Romanian daily climate data from ANM.

The package provides the public ANM daily gridded cube and official station
observations from the ANM Open Data Portal (ODP).

Quick start::

    from ro_climate import RoClimate, load_station_series

    with RoClimate() as client:
        p = client.point(
            26.1025, 44.4325, start="2020-01-01", end="2020-12-31"
        )
        print(p.mean())
        print(p.sum("precipitation"))
        p.to_csv("bucharest_2020.csv")

    station = load_station_series(
        "15014", start="2020-01-01", end="2020-12-31"
    )
    print(station.mean())
"""

from typing import TYPE_CHECKING

from .area import RoClimateArea
from .client import RoClimate, DEFAULT_STORE, ITEM_URL
from .exceptions import (
    RoClimateError,
    InvalidTimeRangeError,
    NoDataError,
    NonAdditiveVariableError,
    OutsideGridError,
    StationNotFoundError,
    StoreUnavailableError,
    UnknownVariableError,
)
from .point import RoClimatePoint
from .variables import (
    ATTRIBUTION,
    DEFAULT_STATION_VARIABLES,
    DEFAULT_VARIABLES,
    MAX_TEMP,
    MEAN_TEMP,
    MIN_TEMP,
    PRECIPITATION,
    PRESSURE,
    STATION_ATTRIBUTION,
    STATION_VARIABLES,
    VARIABLES,
    WIND_SPEED,
    Variable,
    resolve_station_variables,
    resolve_variables,
)

if TYPE_CHECKING:
    from .stations import (
        ODP_BASE_URL,
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

__version__ = "0.2.0"

_STATION_EXPORTS = frozenset(
    {
        "ODP_BASE_URL",
        "RoClimateStation",
        "Station",
        "clear_station_cache",
        "default_station_cache_dir",
        "get_station",
        "list_stations",
        "load_station_series",
        "nearest_station",
        "nearest_stations",
    }
)


def __getattr__(name: str):
    """Resolve station API names lazily.

    The lazy import keeps ``python -m ro_climate.stations`` free of runpy's
    "found in sys.modules" warning while preserving the public API.
    """

    if name in _STATION_EXPORTS:
        from . import stations as _stations

        value = getattr(_stations, name)
        globals()[name] = value
        return value
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "RoClimate",
    "RoClimatePoint",
    "RoClimateArea",
    "RoClimateStation",
    "Station",
    "load_station_series",
    "list_stations",
    "get_station",
    "nearest_stations",
    "nearest_station",
    "clear_station_cache",
    "default_station_cache_dir",
    "ODP_BASE_URL",
    "Variable",
    "VARIABLES",
    "STATION_VARIABLES",
    "DEFAULT_VARIABLES",
    "DEFAULT_STATION_VARIABLES",
    "MAX_TEMP",
    "MIN_TEMP",
    "MEAN_TEMP",
    "PRECIPITATION",
    "WIND_SPEED",
    "PRESSURE",
    "ATTRIBUTION",
    "STATION_ATTRIBUTION",
    "resolve_variables",
    "resolve_station_variables",
    "DEFAULT_STORE",
    "ITEM_URL",
    "RoClimateError",
    "OutsideGridError",
    "NoDataError",
    "InvalidTimeRangeError",
    "UnknownVariableError",
    "NonAdditiveVariableError",
    "StoreUnavailableError",
    "StationNotFoundError",
]
