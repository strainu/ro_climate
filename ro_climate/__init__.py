"""Romanian daily gridded climate data from ANM.

Quick start::

    from ro_climate import RoClimate

    with RoClimate() as client:
        p = client.point(
            26.1025, 44.4325, start="2020-01-01", end="2020-12-31"
        )
        print(p.mean())
        print(p.sum("precipitation"))
        p.to_csv("bucharest_2020.csv")
"""

from .client import RoClimate, DEFAULT_STORE, ITEM_URL
from .exceptions import (
    RoClimateError,
    CacheError,
    CacheMissError,
    InvalidTimeRangeError,
    NoDataError,
    NonAdditiveVariableError,
    OutsideGridError,
    StoreUnavailableError,
    UnknownVariableError,
)
from .point import RoClimatePoint
from .variables import (
    ATTRIBUTION,
    DEFAULT_VARIABLES,
    MAX_TEMP,
    MIN_TEMP,
    PRECIPITATION,
    VARIABLES,
    Variable,
    resolve_variables,
)

__version__ = "0.1.0"

__all__ = [
    "RoClimate",
    "RoClimatePoint",
    "Variable",
    "VARIABLES",
    "DEFAULT_VARIABLES",
    "MAX_TEMP",
    "MIN_TEMP",
    "PRECIPITATION",
    "ATTRIBUTION",
    "resolve_variables",
    "DEFAULT_STORE",
    "ITEM_URL",
    "RoClimateError",
    "OutsideGridError",
    "NoDataError",
    "InvalidTimeRangeError",
    "UnknownVariableError",
    "NonAdditiveVariableError",
    "StoreUnavailableError",
    "CacheError",
    "CacheMissError",
]
