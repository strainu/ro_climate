"""Exceptions raised by the :mod:`ro_climate` package."""


class RoClimateError(Exception):
    """Base class for all ro-climate errors."""


class OutsideGridError(RoClimateError, ValueError):
    """The requested coordinate is outside the ANM grid."""


class NoDataError(RoClimateError):
    """No data are available for the requested point/date range."""


class InvalidTimeRangeError(RoClimateError, ValueError):
    """The requested date range is invalid."""


class UnknownVariableError(RoClimateError, KeyError):
    """The requested variable is not part of the ANM cube."""


class NonAdditiveVariableError(RoClimateError, ValueError):
    """A non-additive variable was used with a sum operation."""


class StoreUnavailableError(RoClimateError):
    """The remote Zarr store could not be reached or read."""


class CacheError(RoClimateError):
    """The local cache is invalid or cannot be used."""


class CacheMissError(CacheError):
    """Offline mode was requested but no suitable cached point exists."""
