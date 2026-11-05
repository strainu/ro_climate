"""Variable metadata for the ANM daily cube."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .exceptions import UnknownVariableError


@dataclass(frozen=True)
class Variable:
    """Description of one data variable in the cube."""

    name: str
    units: str
    long_name: str
    additive: bool = False


MAX_TEMP = Variable("max_temp", "°C", "daily maximum temperature")
MIN_TEMP = Variable("min_temp", "°C", "daily minimum temperature")
PRECIPITATION = Variable(
    "precipitation", "mm", "daily precipitation", additive=True
)

VARIABLES: dict[str, Variable] = {
    MAX_TEMP.name: MAX_TEMP,
    MIN_TEMP.name: MIN_TEMP,
    PRECIPITATION.name: PRECIPITATION,
}

DEFAULT_VARIABLES: tuple[str, ...] = (
    MAX_TEMP.name,
    MIN_TEMP.name,
    PRECIPITATION.name,
)

ATTRIBUTION = (
    "Data © Administrația Națională de Meteorologie (MeteoRomania), "
    "CC BY 4.0, via data.gov.ro; cube assembly and hosting by "
    "West University of Timișoara (ROCS / EOCube.RO)."
)


def resolve_variables(
    variables: Iterable[str] | str | None,
) -> tuple[Variable, ...]:
    """Return validated :class:`Variable` objects in the requested order."""

    if variables is None:
        names: tuple[str, ...] = DEFAULT_VARIABLES
    elif isinstance(variables, str):
        names = (variables,)
    else:
        names = tuple(variables)

    if not names:
        raise ValueError("at least one variable must be selected")

    seen: set[str] = set()
    out: list[Variable] = []
    for name in names:
        if name not in VARIABLES:
            raise UnknownVariableError(
                f"unknown variable {name!r}; expected one of "
                f"{sorted(VARIABLES)}"
            )
        if name in seen:
            continue
        seen.add(name)
        out.append(VARIABLES[name])
    return tuple(out)
