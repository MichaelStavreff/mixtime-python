"""Common chronon selection for Gregorian time vectors.

R mixtime's ``chronon_glb()`` and ``chronon_lub()`` walk its chronon graph.
This port uses the same partial order explicitly: calendar units
(year > quarter > month) and civil units (week > day > hour > ...) share the
day-and-finer chain, while week and calendar units are otherwise incomparable.
"""

from __future__ import annotations

from math import gcd
from typing import Any

from .time import Mixtime, TimeVector
from .units import Unit, _nests_in, cal_gregorian

_BASE_ORDER = (
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
)


def chronon_glb(*objects: Any, ptype: Unit | str | None = None) -> Unit:
    """Return the greatest lower bound of the supplied chronons.

    This is the closest common chronon that is no coarser than every input,
    for example ``day`` for a month and a week. ``ptype`` is returned as-is
    when supplied, matching R's current unchecked behaviour.
    """
    units = _collect_units(objects)
    if ptype is not None:
        return _normalize_unit(ptype)
    if not units:
        raise ValueError("chronon_glb() needs at least one chronon")
    return _bound(units, finest=False)


def chronon_lub(*objects: Any, ptype: Unit | str | None = None) -> Unit:
    """Return the least upper bound of the supplied chronons.

    This is the closest common chronon that is no finer than every input.
    Week and calendar units have no common coarser chronon, so that combination
    raises ``ValueError``.
    """
    units = _collect_units(objects)
    if ptype is not None:
        return _normalize_unit(ptype)
    if not units:
        raise ValueError("chronon_lub() needs at least one chronon")
    return _bound(units, finest=True)


def chronon_common(*objects: Any, ptype: Unit | str | None = None) -> Unit:
    """Alias for :func:`chronon_glb`, kept for R API familiarity."""
    return chronon_glb(*objects, ptype=ptype)


def _collect_units(objects: tuple[Any, ...]) -> list[Unit]:
    units: list[Unit] = []

    def add(value: Any) -> None:
        """Collect one object's chronons, recursing into sequences and parts."""
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        elif isinstance(value, Mixtime):
            for part in value.parts:
                add(part)
        elif isinstance(value, Unit):
            units.append(value)
        elif isinstance(value, str):
            units.append(cal_gregorian.unit(value))
        elif isinstance(value, TimeVector):
            units.append(value.chronon)
        else:
            raise TypeError(
                "chronon bounds expect time vectors, units, unit names, or "
                "Mixtime objects"
            )

    for obj in objects:
        add(obj)
    return units


def _normalize_unit(value: Unit | str) -> Unit:
    if isinstance(value, Unit):
        return value
    if isinstance(value, str):
        return cal_gregorian.unit(value)
    raise TypeError("ptype must be a Unit or unit name")


def _bound(units: list[Unit], *, finest: bool) -> Unit:
    names = [unit.name for unit in units]
    if len(set(names)) == 1:
        return _make_unit(units, names[0], _gcd_n([unit.n for unit in units]))

    candidates = reversed(_BASE_ORDER) if finest else _BASE_ORDER
    for name in candidates:
        candidate = cal_gregorian.unit(name)
        if finest:
            compatible = all(
                _nests_in(cal_gregorian.unit(unit.name), candidate)
                for unit in units
            )
        else:
            compatible = all(
                _nests_in(candidate, cal_gregorian.unit(unit.name))
                for unit in units
            )
        if compatible:
            return _make_unit(units, name, 1)

    if finest:
        raise ValueError(
            "One or more of the provided chronons do not share a common "
            "coarser chronon."
        )
    raise ValueError("No common chronon found")


def _make_unit(units: list[Unit], name: str, n: int) -> Unit:
    timezones = {unit.timezone for unit in units}
    timezone = next(iter(timezones)) if len(timezones) == 1 else None
    calendar = next((unit.calendar for unit in units if unit.calendar), None)
    if calendar is not None and name in calendar:
        return calendar.unit(name, n, tz=timezone)
    return cal_gregorian.unit(name, n, tz=timezone)


def _gcd_n(values: list[int | float]) -> int:
    integers: list[int] = []
    for value in values:
        numeric = float(value)
        if not numeric.is_integer():
            return 1
        integers.append(int(numeric))
    result = 0
    for value in integers:
        result = gcd(result, value)
    return max(1, result)
