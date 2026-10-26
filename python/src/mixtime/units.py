"""Time units, granules, and calendar conversion.

The public API keeps the R package's chronon semantics: values are counts of a
calendar unit since the Unix epoch (1970-01-01), except for durations, which are
plain counts of the unit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

import mixtime_rs as _rs


_CALENDAR_UNITS = (
    "year",
    "quarter",
    "month",
    "day",
    "week",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
)

_MONTH_FACTOR = {"year": 12.0, "quarter": 3.0, "month": 1.0}
_SECONDS_FACTOR = {
    "day": 86_400.0,
    "hour": 3_600.0,
    "minute": 60.0,
    "second": 1.0,
    "millisecond": 1e-3,
    "microsecond": 1e-6,
    "nanosecond": 1e-9,
}


@dataclass(frozen=True)
class Unit:
    """A sized time unit (a granule).

    ``cal_gregorian.month`` is the one-month unit; ``cal_gregorian.month(3)``
    is a three-month granule. The optional ``timezone`` is carried by the unit
    so conversions and formatting can preserve it.
    """

    name: str
    n: int | float = 1
    calendar: "Calendar | None" = None
    timezone: str | None = None

    def __call__(self, n: int | float = 1, *, tz: str | None = None) -> "Unit":
        return replace(self, n=n, timezone=tz if tz is not None else self.timezone)

    def __repr__(self) -> str:
        return f"{self.n:g} {self.name}"


class Calendar:
    """Base class for calendar systems."""

    name = "calendar"

    def __init__(self, units: Iterable[str] = _CALENDAR_UNITS) -> None:
        self.units: dict[str, Unit] = {
            unit: Unit(unit, 1, self) for unit in units
        }

    def __getattr__(self, item: str) -> Unit:
        try:
            return self.units[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __getitem__(self, item: str) -> Unit:
        return self.units[item]

    def __contains__(self, item: str) -> bool:
        return item in self.units

    def convert(
        self,
        values: np.ndarray,
        from_unit: Unit,
        to_unit: Unit,
        *,
        discrete: bool = True,
        duration: bool = False,
        anchored: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError

    def unit(self, name: str, n: int | float = 1, *, tz: str | None = None) -> Unit:
        if name not in self.units:
            raise ValueError(f"{name!r} is not a unit in the {self.name} calendar")
        return self.units[name](n, tz=tz)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class GregorianCalendar(Calendar):
    """Proleptic Gregorian calendar with civil time units."""

    name = "gregorian"

    def __init__(self) -> None:
        super().__init__(_CALENDAR_UNITS)

    def convert(
        self,
        values: np.ndarray,
        from_unit: Unit,
        to_unit: Unit,
        *,
        discrete: bool = True,
        duration: bool = False,
        anchored: bool = True,
    ) -> np.ndarray:
        """Convert values between Gregorian units.

        Conversions through months and days use the same epoch and floor
        semantics as R mixtime. For continuous values, the result is kept
        fractional; for discrete values, finer-to-coarser conversions floor.
        """
        values = _as_float_array(values)
        if from_unit.name == to_unit.name and float(from_unit.n) == float(to_unit.n):
            return _maybe_integral(values, discrete)

        if duration:
            # Durations have no calendar anchor. `anchored` is accepted for
            # call-site clarity, but a duration conversion never crosses the
            # calendar/fixed boundary.
            calendar_like = {"year", "quarter", "month"}
            if (from_unit.name in calendar_like) != (to_unit.name in calendar_like):
                raise ValueError(
                    "Cannot convert a duration between calendar and fixed units; "
                    "their ratio depends on where in the calendar the duration falls"
                )

        # Calendar units: year / quarter / month.
        if from_unit.name in _MONTH_FACTOR and to_unit.name in _MONTH_FACTOR:
            months = values * float(from_unit.n) * _MONTH_FACTOR[from_unit.name]
            result = months / (float(to_unit.n) * _MONTH_FACTOR[to_unit.name])
            return _round_for_discrete(result, discrete)

        # Week chronons are Monday-start 7-day blocks. The integer value is
        # the block index; the Thursday is value * 7, which is what ISO week
        # formatting uses. Converting a week to a day yields the Monday start.
        if from_unit.name == "week":
            if to_unit.name == "week":
                result = values * float(from_unit.n) / float(to_unit.n)
                return _round_for_discrete(result, discrete)
            if duration:
                days = values * float(from_unit.n) * 7.0
            else:
                days = values * float(from_unit.n) * 7.0 - 3.0
            if to_unit.name in _MONTH_FACTOR:
                return self._days_to_calendar(days, to_unit, discrete=discrete)
            return self._days_to_unit(days, to_unit, discrete=discrete)
        if to_unit.name == "week":
            if from_unit.name in _MONTH_FACTOR:
                days = self._calendar_to_days(values, from_unit)
            else:
                days = self._unit_to_days(values, from_unit)
            if duration:
                return _round_for_discrete(
                    days / (float(to_unit.n) * 7.0), discrete
                )
            return _round_for_discrete(
                (days + 3.0) / (float(to_unit.n) * 7.0), discrete
            )

        # Calendar units -> days or finer.
        if from_unit.name in _MONTH_FACTOR:
            days = self._calendar_to_days(values, from_unit)
            return self._days_to_unit(days, to_unit, discrete=discrete)

        # Days or finer -> calendar units.
        if to_unit.name in _MONTH_FACTOR:
            days = self._unit_to_days(values, from_unit)
            return self._days_to_calendar(days, to_unit, discrete=discrete)

        # Civil time units.
        if from_unit.name in _SECONDS_FACTOR and to_unit.name in _SECONDS_FACTOR:
            seconds = values * float(from_unit.n) * _SECONDS_FACTOR[from_unit.name]
            result = seconds / (float(to_unit.n) * _SECONDS_FACTOR[to_unit.name])
            return _round_for_discrete(result, discrete)

        raise ValueError(f"Cannot convert {from_unit!r} to {to_unit!r}")

    def _calendar_to_days(self, values: np.ndarray, unit: Unit) -> np.ndarray:
        n = float(unit.n)
        if unit.name == "year":
            days, _ = _rs.year_to_day(values, n, 1.0)
            return days
        if unit.name == "quarter":
            months = values * n * 3.0
            days, _ = _rs.month_to_day(months, 1.0, 1.0)
            return days
        if unit.name == "month":
            days, _ = _rs.month_to_day(values, n, 1.0)
            return days
        raise ValueError(f"Not a calendar unit: {unit!r}")

    def _days_to_calendar(
        self, days: np.ndarray, unit: Unit, *, discrete: bool
    ) -> np.ndarray:
        n = float(unit.n)
        if unit.name == "year":
            years, mod = _rs.day_to_year(days, 1.0, 1.0)
            # The fractional part is the fraction of the calendar year, not
            # the fraction of 365 days. This matches R's continuous year.
            next_year_days, _ = _rs.year_to_day(years + 1.0, 1.0, 1.0)
            this_year_days, _ = _rs.year_to_day(years, 1.0, 1.0)
            result = years + mod / (next_year_days - this_year_days)
            return _round_for_discrete(result / n, discrete)
        if unit.name == "month":
            months, mod = _rs.day_to_month(days, 1.0, 1.0)
            result = months + mod / self._month_days(months)
            return _round_for_discrete(result / n, discrete)
        if unit.name == "quarter":
            months, mod = _rs.day_to_month(days, 1.0, 1.0)
            month_continuous = months + mod / self._month_days(months)
            return _round_for_discrete(month_continuous / (3.0 * n), discrete)
        raise ValueError(f"Not a calendar unit: {unit!r}")

    @staticmethod
    def _month_days(month_index: np.ndarray) -> np.ndarray:
        start, _ = _rs.month_to_day(month_index, 1.0, 1.0)
        end, _ = _rs.month_to_day(month_index + 1.0, 1.0, 1.0)
        return end - start

    def _days_to_unit(
        self, days: np.ndarray, unit: Unit, *, discrete: bool
    ) -> np.ndarray:
        n = float(unit.n)
        if unit.name == "day":
            return _round_for_discrete(days / n, discrete)
        if unit.name in _SECONDS_FACTOR:
            seconds = days * 86_400.0
            result = seconds / (n * _SECONDS_FACTOR[unit.name])
            return _round_for_discrete(result, discrete)
        raise ValueError(f"Cannot convert days to {unit!r}")

    def _unit_to_days(self, values: np.ndarray, unit: Unit) -> np.ndarray:
        n = float(unit.n)
        if unit.name == "day":
            return values * n
        if unit.name in _SECONDS_FACTOR:
            seconds = values * n * _SECONDS_FACTOR[unit.name]
            return seconds / 86_400.0
        raise ValueError(f"Cannot convert {unit!r} to days")


def _as_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("time values must be one-dimensional")
    return arr


def _round_for_discrete(values: np.ndarray, discrete: bool) -> np.ndarray:
    if discrete:
        return np.floor(values)
    return values


def _maybe_integral(values: np.ndarray, discrete: bool) -> np.ndarray:
    if discrete and np.all(np.isfinite(values)) and np.all(values == np.floor(values)):
        return values.astype(np.int64)
    return values


# The singleton Gregorian calendar used by the convenience constructors.
cal_gregorian = GregorianCalendar()
