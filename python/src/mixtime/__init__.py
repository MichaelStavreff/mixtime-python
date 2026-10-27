"""mixtime: Gregorian time vectors for Python.

The package has two layers:

* a low-level kernel (``day_to_month``, ``month_to_day``, ...) implemented in
  Rust and exposed through PyO3; and
* a Pythonic time-vector model (``yearmonth``, ``date``, ``datetime``, ...)
  built on top of that kernel.

The high-level API is deliberately Pythonic rather than a literal port of R's
S7/vctrs classes. Observable Gregorian chronon semantics are kept compatible
with mixtime 0.4.0.
"""

from __future__ import annotations

import numpy as np

import mixtime_rs as _rs

from .components import (
    ComponentSpec,
    ComponentValue,
    cyc,
    lin,
    time_components,
    time_compose,
)
from .granules import chronon_common, chronon_glb, chronon_lub
from .metadata import is_mixtime, time_calendar, time_chronon, time_cycle
from .ops import circsum, seq, time_ceiling, time_floor, time_round, time_shift
from .predicates import (
    time_is_complete_at,
    time_is_cyclical,
    time_is_determinate_at,
    time_is_duration,
    time_is_linear,
)
from .scalar import TimeScalar
from .time import (
    CyclicalTime,
    Duration,
    LinearTime,
    Mixtime,
    TimeVector,
    cyclical_time,
    date,
    datetime,
    day_of_month,
    day_of_week,
    day_of_year,
    days,
    duration,
    hours,
    linear_time,
    milliseconds,
    minutes,
    mixtime,
    month_of_year,
    months,
    quarters,
    seconds,
    time_of_day,
    week_of_year,
    weeks,
    year,
    yearmonth,
    yearquarter,
    yearweek,
    years,
)
from .timezones import tz_abbreviation, tz_name, tz_offset, tz_transitions
from .units import Calendar, GregorianCalendar, Unit, cal_gregorian

__all__ = [
    # low-level kernel
    "day_to_month",
    "day_to_quarter",
    "day_to_year",
    "month_to_day",
    "year_to_day",
    # model
    "TimeVector",
    "TimeScalar",
    "LinearTime",
    "CyclicalTime",
    "Duration",
    "Mixtime",
    "Unit",
    "Calendar",
    "GregorianCalendar",
    "cal_gregorian",
    "linear_time",
    "cyclical_time",
    "duration",
    "mixtime",
    "year",
    "yearquarter",
    "yearmonth",
    "yearweek",
    "date",
    "datetime",
    "month_of_year",
    "day_of_year",
    "day_of_month",
    "day_of_week",
    "week_of_year",
    "time_of_day",
    "years",
    "quarters",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    # operations
    "time_floor",
    "time_ceiling",
    "time_round",
    "time_shift",
    "seq",
    "circsum",
    "chronon_glb",
    "chronon_lub",
    "chronon_common",
    # metadata
    "is_mixtime",
    "time_chronon",
    "time_cycle",
    "time_calendar",
    # timezones
    "tz_name",
    "tz_abbreviation",
    "tz_offset",
    "tz_transitions",
    # predicates
    "time_is_linear",
    "time_is_cyclical",
    "time_is_duration",
    "time_is_determinate_at",
    "time_is_complete_at",
    # components
    "ComponentSpec",
    "ComponentValue",
    "lin",
    "cyc",
    "time_components",
    "time_compose",
]


def __getattr__(name: str):
    """Lazily expose the optional pandas extension classes."""
    if name in {"MixtimeDtype", "MixtimeArray"}:
        from . import pandas_ext

        return getattr(pandas_ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def day_to_month(
    x: np.ndarray, from_n: float = 1.0, to_n: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert day chronons to month chronons.

    ``x`` must be a one-dimensional ``float64`` ``numpy.ndarray``; its buffer
    is read in place. Returns ``(div, mod)`` as NumPy arrays: the containing
    month block and the day offset from the start of that block.
    """
    return _rs.day_to_month(x, from_n, to_n)


def day_to_quarter(
    x: np.ndarray, from_n: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert day chronons to quarter chronons (3-month blocks)."""
    return _rs.day_to_month(x, from_n, 3.0)


def day_to_year(
    x: np.ndarray, from_n: float = 1.0, to_n: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert day chronons to year chronons.

    Returns ``(div, mod)``: the containing year and the day-of-year offset.
    """
    return _rs.day_to_year(x, from_n, to_n)


def month_to_day(
    x: np.ndarray, from_n: float = 1.0, to_n: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert month chronons to day chronons (``mod`` is always zero)."""
    return _rs.month_to_day(x, from_n, to_n)


def year_to_day(
    x: np.ndarray, from_n: float = 1.0, to_n: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert year chronons to day chronons (``mod`` is always zero)."""
    return _rs.year_to_day(x, from_n, to_n)
