"""Calendar-aware operations on time vectors."""

from __future__ import annotations

import datetime as _dt
import re
from math import floor, gcd
from typing import Any

import numpy as np

from .time import (
    CyclicalTime,
    Duration,
    LinearTime,
    Mixtime,
    TimeVector,
    _MONTHS_PER_UNIT,
    _check_discrete_shift,
    _check_duration_shift,
    _coerce_values,
    _date_from_day,
    _local_naive_from_seconds,
)
from .units import Calendar, Unit


def time_floor(
    x: TimeVector | Mixtime, granule: Unit | str | Duration | TimeVector
) -> TimeVector | Mixtime:
    """Floor time points to the start of the given granule."""
    return _round_to(x, granule, np.floor)


def time_ceiling(
    x: TimeVector | Mixtime, granule: Unit | str | Duration | TimeVector
) -> TimeVector | Mixtime:
    """Ceil time points to the next boundary of the given granule.

    R's ``time_ceiling()`` uses ``ceiling(floor(x) + 0.5)``, so an exact
    boundary moves forward rather than staying put.
    """
    return _round_to(x, granule, lambda values: np.ceil(np.floor(values) + 0.5))


def time_round(
    x: TimeVector | Mixtime, granule: Unit | str | Duration | TimeVector
) -> TimeVector | Mixtime:
    """Round time points to the nearest boundary of the given granule."""
    return _round_to(x, granule, np.round)


def time_shift(x: TimeVector | Mixtime, by: Duration) -> TimeVector | Mixtime:
    """Shift time points by a duration."""
    if not isinstance(by, Duration):
        raise TypeError("time_shift() expects a Duration")
    if isinstance(x, Mixtime):
        return Mixtime([time_shift(part, by) for part in x.parts])
    return x + by


def seq(
    start: TimeVector,
    end: TimeVector,
    by: Any = None,
    *,
    length_out: int | None = None,
) -> TimeVector:
    """Generate a regular sequence of time points.

    ``by`` may be a number of source granules, a :class:`Duration`, a
    :class:`Unit`, or a unit name such as ``"month"``. Counted unit strings
    such as ``"1 month"`` and ``"3 days"`` are also accepted.

    Day, week, month, quarter, and year steps on ``date`` and ``datetime`` use
    calendar arithmetic. Month-like steps clamp month ends and preserve the
    local wall-clock time across DST changes. R's endpoint alignment for
    month-like steps is preserved, so a clamped month end may be excluded when
    the end value is earlier in its month/quarter/year than the start value.
    """
    if not isinstance(start, TimeVector) or not isinstance(end, TimeVector):
        raise TypeError("seq() expects TimeVector start and end values")
    if len(start) != 1 or len(end) != 1:
        raise ValueError("seq() expects length-one start and end values")
    if (
        start.chronon.name != end.chronon.name
        or float(start.chronon.n) != float(end.chronon.n)
    ):
        raise ValueError("Incompatible time chronons in seq()")
    if start.calendar is not end.calendar:
        raise ValueError("seq() does not support mixing calendars")

    if length_out is not None:
        if by is not None:
            raise ValueError(
                "seq() does not support combining by and length_out; use one "
                "or the other"
            )
        if not np.isfinite(length_out) or length_out < 0:
            raise ValueError("length_out must be a finite value >= 0")
        count = int(np.ceil(length_out))
        if count == 0:
            return start[:0]
        values = np.linspace(
            float(start.values[0]), float(end.values[0]), count
        )
        discrete = start.discrete and bool(
            np.all(np.isfinite(values) & (values == np.floor(values)))
        )
        return type(start)(
            _coerce_values(values, discrete),
            start.chronon,
            discrete=discrete,
            calendar=start.calendar,
            cycle=start.cycle,
            timezone=start.timezone,
        )

    calendar_unit = _calendar_step_unit(by, start)
    if calendar_unit is not None:
        return _seq_calendar(start, end, calendar_unit)

    step = _step_in_unit(1 if by is None else by, start)
    if not np.isfinite(step) or step == 0:
        raise ValueError("by must be a finite non-zero step")
    begin = float(start.values[0])
    finish = float(end.values[0])
    if (finish - begin) * step < 0:
        return start[:0]
    count = int(np.floor((finish - begin) / step + 1e-12))
    values = begin + step * np.arange(count + 1)
    # R's discrete shifts floor each fractional step; they do not round.
    if start.discrete and step != np.floor(step):
        values = np.floor(values)
    # Keep the endpoint exactly when it is a step boundary.
    if np.isclose(values[-1], finish):
        values[-1] = finish
    return start._new(_coerce_values(values, start.discrete))


_CALENDAR_STEP_UNITS = {"day", "week", "month", "quarter", "year"}
_STEP_STRING = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z]+)\s*$"
)


def _parse_step_unit(by: str, calendar: Calendar) -> Unit:
    """Parse a unit name or a counted unit string such as ``"3 days"``."""
    try:
        return calendar.unit(by)
    except ValueError:
        match = _STEP_STRING.fullmatch(by)
        if match is None:
            if by.endswith("s") and by[:-1] in calendar:
                return calendar.unit(by[:-1])
            raise
        name = match.group(2)
        if name not in calendar and name.endswith("s"):
            name = name[:-1]
        return calendar.unit(name, n=float(match.group(1)))


def _step_unit(by: Any, start: TimeVector) -> Unit | None:
    """Normalize a sequence step to a unit, when it is unit-like."""
    if isinstance(by, Duration):
        if len(by) != 1:
            raise ValueError("by duration must have length 1")
        return Unit(
            by.chronon.name,
            float(by.chronon.n) * float(by.values[0]),
            by.chronon.calendar or start.calendar,
            by.chronon.timezone,
        )
    if isinstance(by, Unit):
        return by
    if isinstance(by, str):
        return _parse_step_unit(by, start.calendar)
    return None


def _calendar_step_unit(by: Any, start: TimeVector) -> Unit | None:
    """Return the calendar unit for a date/datetime sequence step."""
    if not isinstance(start, LinearTime):
        return None
    if float(start.chronon.n) != 1.0:
        return None
    if start.chronon.name not in {"day", "second"}:
        return None
    unit = _step_unit(by, start)
    if unit is None or unit.name not in _CALENDAR_STEP_UNITS:
        return None
    amount = float(unit.n)
    if not np.isfinite(amount) or amount == 0:
        raise ValueError("by must be a finite non-zero step")
    return unit


def _seq_calendar(start: TimeVector, end: TimeVector, unit: Unit) -> TimeVector:
    """Generate a calendar-aware sequence for date/datetime values."""
    amount = float(unit.n)
    direction = 1 if amount > 0 else -1
    step_units = abs(amount)
    if unit.name in _MONTHS_PER_UNIT:
        unit_months = _MONTHS_PER_UNIT[unit.name]
        step_months = step_units * unit_months
        if not float(step_months).is_integer():
            raise NotImplementedError(
                "fractional month shifts are not implemented; the sequence "
                "step must be a whole number of months. Use a whole-month "
                "step or a day/week step."
            )
        return _seq_calendar_months(
            start,
            end,
            unit.name,
            step_units,
            step_months,
            direction,
        )
    step_days = step_units * (7.0 if unit.name == "week" else 1.0)
    return _seq_calendar_days(start, end, step_days, direction)


def _seq_calendar_months(
    start: TimeVector,
    end: TimeVector,
    unit_name: str,
    step_units: float,
    step_months: float,
    direction: int,
) -> TimeVector:
    """R's month-like sequence alignment and anchor-based shifts."""
    begin = float(start.values[0])
    finish = float(end.values[0])
    if (finish - begin) * direction < 0:
        return start[:0]

    unit_months = _MONTHS_PER_UNIT[unit_name]
    start_local = _seq_local_datetime(start)
    end_local = _seq_local_datetime(end)

    from_div = _month_index(start_local) // unit_months
    from_unit_start = _datetime_from_month_index(from_div * unit_months)
    offset = start_local - from_unit_start

    aligned_end = end_local - offset
    to_div = _month_index(aligned_end) // unit_months
    to_unit_start = _datetime_from_month_index(to_div * unit_months)
    if direction < 0 and aligned_end != to_unit_start:
        to_div += 1

    n_steps = floor(direction * (to_div - from_div) / step_units + 1e-12)
    if n_steps < 0:
        return start[:0]

    month_unit = Unit("month", 1, start.calendar, start.timezone)
    values = np.empty(n_steps + 1, dtype=float)
    for k in range(n_steps + 1):
        duration = Duration(
            np.asarray([direction * k * step_months], dtype=float),
            month_unit,
            discrete=False,
            calendar=start.calendar,
        )
        values[k] = float((start + duration).values[0])
    return start._new(_coerce_values(values, start.discrete))


def _seq_calendar_days(
    start: TimeVector,
    end: TimeVector,
    step_days: float,
    direction: int,
) -> TimeVector:
    """Anchor-based day/week sequences, including DST-safe local shifts."""
    begin = float(start.values[0])
    finish = float(end.values[0])
    if (finish - begin) * direction < 0:
        return start[:0]

    start_local = _seq_local_datetime(start)
    end_local = _seq_local_datetime(end)
    diff_days = (end_local - start_local).total_seconds() / 86_400.0
    count = floor(direction * diff_days / step_days + 1e-12)
    if count < 0:
        return start[:0]

    day_unit = Unit("day", 1, start.calendar, start.timezone)
    values = np.empty(count + 1, dtype=float)
    for k in range(count + 1):
        duration = Duration(
            np.asarray([direction * k * step_days], dtype=float),
            day_unit,
            discrete=False,
            calendar=start.calendar,
        )
        values[k] = float((start + duration).values[0])
    return start._new(_coerce_values(values, start.discrete))


def _seq_local_datetime(x: TimeVector) -> _dt.datetime:
    value = float(x.values[0])
    if x.chronon.name == "day":
        base_day = int(np.floor(value))
        fraction = value - base_day
        return _dt.datetime.combine(
            _date_from_day(base_day), _dt.time()
        ) + _dt.timedelta(days=fraction)
    return _local_naive_from_seconds(value, x.timezone)


def _month_index(value: _dt.datetime) -> int:
    return (value.year - 1970) * 12 + (value.month - 1)


def _datetime_from_month_index(index: int) -> _dt.datetime:
    year = index // 12 + 1970
    month = index % 12 + 1
    return _dt.datetime(year, month, 1)


def _round_to(
    x: TimeVector | Mixtime,
    granule: Unit | str | Duration | TimeVector,
    rounder,
) -> TimeVector | Mixtime:
    if isinstance(x, Mixtime):
        return Mixtime([_round_to(part, granule, rounder) for part in x.parts])
    if not isinstance(x, (LinearTime, CyclicalTime)):
        raise TypeError("rounding is only implemented for linear and cyclical time")
    unit = _granule_unit(granule, x.calendar)
    converted = x.calendar.convert(
        x.values, x.chronon, unit, discrete=False
    )
    rounded = rounder(converted)
    back = x.calendar.convert(
        rounded, unit, x.chronon, discrete=False
    )
    return x._new(_coerce_values(back, x.discrete))


def _granule_unit(
    granule: Unit | str | Duration | TimeVector,
    calendar: Calendar,
) -> Unit:
    if isinstance(granule, str):
        return calendar.unit(granule)
    if isinstance(granule, Unit):
        if float(granule.n) == 0:
            raise ValueError("granule unit must not be zero")
        return granule
    if isinstance(granule, Duration):
        if len(granule) != 1:
            raise ValueError("granule duration must have length 1")
        unit = granule.chronon
        size = float(unit.n) * float(granule.values[0])
        if size == 0:
            raise ValueError("granule duration must not be zero")
        return Unit(
            unit.name,
            size,
            unit.calendar or calendar,
            unit.timezone,
        )
    if isinstance(granule, TimeVector):
        return granule.chronon
    raise TypeError(f"Unsupported granule: {type(granule)!r}")


def _step_in_unit(by: Any, start: TimeVector) -> float:
    if isinstance(by, Duration):
        if len(by) != 1:
            raise ValueError("by duration must have length 1")
        _check_duration_shift(start.chronon, by.chronon, start.discrete)
        converted = start.calendar.convert(
            by.values,
            by.chronon,
            start.chronon,
            discrete=False,
            duration=True,
            anchored=True,
        )
        if start.discrete:
            _check_discrete_shift(start.chronon, by.chronon, converted)
        return float(converted[0])
    if isinstance(by, Unit):
        _check_duration_shift(start.chronon, by, start.discrete)
        converted = start.calendar.convert(
            np.asarray([1.0]),
            by,
            start.chronon,
            discrete=False,
            duration=True,
            anchored=True,
        )
        if start.discrete:
            _check_discrete_shift(start.chronon, by, converted)
        return float(converted[0])
    if isinstance(by, str):
        unit = _parse_step_unit(by, start.calendar)
        _check_duration_shift(start.chronon, unit, start.discrete)
        converted = start.calendar.convert(
            np.asarray([1.0]),
            unit,
            start.chronon,
            discrete=False,
            duration=True,
            anchored=True,
        )
        if start.discrete:
            _check_discrete_shift(start.chronon, unit, converted)
        return float(converted[0])
    return float(by)


def circsum(x: Any, size: int, step: int | None = None) -> np.ndarray:
    """Sum windows that wrap around a circular vector.

    ``size`` is the window length; a negative size anchors the window at its
    end. ``step`` is the starting-index increment and defaults to ``size``.
    The result length is the number of windows before the pattern repeats.
    """
    values = np.asarray(x)
    n = len(values)
    size = int(size)
    step = size if step is None else int(step)
    size_abs = abs(size)
    step_abs = abs(step)
    if n == 0 or size_abs == 0 or step_abs == 0:
        return np.asarray([], dtype=values.dtype)

    offsets = np.arange(size_abs)
    if size < 0:
        offsets = -offsets[::-1]

    if size_abs == 1:
        if step == 1:
            return values.copy()
        windows = n // gcd(n, step_abs)
        indices = ((np.arange(windows) * step) % n).astype(int)
        return values[indices]

    windows = n // gcd(n, step_abs)
    result = np.empty(windows, dtype=np.result_type(values.dtype, float))
    for index in range(windows):
        start = (index * step) % n
        result[index] = values[(start + offsets) % n].sum()
    return result
