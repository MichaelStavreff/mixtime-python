"""Calendar-aware operations on time vectors."""

from __future__ import annotations

from math import gcd
from typing import Any

import numpy as np

from .time import (
    CyclicalTime,
    Duration,
    LinearTime,
    Mixtime,
    TimeVector,
    _check_discrete_shift,
    _check_duration_shift,
    _coerce_values,
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

    ``by`` may be a number of source granules, a :class:`Duration`, or a unit
    name such as ``"month"``. The result includes ``end`` when it falls on a
    step boundary.
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
        unit = start.calendar.unit(by)
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
