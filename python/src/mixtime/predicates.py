"""Predicates for time-vector types and granularity.

These are Pythonic counterparts of R mixtime's ``time_is_*()`` helpers. Boolean
vectors use ``True``/``False`` and ``None`` for missing or infinite values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .time import CyclicalTime, Duration, LinearTime, Mixtime, TimeVector
from .units import Unit, _nests_in


def time_is_linear(x: Any) -> np.ndarray:
    """Return a boolean vector identifying linear time elements."""
    return _type_mask(x, LinearTime)


def time_is_cyclical(x: Any) -> np.ndarray:
    """Return a boolean vector identifying cyclical time elements."""
    return _type_mask(x, CyclicalTime)


def time_is_duration(x: Any) -> np.ndarray:
    """Return a boolean vector identifying duration elements."""
    return _type_mask(x, Duration)


def time_is_determinate_at(x: Any, granule: Unit | str | Duration) -> np.ndarray:
    """Test whether each element is determinate at ``granule``.

    A discrete time point cannot resolve a granule finer than its own chronon;
    continuous time points resolve every granule. Missing and infinite values
    return ``None``.
    """
    if isinstance(x, Mixtime):
        if not x.parts:
            return np.asarray([], dtype=object)
        return np.concatenate(
            [time_is_determinate_at(part, granule) for part in x.parts]
        )
    if not isinstance(x, TimeVector):
        raise TypeError("time_is_determinate_at() expects a TimeVector or Mixtime")

    unit = _as_granule(granule, x.calendar)
    values = np.asarray(x.values, dtype=float)
    out = np.empty(len(x), dtype=object)
    special = ~np.isfinite(values)
    out[:] = not x.discrete or _nests_in(x.chronon, unit)
    out[special] = None
    return out


def time_is_complete_at(x: Any, granule: Unit | str | Duration) -> np.ndarray:
    """Test whether each element's coarser granule is fully observed.

    Completeness is a property of the whole vector: every finer chronon making
    up the granule must be present somewhere in ``x``. Mixed-granularity
    ``Mixtime`` inputs are rejected, matching R's current restriction.
    """
    if isinstance(x, Mixtime):
        if len(x.parts) > 1:
            raise ValueError(
                "time_is_complete_at() does not support mixed-granularity Mixtime"
            )
        if not x.parts:
            return np.asarray([], dtype=object)
        return time_is_complete_at(x.parts[0], granule)
    if not isinstance(x, TimeVector):
        raise TypeError("time_is_complete_at() expects a TimeVector or Mixtime")

    unit = _as_granule(granule, x.calendar)
    values = np.asarray(x.values, dtype=float)
    out = np.empty(len(x), dtype=object)
    special = ~np.isfinite(values)
    out[special] = None
    keep = ~special
    if not np.any(keep):
        return out

    if not _nests_in(x.chronon, unit):
        out[keep] = False
        return out

    positions = np.floor(values[keep]).astype(np.int64)
    groups = np.asarray(
        x.calendar.convert(
            positions.astype(float), x.chronon, unit, discrete=True
        ),
        dtype=np.int64,
    )
    unique_groups, inverse = np.unique(groups, return_inverse=True)

    starts = np.asarray(
        x.calendar.convert(
            unique_groups.astype(float), unit, x.chronon, discrete=True
        ),
        dtype=float,
    )
    ends = np.asarray(
        x.calendar.convert(
            (unique_groups + 1).astype(float), unit, x.chronon, discrete=True
        ),
        dtype=float,
    )
    expected = np.rint(ends - starts).astype(np.int64)

    observed = np.zeros(len(unique_groups), dtype=np.int64)
    for i in range(len(unique_groups)):
        observed[i] = len(np.unique(positions[inverse == i]))

    complete = observed >= expected
    out[keep] = complete[inverse]
    return out


def _type_mask(x: Any, cls: type[TimeVector]) -> np.ndarray:
    if isinstance(x, Mixtime):
        if not x.parts:
            return np.asarray([], dtype=bool)
        return np.concatenate([_type_mask(part, cls) for part in x.parts])
    if isinstance(x, TimeVector):
        return np.full(len(x), isinstance(x, cls), dtype=bool)
    try:
        length = len(x)
    except TypeError:
        length = 1
    return np.zeros(length, dtype=bool)


def _as_granule(granule: Unit | str | Duration, calendar) -> Unit:
    if isinstance(granule, str):
        return calendar.unit(granule)
    if isinstance(granule, Duration):
        if len(granule) != 1:
            raise ValueError("granule duration must have length 1")
        unit = granule.chronon
        size = float(unit.n) * float(granule.values[0])
        if size == 0:
            raise ValueError("granule duration must not be zero")
        return Unit(unit.name, size, calendar, unit.timezone)
    if isinstance(granule, Unit):
        if float(granule.n) == 0:
            raise ValueError("granule unit must not be zero")
        return granule
    raise TypeError(f"Expected a Unit, unit name, or Duration; got {type(granule)!r}")
