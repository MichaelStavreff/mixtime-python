"""Metadata accessors for time vectors.

R's ``time_chronon()``, ``time_cycle()``, and ``time_calendar()`` return
duration vectors and calendar objects. This port exposes the underlying
metadata directly: a :class:`~mixtime.Unit` for a chronon or cycle and a
:class:`~mixtime.Calendar` for a calendar. A :class:`~mixtime.Mixtime` returns
per-part metadata where the parts can differ.

The Python return types are intentionally different from R's duration-vector
returns. The chronon and cycle values themselves are the same.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np

from .time import Mixtime, TimeVector
from .units import Calendar, Unit, cal_gregorian


def is_mixtime(x: Any) -> bool:
    """Return ``True`` for a mixed-granularity :class:`~mixtime.Mixtime`."""
    return isinstance(x, Mixtime)


def time_chronon(x: Any) -> Unit | list[Unit] | None:
    """Return the chronon (smallest granule) of a time object.

    A single time vector returns its ``Unit``. A ``Mixtime`` returns one
    ``Unit`` per part. Python ``date`` and ``datetime`` values return the
    one-day and one-second Gregorian chronons respectively; sequences are not
    interval-inferred.
    """
    if isinstance(x, Mixtime):
        return [_chronon_of(part) for part in x.parts]
    if isinstance(x, TimeVector):
        return x.chronon
    if isinstance(x, Unit):
        return x
    if isinstance(x, _dt.datetime):
        return cal_gregorian.second
    if isinstance(x, _dt.date):
        return cal_gregorian.day
    if isinstance(x, np.datetime64):
        return cal_gregorian.second
    return None


def time_cycle(x: Any) -> Unit | None | list[Unit | None]:
    """Return the cyclical granule of a time object, or ``None`` if absent.

    A ``Mixtime`` returns one entry per part.
    """
    if isinstance(x, Mixtime):
        return [_cycle_of(part) for part in x.parts]
    if isinstance(x, TimeVector):
        return x.cycle
    if isinstance(x, Unit):
        return None
    return None


def time_calendar(x: Any) -> Calendar | list[Calendar]:
    """Return the calendar of a time object.

    Python currently implements one Gregorian calendar. A ``Mixtime`` whose
    parts share a calendar returns that calendar; otherwise a per-part list is
    returned.
    """
    if isinstance(x, Mixtime):
        calendars = [_calendar_of(part) for part in x.parts]
        if not calendars:
            return cal_gregorian
        first = calendars[0]
        if all(calendar is first for calendar in calendars[1:]):
            return first
        return calendars
    if isinstance(x, TimeVector):
        return x.calendar
    if isinstance(x, Unit):
        return x.calendar or cal_gregorian
    return cal_gregorian


def _chronon_of(x: TimeVector) -> Unit:
    return x.chronon


def _cycle_of(x: TimeVector) -> Unit | None:
    return x.cycle


def _calendar_of(x: TimeVector) -> Calendar:
    return x.calendar
