"""Timezone helpers for Pythonic time vectors.

The R original exposes ``tz_name()``, ``tz_abbreviation()``, ``tz_offset()``,
and ``tz_transitions()``. This module keeps the same observable behaviour for
Gregorian chronons while returning ordinary Python lists, dictionaries, and
``Duration`` vectors.

Naive time has no timezone name. R uses ``NA_character_`` there; this port uses
``None`` for a missing timezone name and ``""`` for a missing abbreviation,
matching R's character-vector result for abbreviations.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .time import Duration, LinearTime, Mixtime, TimeVector, _is_missing
from .units import Unit, cal_gregorian

_EPOCH_DATE = _dt.date(1970, 1, 1)
_FIXED_SUBDAY = {
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
}
_CALENDAR_LIKE = {"year", "quarter", "month"}
_SECONDS_FACTOR = {
    "day": 86_400.0,
    "hour": 3_600.0,
    "minute": 60.0,
    "second": 1.0,
    "millisecond": 1e-3,
    "microsecond": 1e-6,
    "nanosecond": 1e-9,
}


def tz_name(x: Any) -> list[str | None] | str | None:
    """Return the timezone name for each element of ``x``.

    Time vectors return one name per element (``None`` when naive). A
    :class:`~mixtime.Unit` returns its scalar timezone. Python date/time
    objects and sequences are also accepted, which makes the helper useful
    alongside the time-vector API.
    """
    if isinstance(x, Mixtime):
        return [name for part in x.parts for name in tz_name(part)]
    if isinstance(x, TimeVector):
        name = _normalise_zone(x.timezone)
        return [name] * len(x)
    if isinstance(x, Unit):
        return _normalise_zone(x.timezone)
    if _is_sequence(x):
        return [_name_of(item) for item in x]
    return [_name_of(x)]


def tz_abbreviation(x: Any, tz: Any = None) -> list[str]:
    """Return the timezone abbreviation at each element of ``x``.

    ``tz`` may be a single zone name or a sequence of zone names. When omitted,
    each element uses its own timezone. Naive values and missing values return
    an empty string, matching R.
    """
    if isinstance(x, Mixtime):
        return _mixtime_abbreviation(x, tz)
    if isinstance(x, TimeVector):
        return _abbreviation_vector(x, tz)
    if isinstance(x, Unit):
        raise TypeError("tz_abbreviation() needs a time point, not a Unit")

    items = _as_items(x)
    default = [_name_of(item) for item in items]
    zones = _zone_values(tz, len(items), default)
    return [_abbreviation_one(item, zone) for item, zone in zip(items, zones)]


def tz_offset(x: Any, tz: Any = None) -> Duration | Mixtime:
    """Return the UTC offset as a duration in each element's chronon.

    For a second chronon the result is in seconds, for a day chronon in days,
    for a month chronon in months, and so on. Naive and UTC time have zero
    offsets. Mixed-granularity :class:`~mixtime.Mixtime` input returns a
    ``Mixtime`` of duration parts.
    """
    if isinstance(x, Mixtime):
        return _mixtime_offset(x, tz)
    if isinstance(x, TimeVector):
        return _offset_vector(x, tz)
    if isinstance(x, Unit):
        raise TypeError("tz_offset() needs a time point, not a Unit")

    items = _as_items(x)
    default = [_name_of(item) for item in items]
    zones = _zone_values(tz, len(items), default)
    has_datetime = any(_is_datetime_like(item) for item in items)
    if has_datetime:
        values = np.asarray(
            [_offset_one(item, zone) for item, zone in zip(items, zones)],
            dtype=np.float64,
        )
        return Duration(
            values,
            cal_gregorian.second(1),
            discrete=False,
            calendar=cal_gregorian,
        )

    # R's Date method always returns zero days.
    return Duration(
        np.zeros(len(items), dtype=np.float64),
        cal_gregorian.day(1),
        discrete=False,
        calendar=cal_gregorian,
    )


def tz_transitions(
    start: Any,
    end: Any,
    *,
    tz: str | None = None,
) -> dict[str, TimeVector]:
    """Find timezone transitions between ``start`` and ``end``.

    Returns a dictionary with:

    * ``time``: a continuous UTC-second :class:`~mixtime.LinearTime`;
    * ``offset_before`` / ``offset_after``: second :class:`~mixtime.Duration`
      vectors.

    The timezone is taken from ``start`` unless ``tz`` is supplied. A naive
    ``start`` requires an explicit ``tz`` because Python has no portable
    equivalent of R's ``Sys.timezone()`` default.
    """
    zone = _transition_zone(start, tz)
    begin = _transition_seconds(start, zone)
    finish = _transition_seconds(end, zone)
    if not np.isfinite(begin) or not np.isfinite(finish):
        raise ValueError("tz_transitions() needs finite start and end values")
    if finish <= begin:
        return _transitions_result([], [], [])

    transitions: list[tuple[float, float, float]] = []
    step = 3_600.0
    current = begin
    offset = _offset_seconds_at(current, zone)
    while current < finish:
        nxt = min(current + step, finish)
        next_offset = _offset_seconds_at(nxt, zone)
        if next_offset != offset:
            lo, hi = current, nxt
            while hi - lo > 1.0:
                mid = (lo + hi) / 2.0
                if _offset_seconds_at(mid, zone) == offset:
                    lo = mid
                else:
                    hi = mid
            before = _offset_seconds_at(hi - 1.0, zone)
            after = _offset_seconds_at(hi, zone)
            if before != after:
                transitions.append((hi, before, after))
            offset = after
            current = hi
        else:
            current = nxt

    return _transitions_result(
        [item[0] for item in transitions],
        [item[1] for item in transitions],
        [item[2] for item in transitions],
    )


# ---------------------------------------------------------------------------
# Time-vector helpers


def _abbreviation_vector(x: TimeVector, tz: Any) -> list[str]:
    default = [_normalise_zone(x.timezone)] * len(x)
    zones = _zone_values(tz, len(x), default)
    # R converts the chronon to UTC in its own timezone, then asks for the
    # abbreviation in the requested zone. A naive chronon is UTC here.
    source = _normalise_zone(x.timezone) or "UTC"
    out: list[str] = []
    for value, zone in zip(x.values, zones):
        if _is_missing(value) or zone in (None, ""):
            out.append("")
            continue
        start = _start_utc_seconds(value, x.chronon, x.calendar, source)
        out.append(_abbreviation_at(start, zone))
    return out


def _offset_vector(x: TimeVector, tz: Any) -> Duration:
    default = [_normalise_zone(x.timezone)] * len(x)
    zones = _zone_values(tz, len(x), default)
    values = np.zeros(len(x), dtype=np.float64)
    cardinality = _nominal_chronon_seconds(x.chronon)
    source = _normalise_zone(x.timezone) or "UTC"
    for index, (value, zone) in enumerate(zip(x.values, zones)):
        if _is_missing(value) or zone in (None, "", "UTC", "Etc/UTC"):
            continue
        start = _start_utc_seconds(value, x.chronon, x.calendar, source)
        offset = _offset_seconds_at(start, zone)
        values[index] = offset / cardinality
    return Duration(
        values,
        x.chronon,
        discrete=False,
        calendar=x.calendar,
        timezone=x.timezone,
    )


def _mixtime_abbreviation(x: Mixtime, tz: Any) -> list[str]:
    if not isinstance(tz, str) and tz is not None:
        zones = _zone_values(tz, len(x), [None] * len(x))
        out: list[str] = []
        position = 0
        for part in x.parts:
            end = position + len(part)
            out.extend(tz_abbreviation(part, zones[position:end]))
            position = end
        return out
    return [value for part in x.parts for value in tz_abbreviation(part, tz)]


def _mixtime_offset(x: Mixtime, tz: Any) -> Mixtime:
    if not isinstance(tz, str) and tz is not None:
        zones = _zone_values(tz, len(x), [None] * len(x))
        parts: list[TimeVector] = []
        position = 0
        for part in x.parts:
            end = position + len(part)
            parts.append(tz_offset(part, zones[position:end]))
            position = end
        return Mixtime(parts)
    return Mixtime([tz_offset(part, tz) for part in x.parts])


# ---------------------------------------------------------------------------
# Scalar temporal helpers


def _abbreviation_one(value: Any, zone: str | None) -> str:
    if _is_missing(value) or zone in (None, ""):
        return ""
    if _is_datetime_like(value):
        # A fixed-offset datetime may not have a ZoneInfo name. Its own
        # abbreviation is still correct when the requested zone is its own.
        if (
            isinstance(value, _dt.datetime)
            and value.tzinfo is not None
            and _name_of(value) == zone
        ):
            return value.tzname() or ""
        seconds = _datetime_seconds(value, zone)
    elif isinstance(value, _dt.date):
        seconds = _midnight_seconds(value, zone)
    else:
        return ""
    return _abbreviation_at(seconds, zone)


def _offset_one(value: Any, zone: str | None) -> float:
    if _is_missing(value) or zone in (None, ""):
        return 0.0
    if _is_datetime_like(value):
        if (
            isinstance(value, _dt.datetime)
            and value.tzinfo is not None
            and _name_of(value) == zone
        ):
            offset = value.utcoffset()
        else:
            seconds = _datetime_seconds(value, zone)
            offset = _dt.datetime.fromtimestamp(
                seconds, tz=ZoneInfo(zone)
            ).utcoffset()
        return 0.0 if offset is None else offset.total_seconds()
    if isinstance(value, _dt.date):
        return _offset_seconds_at(_midnight_seconds(value, zone), zone)
    return 0.0


def _name_of(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if _is_datetime_like(value):
        tzinfo = getattr(value, "tzinfo", None)
        return _name_from_tzinfo(tzinfo)
    return None


def _name_from_tzinfo(tzinfo: Any) -> str | None:
    if tzinfo is None:
        return None
    key = getattr(tzinfo, "key", None)
    if isinstance(key, str) and key:
        return key
    if (
        isinstance(tzinfo, _dt.timezone)
        and tzinfo.utcoffset(None) == _dt.timedelta(0)
    ):
        return "UTC"
    name = tzinfo.tzname(None) if hasattr(tzinfo, "tzname") else None
    return name or None


# ---------------------------------------------------------------------------
# Chronon start and offset helpers


def _start_utc_seconds(
    value: Any,
    chronon: Unit,
    calendar: Any,
    zone: str,
) -> float:
    name = chronon.name
    if name in _FIXED_SUBDAY:
        return float(value) * _unit_seconds(chronon)
    if name == "day":
        days = int(np.floor(float(value) * float(chronon.n)))
        date = _EPOCH_DATE + _dt.timedelta(days=days)
        return _midnight_seconds(date, zone)
    if name in _CALENDAR_LIKE or name == "week":
        day_value = calendar.convert(
            np.asarray([float(value)]),
            chronon,
            calendar.day,
            discrete=True,
        )[0]
        date = _EPOCH_DATE + _dt.timedelta(days=int(np.floor(day_value)))
        return _midnight_seconds(date, zone)
    return float(value) * _unit_seconds(chronon)


def _midnight_seconds(date: _dt.date, zone: str) -> float:
    return _dt.datetime.combine(
        date, _dt.time(), tzinfo=ZoneInfo(zone)
    ).timestamp()


def _datetime_seconds(value: Any, zone: str) -> float:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=ZoneInfo(zone))
        return value.timestamp()
    # pandas.Timestamp is a datetime subclass, but keep this fallback for
    # other datetime-like objects with a timestamp() method.
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        return float(timestamp())
    raise TypeError(f"Unsupported datetime value: {type(value)!r}")


def _offset_seconds_at(seconds: float, zone: str) -> float:
    offset = _dt.datetime.fromtimestamp(
        float(seconds), tz=ZoneInfo(zone)
    ).utcoffset()
    return 0.0 if offset is None else offset.total_seconds()


def _abbreviation_at(seconds: float, zone: str) -> str:
    value = _dt.datetime.fromtimestamp(float(seconds), tz=ZoneInfo(zone))
    return value.tzname() or ""


def _unit_seconds(unit: Unit) -> float:
    factor = _SECONDS_FACTOR.get(unit.name)
    if factor is None:
        raise ValueError(f"Cannot express {unit!r} in seconds")
    return float(unit.n) * factor


def _nominal_chronon_seconds(unit: Unit) -> float:
    """R's nominal chronon cardinality used to scale UTC offsets."""
    name = unit.name
    n = float(unit.n)
    if name == "year":
        return 365.0 * 86_400.0 * n
    if name == "quarter":
        return 365.0 / 4.0 * 86_400.0 * n
    if name == "month":
        return 31.0 * 86_400.0 * n
    if name == "week":
        return 7.0 * 86_400.0 * n
    return _unit_seconds(unit)


# ---------------------------------------------------------------------------
# Transition helpers


def _transition_zone(start: Any, tz: str | None) -> str:
    if tz is not None:
        zone = _normalise_zone(tz)
        if zone is None:
            raise ValueError("tz_transitions() needs a named timezone")
        return zone
    if isinstance(start, Mixtime):
        if not start.parts:
            raise ValueError("tz_transitions() needs a timezone")
        start = start.parts[0]
    if isinstance(start, TimeVector):
        zone = _normalise_zone(start.timezone)
        if zone is not None:
            return zone
        raise ValueError(
            "tz_transitions() needs a timezone-aware start or an explicit tz"
        )
    zone = _name_of(start)
    if zone is None:
        raise ValueError(
            "tz_transitions() needs a timezone-aware start or an explicit tz"
        )
    return zone


def _transition_seconds(value: Any, zone: str) -> float:
    if isinstance(value, Mixtime):
        if len(value) != 1:
            raise ValueError("tz_transitions() expects length-one start and end")
        return _transition_seconds(value.parts[0], zone)
    if isinstance(value, TimeVector):
        if len(value) != 1:
            raise ValueError("tz_transitions() expects length-one start and end")
        raw = value.values[0]
        if _is_missing(raw):
            return float("nan")
        return _start_utc_seconds(raw, value.chronon, value.calendar, zone)
    if _is_missing(value):
        return float("nan")
    if _is_datetime_like(value):
        return _datetime_seconds(value, zone)
    if isinstance(value, _dt.date):
        return _midnight_seconds(value, zone)
    raise TypeError(
        "tz_transitions() expects datetimes, dates, or length-one time vectors"
    )


def _transitions_result(
    times: list[float],
    before: list[float],
    after: list[float],
) -> dict[str, TimeVector]:
    return {
        "time": LinearTime(
            np.asarray(times, dtype=np.float64),
            cal_gregorian.second(1, tz="UTC"),
            discrete=False,
            calendar=cal_gregorian,
            timezone="UTC",
        ),
        "offset_before": Duration(
            np.asarray(before, dtype=np.float64),
            cal_gregorian.second(1),
            discrete=False,
            calendar=cal_gregorian,
        ),
        "offset_after": Duration(
            np.asarray(after, dtype=np.float64),
            cal_gregorian.second(1),
            discrete=False,
            calendar=cal_gregorian,
        ),
    }


# ---------------------------------------------------------------------------
# Small utilities


def _normalise_zone(zone: Any) -> str | None:
    if zone is None:
        return None
    text = str(zone)
    return text or None


def _zone_values(tz: Any, n: int, default: list[str | None]) -> list[str | None]:
    if n == 0:
        return []
    if tz is None:
        values = list(default)
    elif isinstance(tz, str):
        values = [_normalise_zone(tz)] * n
    else:
        values = [_normalise_zone(item) for item in tz]
    if not values:
        raise ValueError("tz must provide at least one timezone")
    return [values[index % len(values)] for index in range(n)]


def _as_items(value: Any) -> list[Any]:
    if _is_sequence(value):
        return list(value)
    return [value]


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, _dt.date, _dt.datetime, np.datetime64)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    if isinstance(value, (list, tuple)):
        return True
    if (
        type(value).__module__.startswith("pandas")
        and getattr(value, "ndim", None) == 1
    ):
        return True
    return False


def _is_datetime_like(value: Any) -> bool:
    if isinstance(value, _dt.datetime):
        return True
    if isinstance(value, np.datetime64):
        return False
    return (
        type(value).__module__.startswith("pandas")
        and type(value).__name__ == "Timestamp"
    )
