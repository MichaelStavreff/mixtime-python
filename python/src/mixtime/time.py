"""Pythonic time vectors for the Gregorian calendar."""

from __future__ import annotations

import datetime as _dt
import re
from calendar import monthrange
from typing import Any, Iterable, Iterator
from zoneinfo import ZoneInfo

import numpy as np

from .parse import (
    date_to_days,
    datetime_to_seconds,
    parse_date,
    parse_datetime,
    parse_month,
    parse_quarter,
    parse_time,
    parse_week,
    parse_year,
)
from .units import Calendar, Unit, cal_gregorian

_EPOCH_DATE = _dt.date(1970, 1, 1)
_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTH_FULL = ("January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December")
_WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_WEEKDAY_FULL = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday")
_UTC_TIMEZONES = {None, "", "UTC", "Etc/UTC"}
_CALENDAR_SHIFT_UNITS = {"day", "week", "month", "quarter", "year"}
_MONTHS_PER_UNIT = {"month": 1, "quarter": 3, "year": 12}


def _timezone_is_utc(tz: str | None) -> bool:
    return tz in _UTC_TIMEZONES


class TimeVector:
    """A one-dimensional vector of time points or durations.

    Values are stored as chronon counts. For linear and cyclical time, the
    epoch is 1970-01-01; for durations, values are plain counts of ``chronon``.
    """

    kind = "linear"

    def __init__(
        self,
        values: Any,
        chronon: Unit,
        *,
        discrete: bool = True,
        calendar: Calendar | None = None,
        cycle: Unit | None = None,
        timezone: str | None = None,
    ) -> None:
        self.calendar = calendar or chronon.calendar or cal_gregorian
        self.chronon = _normalize_unit(chronon, self.calendar)
        self.cycle = _normalize_unit(cycle, self.calendar) if cycle is not None else None
        self.discrete = bool(discrete)
        self.timezone = timezone or self.chronon.timezone
        self.values = _coerce_values(values, self.discrete)

    # -- sequence protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int | slice) -> "TimeVector":
        return self._new(self.values[index])

    def __iter__(self) -> Iterator["TimeVector"]:
        for index in range(len(self)):
            yield self[index]

    def scalar(self, index: int) -> Any:
        """Return one element as a :class:`~mixtime.scalar.TimeScalar`.

        ``x[i]`` deliberately keeps returning a length-one vector for
        compatibility with R-style extraction.  Use this method when a real
        scalar is needed, for example for pandas.
        """
        from .scalar import TimeScalar

        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("TimeVector index out of range")
        return TimeScalar.from_vector(self, self.values[index])

    def isna(self) -> np.ndarray:
        """Return a boolean mask for missing values."""
        return np.asarray([_is_missing(value) for value in self.values], dtype=bool)

    def notna(self) -> np.ndarray:
        """Return the complement of :meth:`isna`."""
        return np.logical_not(self.isna())

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if dtype is None:
            return np.asarray(self.values)
        return np.asarray(self.values, dtype=dtype)

    def _new(self, values: Any) -> "TimeVector":
        return type(self)(
            values,
            self.chronon,
            discrete=self.discrete,
            calendar=self.calendar,
            cycle=self.cycle,
            timezone=self.timezone,
        )

    # -- display -----------------------------------------------------------
    def format(self, fmt: str | None = None) -> list[str]:
        if fmt is None:
            return [self._format_one(value) for value in self.values]
        return [self._format_with(value, fmt) for value in self.values]

    def __str__(self) -> str:
        return " ".join(self.format())

    def __repr__(self) -> str:
        body = ", ".join(self.format())
        return f"<{type(self).__name__}({self.chronon.name})[{len(self)}]: {body}>"

    # -- conversion --------------------------------------------------------
    def astype(self, chronon: Unit | str, *, discrete: bool | None = None) -> "TimeVector":
        unit = _normalize_unit(chronon, self.calendar)
        target_discrete = self.discrete if discrete is None else discrete
        convert_discrete = target_discrete and self.kind != "duration"
        values = self.calendar.convert(
            self.values,
            self.chronon,
            unit,
            discrete=convert_discrete,
            duration=self.kind == "duration",
            anchored=self.kind != "duration",
        )
        if self.kind == "duration" and target_discrete:
            values = _coerce_values(np.trunc(values), True)
        return type(self)(
            values,
            unit,
            discrete=target_discrete,
            calendar=self.calendar,
            cycle=self.cycle,
            timezone=self.timezone,
        )

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other: Any) -> "TimeVector":
        if isinstance(other, Duration):
            if isinstance(self, Duration):
                unit = _common_unit(self.chronon, other.chronon, duration=True)
                left = self.calendar.convert(
                    self.values,
                    self.chronon,
                    unit,
                    discrete=False,
                    duration=True,
                    anchored=False,
                )
                right = self.calendar.convert(
                    other.values,
                    other.chronon,
                    unit,
                    discrete=False,
                    duration=True,
                    anchored=False,
                )
                # Duration arithmetic returns a continuous duration in R,
                # even when both inputs are whole-granule durations.
                return Duration(
                    _coerce_values(left + right, False),
                    unit,
                    discrete=False,
                    calendar=self.calendar,
                )
            shifted = self._calendar_shift_values(other, 1)
            if shifted is not None:
                return self._new(_coerce_values(shifted, self.discrete))
            values = self._duration_values_in_chronon(other)
            return self._new(_coerce_values(self.values + values, self.discrete))
        if isinstance(self, Duration) and isinstance(other, TimeVector):
            return other + self
        if isinstance(self, Duration) and np.isscalar(other):
            return Duration(
                _coerce_values(self.values + float(other), False),
                self.chronon,
                discrete=False,
                calendar=self.calendar,
            )
        return NotImplemented

    def __radd__(self, other: Any) -> "TimeVector":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "TimeVector":
        if isinstance(self, Duration) and isinstance(other, Duration):
            unit = _common_unit(self.chronon, other.chronon, duration=True)
            left = self.calendar.convert(
                self.values,
                self.chronon,
                unit,
                discrete=False,
                duration=True,
                anchored=False,
            )
            right = self.calendar.convert(
                other.values,
                other.chronon,
                unit,
                discrete=False,
                duration=True,
                anchored=False,
            )
            # A difference between time points is a continuous duration, even
            # when both time points are discrete (matching R's double result).
            return Duration(
                _coerce_values(left - right, False),
                unit,
                discrete=False,
                calendar=self.calendar,
            )
        if isinstance(other, Duration) and not isinstance(self, Duration):
            shifted = self._calendar_shift_values(other, -1)
            if shifted is not None:
                return self._new(_coerce_values(shifted, self.discrete))
            values = self._duration_values_in_chronon(other)
            return self._new(_coerce_values(self.values - values, self.discrete))
        if isinstance(self, TimeVector) and isinstance(other, TimeVector) and not isinstance(self, Duration):
            if self.calendar is not other.calendar:
                raise ValueError("Cannot subtract time vectors from different calendars")
            if (
                self.chronon.name != other.chronon.name
                or float(self.chronon.n) != float(other.chronon.n)
            ) and (self.discrete or other.discrete):
                raise ValueError(
                    "The difference between time points of different chronons is "
                    "indeterminate when either is discrete"
                )
            unit = _common_unit(self.chronon, other.chronon, duration=False)
            left = self.calendar.convert(
                self.values, self.chronon, unit, discrete=False
            )
            right = self.calendar.convert(
                other.values, other.chronon, unit, discrete=False
            )
            # A difference between time points is a continuous duration, even
            # when both time points are discrete (matching R's double result).
            return Duration(
                _coerce_values(left - right, False),
                unit,
                discrete=False,
                calendar=self.calendar,
            )
        if isinstance(self, Duration) and np.isscalar(other):
            return Duration(
                _coerce_values(self.values - float(other), False),
                self.chronon,
                discrete=False,
                calendar=self.calendar,
            )
        return NotImplemented

    def _duration_values_in_chronon(self, other: "Duration") -> np.ndarray:
        _check_duration_shift(self.chronon, other.chronon, self.discrete)
        values = self.calendar.convert(
            other.values,
            other.chronon,
            self.chronon,
            discrete=False,
            duration=True,
            anchored=True,
        )
        if self.discrete:
            _check_discrete_shift(self.chronon, other.chronon, values)
        return values

    def _calendar_shift_values(
        self, other: "Duration", sign: int
    ) -> np.ndarray | None:
        """Shift dates and datetimes with calendar-field arithmetic.

        Day-or-longer durations keep the local wall-clock reading for
        datetimes, which is what R does in a time zone. Month/year shifts clamp
        the day to the last valid day of the target month. Returns ``None``
        when the ordinary chronon conversion path should be used instead.
        """
        if self.kind != "linear" or isinstance(self, Duration):
            return None
        duration_name = other.chronon.name
        if duration_name not in _CALENDAR_SHIFT_UNITS:
            return None
        time_name = self.chronon.name
        if time_name == "day":
            return _shift_date_values(self.values, other, sign, self.discrete)
        if time_name == "second":
            return _shift_datetime_values(
                self.values, other, sign, self.timezone
            )
        return None

    def __mul__(self, other: Any) -> "Duration":
        if isinstance(self, Duration) and np.isscalar(other):
            return Duration(
                _coerce_values(self.values * float(other), False),
                self.chronon,
                discrete=False,
                calendar=self.calendar,
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> "Duration":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Duration":
        if isinstance(self, Duration) and np.isscalar(other):
            return Duration(
                _coerce_values(self.values / float(other), False),
                self.chronon,
                discrete=False,
                calendar=self.calendar,
            )
        return NotImplemented

    def __neg__(self) -> "Duration":
        if isinstance(self, Duration):
            return Duration(
                _coerce_values(-self.values, False),
                self.chronon,
                discrete=False,
                calendar=self.calendar,
            )
        return NotImplemented

    def __eq__(self, other: Any) -> np.ndarray:
        if isinstance(other, TimeVector):
            return self._compare(other, "==")
        if _is_time_scalar(other):
            return self._compare(other.to_vector(), "==")
        return NotImplemented

    def __ne__(self, other: Any) -> np.ndarray:
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return np.logical_not(equal)

    def __lt__(self, other: Any) -> np.ndarray:
        return self._compare(other, "<")

    def __le__(self, other: Any) -> np.ndarray:
        return self._compare(other, "<=")

    def __gt__(self, other: Any) -> np.ndarray:
        return self._compare(other, ">")

    def __ge__(self, other: Any) -> np.ndarray:
        return self._compare(other, ">=")

    def _compare(self, other: Any, op: str) -> np.ndarray:
        if _is_time_scalar(other):
            other = other.to_vector()
        if not isinstance(other, TimeVector):
            raise TypeError(f"Cannot compare TimeVector with {type(other)!r}")
        if isinstance(self, Duration) != isinstance(other, Duration):
            raise TypeError("Cannot compare a duration with a time point")
        if self.kind != other.kind:
            raise TypeError(
                f"Cannot compare {self.kind} time with {other.kind} time"
            )

        if op == "!=":
            return np.logical_not(self._compare(other, "=="))

        self_missing = self.isna()
        other_missing = other.isna()
        missing = self_missing | other_missing

        if isinstance(self, Duration):
            common = _common_unit(self.chronon, other.chronon, duration=True)
            left = _convert_defined(
                self.values,
                self.chronon,
                common,
                self.calendar,
                duration=True,
                anchored=False,
            )
            right = _convert_defined(
                other.values,
                other.chronon,
                common,
                other.calendar,
                duration=True,
                anchored=False,
            )
            result = _duration_relation(op, left, right)
        else:
            common = _comparison_unit(self.chronon, other.chronon)
            if self.kind == "cyclical":
                cycle = _common_cycle([self, other])
                assert cycle is not None
                x_start, x_end = _cyclical_interval(
                    self.values,
                    self.chronon,
                    common,
                    cycle,
                    self.calendar,
                    self.discrete,
                    self.timezone,
                )
                y_start, y_end = _cyclical_interval(
                    other.values,
                    other.chronon,
                    common,
                    cycle,
                    other.calendar,
                    other.discrete,
                    other.timezone,
                )
            else:
                x_start, x_end = _linear_interval(
                    self.values,
                    self.chronon,
                    common,
                    self.calendar,
                    self.discrete,
                )
                y_start, y_end = _linear_interval(
                    other.values,
                    other.chronon,
                    common,
                    other.calendar,
                    other.discrete,
                )
            result = _relation(op, x_start, x_end, y_start, y_end)

        result = np.asarray(result, dtype=bool)
        if op == "==":
            result = np.where(
                self_missing & other_missing,
                True,
                np.where(missing, False, result),
            )
        else:
            result = np.where(missing, False, result)
        return result

    def to_pandas(self):
        """Return a pandas index for linear Gregorian time.

        pandas is optional; importing this method without pandas raises a
        clear ImportError.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "pandas is required for to_pandas(); install mixtime[pandas]"
            ) from exc

        if self.kind != "linear":
            raise TypeError("to_pandas() is only implemented for linear time")

        name = self.chronon.name
        if name == "year":
            periods = [
                pd.Period(year=int(np.floor(value)) + 1970, freq="Y")
                for value in self.values
            ]
            return pd.PeriodIndex(periods, freq="Y")
        if name == "quarter":
            periods = []
            for value in self.values:
                year = int(np.floor(value / 4.0)) + 1970
                quarter = int(np.floor(value) % 4) + 1
                periods.append(pd.Period(year=year, quarter=quarter, freq="Q"))
            return pd.PeriodIndex(periods, freq="Q")
        if name == "month":
            periods = []
            for value in self.values:
                year = int(np.floor(value / 12.0)) + 1970
                month = int(np.floor(value) % 12) + 1
                periods.append(pd.Period(year=year, month=month, freq="M"))
            return pd.PeriodIndex(periods, freq="M")
        if name == "week":
            starts = [
                _date_from_day(float(value) * 7.0 - 3.0) for value in self.values
            ]
            return pd.PeriodIndex(starts, freq="W-SUN")
        if name == "day":
            return pd.DatetimeIndex([_date_from_day(value) for value in self.values])
        if name in {"second", "millisecond", "microsecond", "nanosecond"}:
            if self.timezone in (None, "", "UTC", "Etc/UTC"):
                values = [
                    _datetime_from_seconds(value, "UTC").replace(tzinfo=None)
                    for value in self.values
                ]
            else:
                values = [
                    _datetime_from_seconds(value, self.timezone)
                    for value in self.values
                ]
            return pd.DatetimeIndex(values)
        raise NotImplementedError(f"to_pandas() is not implemented for {name}")

    def to_pandas_series(self, name: str | None = None):
        """Return this vector as a pandas Series with ``MixtimeDtype``.

        This is separate from :meth:`to_pandas`, which returns a pandas
        ``PeriodIndex``/``DatetimeIndex`` for linear time.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "pandas is required for to_pandas_series(); "
                "install mixtime[pandas]"
            ) from exc

        from .pandas_ext import MixtimeArray

        return pd.Series(MixtimeArray(self), name=name)

    # -- default formatting ------------------------------------------------
    def _format_one(self, value: Any) -> str:
        if _is_missing(value):
            return "NA"
        name = self.chronon.name
        if self.kind == "duration":
            return _format_duration_vector(
                np.asarray([value]), self.chronon, self.discrete
            )[0]
        if self.kind == "cyclical":
            return self._format_cyclical(value)
        return self._format_linear(value)

    def _format_linear(self, value: Any) -> str:
        name = self.chronon.name
        fractional = (not self.discrete) or (not _is_integral(value))
        frac = float(value) - np.floor(float(value)) if fractional else 0.0
        suffix = f" {frac * 100:.1f}%" if fractional else ""

        if name == "year":
            return _format_year_value(value, continuous=not self.discrete)
        if name == "quarter":
            year, quarter = _quarter_parts(value)
            return f"{year} Q{quarter}{suffix}"
        if name == "month":
            year, month = _month_parts(value)
            return f"{year} {_MONTH_ABBR[month - 1]}{suffix}"
        if name == "day":
            return f"{_date_from_day(value).isoformat()}{suffix}"
        if name in {"second", "millisecond", "microsecond", "nanosecond"}:
            local = _datetime_from_seconds(value, self.timezone)
            result = f"{local.strftime('%Y-%m-%d %H:%M:%S')}{suffix}"
            if not _timezone_is_utc(self.timezone):
                abbreviation = local.tzname()
                if abbreviation:
                    result = f"{result} {abbreviation}"
            return result
        if name == "week":
            iso = _date_from_day(float(value) * 7.0).isocalendar()
            return f"{iso.year} W{iso.week:02d}{suffix}"
        return str(value)

    def _format_cyclical(self, value: Any) -> str:
        if self.cycle is None:
            return self._format_linear(value)
        chronon = self.chronon.name
        cycle = self.cycle.name
        mod = _cycle_offset(value, self.chronon, self.cycle, self.calendar, self.timezone)
        suffix = ""
        if not self.discrete:
            fraction = float(mod) - np.floor(float(mod))
            suffix = f" {fraction * 100:.1f}%"

        if chronon == "month" and cycle == "year":
            return f"{_MONTH_ABBR[int(np.floor(mod)) % 12]}{suffix}"
        if chronon == "quarter" and cycle == "year":
            return f"Q{int(np.floor(mod)) + 1}{suffix}"
        if chronon == "day" and cycle == "year":
            return f"D{int(np.floor(mod)) + 1:02d}{suffix}"
        if chronon == "day" and cycle == "month":
            return f"D{int(np.floor(mod)) + 1:02d}{suffix}"
        if chronon == "day" and cycle == "week":
            return f"{_WEEKDAY_ABBR[int(np.floor(mod)) % 7]}{suffix}"
        if chronon == "week" and cycle == "year":
            return f"W{int(np.floor(mod)) + 1:02d}{suffix}"
        if chronon == "month" and cycle == "quarter":
            return f"M{int(np.floor(mod))}{suffix}"
        if chronon == "hour" and cycle == "day":
            return f"h{int(np.floor(mod)) % 24:02d}{suffix}"
        if chronon == "minute" and cycle == "hour":
            return f"m{int(np.floor(mod)) % 60:02d}{suffix}"
        if chronon == "second" and cycle == "minute":
            return f"s{int(np.floor(mod)) % 60:02d}{suffix}"
        if chronon == "second" and cycle == "day":
            seconds = int(np.floor(mod)) % 86_400
            hour, rem = divmod(seconds, 3600)
            minute, second = divmod(rem, 60)
            return f"{hour:02d}:{minute:02d}:{second:02d}{suffix}"
        return str(value)

    def _format_with(self, value: Any, fmt: str) -> str:
        if _is_missing(value):
            return "NA"
        return _format_tokens(value, self, fmt)


class LinearTime(TimeVector):
    kind = "linear"


class CyclicalTime(TimeVector):
    kind = "cyclical"

    def __init__(self, values: Any, chronon: Unit, cycle: Unit, **kwargs: Any) -> None:
        super().__init__(values, chronon, cycle=cycle, **kwargs)


class Duration(TimeVector):
    kind = "duration"

    def format(self, fmt: str | None = None) -> list[str]:
        if fmt is None:
            return _format_duration_vector(
                self.values, self.chronon, self.discrete
            )
        return super().format(fmt)


class Mixtime:
    """A mixed-granularity collection of time vectors.

    Each part keeps its own chronon, cycle, calendar, and timezone. This is the
    Python equivalent of mixtime's ``vecvec`` container, without trying to
    reproduce the R class system.
    """

    def __init__(self, parts: Iterable[TimeVector]) -> None:
        self.parts = [_as_time_vector(part) for part in parts]

    def __len__(self) -> int:
        return sum(len(part) for part in self.parts)

    def __getitem__(self, index: int | slice) -> "Mixtime":
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                parts: list[TimeVector] = []
                for position in range(start, stop, step):
                    parts.extend(self[position].parts)
                return Mixtime(parts)
            return self._slice(start, stop)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Mixtime index out of range")
        return self._slice(index, index + 1)

    def scalar(self, index: int) -> Any:
        """Return one element as a :class:`~mixtime.scalar.TimeScalar`."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Mixtime index out of range")
        offset = 0
        for part in self.parts:
            if index < offset + len(part):
                return part.scalar(index - offset)
            offset += len(part)
        raise IndexError("Mixtime index out of range")

    def isna(self) -> np.ndarray:
        """Return a boolean mask for missing values."""
        if not self.parts:
            return np.asarray([], dtype=bool)
        return np.concatenate([part.isna() for part in self.parts])

    def notna(self) -> np.ndarray:
        """Return the complement of :meth:`isna`."""
        return np.logical_not(self.isna())

    def take(
        self,
        indices: Any,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> "Mixtime":
        """Select positions, optionally filling ``-1`` with a missing value."""
        from .scalar import TimeScalar, _mixtime_from_scalars

        positions = np.asarray(indices, dtype=np.int64).ravel()
        if not allow_fill:
            return _mixtime_from_scalars(self.scalar(int(i)) for i in positions)

        fill = (
            fill_value
            if isinstance(fill_value, TimeScalar)
            else TimeScalar.missing()
        )
        scalars = [
            fill if int(i) == -1 else self.scalar(int(i)) for i in positions
        ]
        return _mixtime_from_scalars(scalars)

    def to_pandas_series(self, name: str | None = None):
        """Return this mixed container as a pandas Series with ``MixtimeDtype``."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "pandas is required for to_pandas_series(); "
                "install mixtime[pandas]"
            ) from exc

        from .pandas_ext import MixtimeArray

        return pd.Series(MixtimeArray(self), name=name)

    def __add__(self, other: Any) -> "Mixtime":
        if isinstance(other, Duration):
            return Mixtime([part + other for part in self.parts])
        return NotImplemented

    def __radd__(self, other: Any) -> "Mixtime":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Mixtime":
        if isinstance(other, Duration):
            return Mixtime([part - other for part in self.parts])
        return NotImplemented

    def _operand_scalars(self, other: Any) -> list[Any] | None:
        if isinstance(other, Mixtime):
            _mixtime_mode(other.parts)
            return [other.scalar(i) for i in range(len(other))]
        if isinstance(other, TimeVector):
            return [other.scalar(i) for i in range(len(other))]
        if _is_time_scalar(other):
            return [other]
        return None

    def _compare_scalars_against(
        self, operands: list[Any], op: str
    ) -> np.ndarray:
        _mixtime_mode(self.parts)
        n_self = len(self)
        n_other = len(operands)
        if n_self == 0 or n_other == 0:
            if n_self != n_other:
                raise ValueError(
                    f"Can't recycle a Mixtime of length {n_self} to match "
                    f"length {n_other}."
                )
            return np.asarray([], dtype=bool)
        if n_self == n_other:
            pairs = zip(
                (self.scalar(i) for i in range(n_self)), operands
            )
        elif n_other == 1:
            right = operands[0]
            pairs = ((self.scalar(i), right) for i in range(n_self))
        elif n_self == 1:
            left = self.scalar(0)
            pairs = ((left, right) for right in operands)
        else:
            raise ValueError(
                f"Can't recycle a Mixtime of length {n_self} to match "
                f"length {n_other}."
            )
        return np.asarray(
            [_compare_scalars(left, right, op) for left, right in pairs],
            dtype=bool,
        )

    def __eq__(self, other: Any) -> np.ndarray:
        operands = self._operand_scalars(other)
        if operands is None:
            return NotImplemented
        return self._compare_scalars_against(operands, "==")

    def __ne__(self, other: Any) -> np.ndarray:
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return np.logical_not(equal)

    def __lt__(self, other: Any) -> np.ndarray:
        operands = self._operand_scalars(other)
        if operands is None:
            return NotImplemented
        return self._compare_scalars_against(operands, "<")

    def __le__(self, other: Any) -> np.ndarray:
        operands = self._operand_scalars(other)
        if operands is None:
            return NotImplemented
        return self._compare_scalars_against(operands, "<=")

    def __gt__(self, other: Any) -> np.ndarray:
        operands = self._operand_scalars(other)
        if operands is None:
            return NotImplemented
        return self._compare_scalars_against(operands, ">")

    def __ge__(self, other: Any) -> np.ndarray:
        operands = self._operand_scalars(other)
        if operands is None:
            return NotImplemented
        return self._compare_scalars_against(operands, ">=")

    # -- ordering ----------------------------------------------------------
    def order_values(self) -> np.ndarray:
        """Return R's ``vec_proxy_order`` for this mixed vector.

        Every part is converted to the finest shared chronon. Discrete values
        are nudged by half a chronon before conversion so a span sorts at its
        midpoint, matching R. Cyclical parts then order by their position
        within the shared cycle rather than their absolute chronon count.
        """
        if not self.parts:
            return np.asarray([], dtype=np.float64)

        mode = _mixtime_mode(self.parts)
        assert mode is not None
        common = _common_chronon(self.parts)
        cycle = _common_cycle(self.parts) if mode == "cyclical" else None
        multiple = len(self.parts) > 1

        pieces: list[np.ndarray] = []
        for part in self.parts:
            values = np.asarray(part.values, dtype=np.float64)
            if multiple and part.discrete:
                values = values + 0.5
            if mode == "duration":
                converted = _convert_defined(
                    values,
                    part.chronon,
                    common,
                    part.calendar,
                    duration=True,
                    anchored=False,
                )
            else:
                converted = _convert_defined(
                    values, part.chronon, common, part.calendar
                )
            if mode == "cyclical":
                assert cycle is not None
                converted = _cyclical_positions(
                    converted, common, cycle, part.calendar, part.timezone
                )
            pieces.append(converted)
        return np.concatenate(pieces)

    def argsort(
        self,
        *,
        decreasing: bool = False,
        na_position: str = "last",
    ) -> np.ndarray:
        """Return positions that sort this vector chronologically.

        ``na_position`` is ``"first"``, ``"last"``, or ``"drop"``. The default
        keeps missing values last, matching R's ``order()``.
        """
        if na_position not in {"first", "last", "drop"}:
            raise ValueError(
                "na_position must be 'first', 'last', or 'drop'"
            )
        proxy = self.order_values()
        missing = np.isnan(proxy)
        present = np.flatnonzero(~missing)
        if decreasing:
            order = present[np.argsort(-proxy[present], kind="stable")]
        else:
            order = present[np.argsort(proxy[present], kind="stable")]
        na = np.flatnonzero(missing)
        if na_position == "first":
            return np.concatenate([na, order])
        if na_position == "last":
            return np.concatenate([order, na])
        return order

    def sort(
        self,
        *,
        decreasing: bool = False,
        na_position: str = "drop",
    ) -> "Mixtime":
        """Sort chronologically; missing values are dropped by default.

        R's ``sort()`` drops missing values unless ``na.last`` is supplied, so
        ``na_position="drop"`` is the default here.
        """
        return self.take(
            self.argsort(decreasing=decreasing, na_position=na_position)
        )

    def _extreme(self, which: str, *, skipna: bool = True) -> Any:
        from .scalar import TimeScalar

        mode = _mixtime_mode(self.parts)
        if mode is None:
            raise ValueError(f"{which}() arg is an empty sequence")
        common = _common_chronon(self.parts)
        cycle = _common_cycle(self.parts) if mode == "cyclical" else None
        calendar = self.parts[0].calendar
        timezone = common.timezone or self.parts[0].timezone
        discrete = all(part.discrete for part in self.parts)

        pieces: list[np.ndarray] = []
        for part in self.parts:
            if mode == "duration":
                converted = _convert_defined(
                    part.values,
                    part.chronon,
                    common,
                    part.calendar,
                    duration=True,
                    anchored=False,
                )
            else:
                converted = _convert_defined(
                    part.values, part.chronon, common, part.calendar
                )
            pieces.append(converted)
        values = np.concatenate(pieces)
        missing = np.isnan(values)

        def scalar(value: float | None) -> Any:
            return TimeScalar(
                value,
                common,
                kind=mode,
                discrete=discrete,
                calendar=calendar,
                cycle=cycle,
                timezone=timezone,
            )

        if missing.all() or (not skipna and missing.any()):
            return scalar(None)
        extreme = np.nanmin(values) if which == "min" else np.nanmax(values)
        return scalar(float(extreme))

    def min(self, *, skipna: bool = True) -> Any:
        """Return the earliest value as a :class:`~mixtime.scalar.TimeScalar`.

        All parts are converted to their finest shared chronon first, matching
        R's ``min()``. Missing values are skipped unless ``skipna=False``.
        """
        return self._extreme("min", skipna=skipna)

    def max(self, *, skipna: bool = True) -> Any:
        """Return the latest value as a :class:`~mixtime.scalar.TimeScalar`."""
        return self._extreme("max", skipna=skipna)

    def _n_extreme(
        self, n: int, *, largest: bool, keep: str
    ) -> "Mixtime":
        if keep not in {"first", "last", "all"}:
            raise ValueError("keep must be 'first', 'last', or 'all'")
        if n <= 0:
            return Mixtime([])
        proxy = self.order_values()
        present = np.flatnonzero(~np.isnan(proxy))
        if present.size == 0:
            return Mixtime([])
        values = -proxy[present] if largest else proxy[present]
        if keep == "last":
            order = present[::-1][np.argsort(values[::-1], kind="stable")]
        else:
            order = present[np.argsort(values, kind="stable")]
        if keep == "all" and order.size > n:
            boundary = proxy[order[n - 1]]
            order = order[proxy[order] == boundary]
        else:
            order = order[:n]
        return self.take(order)

    def nlargest(self, n: int = 5, keep: str = "first") -> "Mixtime":
        """Return the ``n`` latest values, chronologically."""
        return self._n_extreme(n, largest=True, keep=keep)

    def nsmallest(self, n: int = 5, keep: str = "first") -> "Mixtime":
        """Return the ``n`` earliest values, chronologically."""
        return self._n_extreme(n, largest=False, keep=keep)

    def _slice(self, start: int, stop: int) -> "Mixtime":
        if start < 0:
            start += len(self)
        if stop < 0:
            stop += len(self)
        out: list[TimeVector] = []
        offset = 0
        for part in self.parts:
            part_start = offset
            part_stop = offset + len(part)
            offset = part_stop
            lo = max(start, part_start)
            hi = min(stop, part_stop)
            if lo < hi:
                out.append(part[lo - part_start:hi - part_start])
        return Mixtime(out)

    def __iter__(self) -> Iterator[TimeVector]:
        for part in self.parts:
            yield from part

    def format(self, fmt: str | None = None) -> list[str]:
        out: list[str] = []
        for part in self.parts:
            out.extend(part.format(fmt))
        return out

    def __str__(self) -> str:
        return " ".join(self.format())

    def __repr__(self) -> str:
        body = ", ".join(self.format())
        return f"<Mixtime[{len(self)}]: {body}>"

    @property
    def values(self) -> np.ndarray:
        if not self.parts:
            return np.asarray([], dtype=np.float64)
        return np.concatenate([np.asarray(part.values) for part in self.parts])

    def astype(self, chronon: Unit | str, *, discrete: bool | None = None) -> "Mixtime":
        return Mixtime([part.astype(chronon, discrete=discrete) for part in self.parts])


# ---------------------------------------------------------------------------
# Constructors


def linear_time(
    data: Any,
    chronon: Unit | str | None = None,
    *,
    discrete: bool | None = None,
    calendar: Calendar = cal_gregorian,
    tz: str | None = None,
) -> LinearTime:
    """Create a linear time vector with a chosen chronon."""
    calendar = _normalize_calendar(calendar)
    if chronon is None:
        chronon = _infer_chronon(data, calendar)
    discrete = _infer_discrete(data, discrete)
    unit = _normalize_unit(chronon, calendar, tz=tz)
    values = _values_from_data(
        data, unit, discrete=discrete, calendar=calendar, tz=tz
    )
    return LinearTime(
        values,
        unit,
        discrete=discrete,
        calendar=calendar,
        timezone=tz,
    )


def cyclical_time(
    data: Any,
    chronon: Unit | str,
    cycle: Unit | str,
    *,
    discrete: bool | None = None,
    calendar: Calendar = cal_gregorian,
    tz: str | None = None,
) -> CyclicalTime:
    """Create a cyclical time vector (a position within a repeating cycle)."""
    calendar = _normalize_calendar(calendar)
    discrete = _infer_discrete(data, discrete)
    unit = _normalize_unit(chronon, calendar, tz=tz)
    cycle_unit = _normalize_unit(cycle, calendar, tz=tz)
    values = _values_from_data(
        data, unit, discrete=discrete, calendar=calendar, tz=tz
    )
    return CyclicalTime(
        values,
        unit,
        cycle_unit,
        discrete=discrete,
        calendar=calendar,
        timezone=tz,
    )


def duration(
    data: Any,
    chronon: Unit | str,
    *,
    discrete: bool | None = None,
    calendar: Calendar = cal_gregorian,
) -> Duration:
    """Create a duration vector in ``chronon`` units.

    Durations are continuous by default, matching R's double-valued helpers
    such as ``years(1)``. Pass ``discrete=True`` for whole-granule durations.
    """
    calendar = _normalize_calendar(calendar)
    unit = _normalize_unit(chronon, calendar)
    if isinstance(data, TimeVector) and not isinstance(data, Duration):
        raise ValueError(
            "duration() converts duration vectors; use mixtime() to convert "
            "time points"
        )
    if discrete is None:
        discrete = data.discrete if isinstance(data, Duration) else False
    if isinstance(data, Duration):
        values = calendar.convert(
            data.values,
            data.chronon,
            unit,
            discrete=False,
            duration=True,
            anchored=False,
        )
    else:
        values = _as_values(data)
    if discrete:
        values = _coerce_values(np.trunc(values), True)
    return Duration(values, unit, discrete=discrete, calendar=calendar)


def year(data: Any, *, discrete: bool | None = None, calendar: Calendar = cal_gregorian,
         tz: str | None = None) -> LinearTime:
    return linear_time(data, calendar.year, discrete=discrete, calendar=calendar, tz=tz)


def yearquarter(data: Any, *, discrete: bool | None = None,
                calendar: Calendar = cal_gregorian, tz: str | None = None) -> LinearTime:
    return linear_time(data, calendar.quarter, discrete=discrete, calendar=calendar, tz=tz)


def yearmonth(data: Any, *, discrete: bool | None = None,
              calendar: Calendar = cal_gregorian, tz: str | None = None) -> LinearTime:
    return linear_time(data, calendar.month, discrete=discrete, calendar=calendar, tz=tz)


def yearweek(data: Any, *, discrete: bool | None = None,
             calendar: Calendar = cal_gregorian, tz: str | None = None) -> LinearTime:
    return linear_time(data, _week_unit(calendar), discrete=discrete,
                       calendar=calendar, tz=tz)


def date(data: Any, *, discrete: bool | None = None,
         calendar: Calendar = cal_gregorian, tz: str | None = None) -> LinearTime:
    return linear_time(data, calendar.day, discrete=discrete, calendar=calendar, tz=tz)


def datetime(data: Any, *, discrete: bool | None = None,
             calendar: Calendar = cal_gregorian, tz: str | None = None) -> LinearTime:
    return linear_time(data, calendar.second, discrete=discrete, calendar=calendar, tz=tz)


def month_of_year(data: Any, *, discrete: bool | None = None,
                  calendar: Calendar = cal_gregorian, tz: str | None = None) -> CyclicalTime:
    return cyclical_time(data, calendar.month, calendar.year,
                         discrete=discrete, calendar=calendar, tz=tz)


def day_of_year(data: Any, *, discrete: bool | None = None,
                calendar: Calendar = cal_gregorian, tz: str | None = None) -> CyclicalTime:
    return cyclical_time(data, calendar.day, calendar.year,
                         discrete=discrete, calendar=calendar, tz=tz)


def day_of_month(data: Any, *, discrete: bool | None = None,
                 calendar: Calendar = cal_gregorian, tz: str | None = None) -> CyclicalTime:
    return cyclical_time(data, calendar.day, calendar.month,
                         discrete=discrete, calendar=calendar, tz=tz)


def day_of_week(data: Any, *, discrete: bool | None = None,
                calendar: Calendar = cal_gregorian, tz: str | None = None) -> CyclicalTime:
    return cyclical_time(data, calendar.day, _week_unit(calendar),
                         discrete=discrete, calendar=calendar, tz=tz)


def week_of_year(data: Any, *, discrete: bool | None = None,
                 calendar: Calendar = cal_gregorian, tz: str | None = None) -> CyclicalTime:
    return cyclical_time(data, _week_unit(calendar), calendar.year,
                         discrete=discrete, calendar=calendar, tz=tz)


def time_of_day(data: Any, *, discrete: bool | None = None,
                calendar: Calendar = cal_gregorian, tz: str | None = None) -> CyclicalTime:
    return cyclical_time(data, calendar.second, calendar.day,
                         discrete=discrete, calendar=calendar, tz=tz)


def years(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, cal_gregorian.year, discrete=discrete)


def quarters(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, cal_gregorian.quarter, discrete=discrete)


def months(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, cal_gregorian.month, discrete=discrete)


def weeks(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, _week_unit(cal_gregorian), discrete=discrete)


def days(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, cal_gregorian.day, discrete=discrete)


def hours(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, cal_gregorian.hour, discrete=discrete)


def minutes(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, cal_gregorian.minute, discrete=discrete)


def seconds(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, cal_gregorian.second, discrete=discrete)


def milliseconds(n: Any, *, discrete: bool = False) -> Duration:
    return duration(n, cal_gregorian.millisecond, discrete=discrete)


def mixtime(*parts: Any) -> Mixtime:
    """Combine time vectors of different granularities into one container."""
    flattened: list[TimeVector] = []
    for part in parts:
        if isinstance(part, Mixtime):
            flattened.extend(part.parts)
        else:
            flattened.append(_as_time_vector(part))
    return Mixtime(flattened)


# ---------------------------------------------------------------------------
# Helpers


_UNIT_RANK = {
    "year": 0,
    "quarter": 1,
    "month": 2,
    "week": 3,
    "day": 4,
    "hour": 5,
    "minute": 6,
    "second": 7,
    "millisecond": 8,
    "microsecond": 9,
    "nanosecond": 10,
}


def _convert_defined(
    values: Any,
    from_unit: Unit,
    to_unit: Unit,
    calendar: Calendar,
    *,
    discrete: bool = False,
    duration: bool = False,
    anchored: bool = True,
) -> np.ndarray:
    """Convert values, leaving missing entries as ``NaN``.

    The Rust calendar kernel indexes month/year tables directly, so ``NaN``
    must never reach it.  Comparisons and ordering routinely contain missing
    values, hence this wrapper.
    """
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    defined = np.isfinite(values)
    if np.any(defined):
        result[defined] = calendar.convert(
            values[defined],
            from_unit,
            to_unit,
            discrete=discrete,
            duration=duration,
            anchored=anchored,
        )
    return result


def _linear_interval(
    values: np.ndarray,
    chronon: Unit,
    common: Unit,
    calendar: Calendar,
    discrete: bool,
) -> tuple[np.ndarray, np.ndarray]:
    start = _convert_defined(values, chronon, common, calendar, discrete=False)
    if not discrete:
        return start, start
    end = _convert_defined(
        np.asarray(values, dtype=np.float64) + 1.0,
        chronon,
        common,
        calendar,
        discrete=False,
    )
    return start, end


def _cyclical_positions(
    values: Any,
    chronon: Unit,
    cycle: Unit,
    calendar: Calendar,
    timezone: str | None,
) -> np.ndarray:
    """Return each value's 0-based position within ``cycle``.

    Missing values stay ``NaN``; :func:`_cycle_offset` is only called for
    defined entries because the calendar kernel cannot handle ``NaN``.
    """
    values = np.asarray(values, dtype=np.float64)
    positions = np.full(values.shape, np.nan, dtype=np.float64)
    defined = np.isfinite(values)
    if np.any(defined):
        positions[defined] = [
            _cycle_offset(value, chronon, cycle, calendar, timezone)
            for value in values[defined]
        ]
    return positions


def _cyclical_interval(
    values: Any,
    chronon: Unit,
    common: Unit,
    cycle: Unit,
    calendar: Calendar,
    discrete: bool,
    timezone: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    converted = _convert_defined(
        values, chronon, common, calendar, discrete=False
    )
    start = _cyclical_positions(converted, common, cycle, calendar, timezone)
    if not discrete:
        return start, start
    following = _convert_defined(
        values + 1.0, chronon, common, calendar, discrete=False
    )
    return start, start + (following - converted)


def _cycle_key(cycle: Unit | None) -> tuple[Any, ...] | None:
    if cycle is None:
        return None
    return (
        cycle.name,
        float(cycle.n),
        cycle.timezone,
        cycle.calendar.name if cycle.calendar is not None else None,
    )


def _common_cycle(parts: Iterable[Any]) -> Unit | None:
    """Return the shared cycle of cyclical parts, or ``None``.

    A cycle is a modulus rather than a unit, so differing cycles cannot be
    reconciled: R raises instead of picking a finer common cycle.
    """
    cycle: Unit | None = None
    for part in parts:
        if getattr(part, "kind", None) != "cyclical":
            continue
        current = getattr(part, "cycle", None)
        if current is None:
            raise ValueError("cyclical time is missing its cycle")
        if cycle is None:
            cycle = current
        elif _cycle_key(current) != _cycle_key(cycle):
            raise ValueError(
                "Can't compare cyclical time of differing cycles."
            )
    return cycle


def _comparison_unit(left: Unit, right: Unit) -> Unit:
    """Return R's ``chronon_glb`` for two comparison operands."""
    from .granules import chronon_glb

    return chronon_glb(left, right)


def _relation(
    op: str,
    x_start: Any,
    x_end: Any,
    y_start: Any,
    y_end: Any,
) -> Any:
    """Apply R's half-open interval comparison operators."""
    if op == "==":
        return (x_start == y_start) & (x_end == y_end)
    if op == "!=":
        return ~((x_start == y_start) & (x_end == y_end))
    if op == "<":
        return x_end <= y_start
    if op == ">":
        return x_start >= y_end
    if op == "<=":
        return x_end <= y_end
    if op == ">=":
        return x_start >= y_start
    raise ValueError(f"unknown comparison operator {op!r}")


def _duration_relation(op: str, left: Any, right: Any) -> Any:
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == ">":
        return left > right
    if op == "<=":
        return left <= right
    if op == ">=":
        return left >= right
    raise ValueError(f"unknown comparison operator {op!r}")


def _scalar_interval(
    scalar: Any,
    common: Unit,
    cycle: Unit | None,
) -> tuple[float, float]:
    """Return the half-open interval bounds of one scalar in ``common``."""
    if scalar.kind == "cyclical":
        assert cycle is not None
        start, end = _cyclical_interval(
            [scalar.value],
            scalar.chronon,
            common,
            cycle,
            scalar.calendar,
            scalar.discrete,
            scalar.timezone,
        )
    else:
        start, end = _linear_interval(
            np.asarray([scalar.value], dtype=np.float64),
            scalar.chronon,
            common,
            scalar.calendar,
            scalar.discrete,
        )
    return float(start[0]), float(end[0])


def _compare_scalars(left: Any, right: Any, op: str) -> bool:
    """Compare two scalars with R's interval semantics.

    Missing values compare equal to other missing values for ``==`` and
    ``!=`` is its negation.  Every ordering relation involving a missing value
    is ``False``.  This keeps pandas masks boolean while R would propagate a
    three-valued ``NA``.
    """
    if op == "!=":
        return not _compare_scalars(left, right, "==")
    if left.isna() or right.isna():
        return op == "==" and left.isna() and right.isna()
    if left.kind != right.kind:
        raise TypeError(
            "Cannot compare a duration with a time point"
            if {left.kind, right.kind} == {"duration", "linear"}
            or {left.kind, right.kind} == {"duration", "cyclical"}
            else f"Cannot compare {left.kind} time with {right.kind} time"
        )

    if left.kind == "duration":
        common = _common_unit(left.chronon, right.chronon, duration=True)
        left_value = _convert_defined(
            [left.value],
            left.chronon,
            common,
            left.calendar,
            duration=True,
            anchored=False,
        )[0]
        right_value = _convert_defined(
            [right.value],
            right.chronon,
            common,
            right.calendar,
            duration=True,
            anchored=False,
        )[0]
        return bool(_duration_relation(op, left_value, right_value))

    common = _comparison_unit(left.chronon, right.chronon)
    cycle = _common_cycle([left, right]) if left.kind == "cyclical" else None
    x_start, x_end = _scalar_interval(left, common, cycle)
    y_start, y_end = _scalar_interval(right, common, cycle)
    return bool(_relation(op, x_start, x_end, y_start, y_end))


def _mixtime_mode(parts: Iterable[Any]) -> str | None:
    modes = {part.kind for part in parts}
    if len(modes) > 1:
        names = " and ".join(sorted(modes))
        raise TypeError(
            f"Can't compare {names} time in a single Mixtime vector. "
            "Comparing, sorting and de-duplicating a Mixtime requires every "
            "element to share one mode of time."
        )
    return modes.pop() if modes else None


def _common_chronon(parts: Iterable[Any]) -> Unit:
    from .granules import chronon_glb

    units = [part.chronon for part in parts]
    if not units:
        return cal_gregorian.day
    return chronon_glb(units)


def _common_unit(left: Unit, right: Unit, *, duration: bool) -> Unit:
    """Return the finest unit that can represent both operands.

    Durations have no calendar anchor, so they cannot cross the calendar/fixed
    boundary. Time points can, because converting a calendar chronon to days
    uses the start of that calendar period.
    """
    if duration:
        calendar_like = {"year", "quarter", "month"}
        if (left.name in calendar_like) != (right.name in calendar_like):
            raise ValueError(
                "Cannot combine durations between calendar and fixed units; "
                "their ratio depends on where in the calendar the duration falls"
            )
    if left.name == right.name:
        return Unit(left.name, 1, left.calendar or right.calendar, left.timezone)
    try:
        left_rank = _UNIT_RANK[left.name]
        right_rank = _UNIT_RANK[right.name]
    except KeyError as exc:
        raise ValueError(f"Unsupported unit: {exc.args[0]!r}") from None
    if not duration:
        calendar_like = {"year", "quarter", "month"}
        left_calendar = left.name in calendar_like
        right_calendar = right.name in calendar_like
        if left_calendar != right_calendar:
            fixed = right if left_calendar else left
            # A week does not nest in a calendar month/year, but a day nests in
            # both, so the common chronon is a day rather than a week.
            name = "day" if fixed.name == "week" else fixed.name
            return Unit(name, 1, left.calendar or right.calendar, fixed.timezone)
    finer = left if left_rank >= right_rank else right
    return Unit(finer.name, 1, left.calendar or right.calendar, finer.timezone)


_CALENDAR_LIKE = {"year", "quarter", "month"}
_FIXED_LIKE = {
    "day",
    "week",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
}


def _check_duration_shift(
    time_unit: Unit, duration_unit: Unit, discrete: bool
) -> None:
    """Reject cross-category duration shifts we cannot reproduce faithfully."""
    time_name = time_unit.name
    duration_name = duration_unit.name
    cross = (
        time_name in _CALENDAR_LIKE and duration_name in _FIXED_LIKE
    ) or (
        time_name in _FIXED_LIKE and duration_name in _CALENDAR_LIKE
    )
    if not cross:
        return
    if discrete:
        raise ValueError(
            f"Cannot shift {time_unit!r} time points by {duration_unit!r}; "
            "the result is temporally indeterminate"
        )
    raise NotImplementedError(
        "Continuous shifts between calendar units and fixed units are not "
        "implemented; use discrete calendar chronons or convert explicitly"
    )


def _check_discrete_shift(
    time_unit: Unit, duration_unit: Unit, converted: np.ndarray
) -> None:
    """Reject discrete shifts that R treats as temporally indeterminate."""
    time_name = time_unit.name
    duration_name = duration_unit.name
    calendar_like = {"year", "quarter", "month"}
    variable_to_fixed = {"day", "week"}
    if time_name in calendar_like and duration_name in variable_to_fixed:
        raise ValueError(
            f"Cannot shift {time_unit!r} time points by {duration_unit!r}; "
            "the result is temporally indeterminate"
        )
    if time_name in {"day", "week"} and duration_name in calendar_like:
        raise ValueError(
            f"Cannot shift {time_unit!r} time points by {duration_unit!r}; "
            "the result is temporally indeterminate"
        )
    if not np.all(np.isfinite(converted) & (converted == np.floor(converted))):
        raise ValueError(
            f"Cannot shift {time_unit!r} time points by {duration_unit!r}; "
            "the result is temporally indeterminate"
        )


def _normalize_calendar(calendar: Calendar | str) -> Calendar:
    if isinstance(calendar, Calendar):
        return calendar
    if calendar in {"gregorian", "cal_gregorian"}:
        return cal_gregorian
    raise ValueError(f"Unknown calendar: {calendar!r}")


def _normalize_unit(
    unit: Unit | str,
    calendar: Calendar,
    *,
    tz: str | None = None,
) -> Unit:
    if isinstance(unit, str):
        unit = calendar.unit(unit)
    if not isinstance(unit, Unit):
        raise TypeError(f"Expected a Unit or unit name, got {type(unit)!r}")
    if unit.calendar is None:
        unit = Unit(unit.name, unit.n, calendar, unit.timezone)
    if tz is not None and unit.timezone is None:
        unit = Unit(unit.name, unit.n, unit.calendar, tz)
    return unit


def _week_unit(calendar: Calendar) -> Unit:
    return Unit("week", 1, calendar)


def _infer_discrete(data: Any, discrete: bool | None) -> bool:
    if discrete is not None:
        return bool(discrete)
    if isinstance(data, TimeVector):
        return data.discrete
    if isinstance(data, Mixtime):
        if len(data.parts) != 1:
            raise ValueError("mixtime input has multiple chronons; choose a chronon")
        return data.parts[0].discrete
    if isinstance(data, (str, bytes)) or _is_string_sequence(data):
        return True
    if _is_temporal(data):
        return True
    try:
        kind = np.asarray(data).dtype.kind
    except (TypeError, ValueError):
        return True
    if kind in "iu":
        return True
    if kind == "f":
        return False
    return True


def _infer_chronon(data: Any, calendar: Calendar) -> Unit:
    if isinstance(data, TimeVector):
        return data.chronon
    if isinstance(data, Mixtime):
        if len(data.parts) != 1:
            raise ValueError("mixtime input has multiple chronons; choose a chronon")
        return data.parts[0].chronon
    if isinstance(data, _dt.datetime):
        return calendar.second
    if isinstance(data, _dt.date):
        return calendar.day
    if isinstance(data, np.datetime64):
        return calendar.second
    return calendar.day


def _values_from_data(
    data: Any,
    unit: Unit,
    *,
    discrete: bool,
    calendar: Calendar,
    tz: str | None,
) -> np.ndarray:
    if isinstance(data, TimeVector):
        if isinstance(data, Duration):
            raise ValueError(
                "Cannot convert a duration to a time point; durations have no "
                "calendar position"
            )
        values = calendar.convert(data.values, data.chronon, unit, discrete=discrete)
        return _coerce_values(values, discrete)
    if isinstance(data, Mixtime):
        if len(data.parts) != 1:
            raise ValueError("mixtime input has multiple chronons; choose a chronon")
        return _values_from_data(
            data.parts[0], unit, discrete=discrete, calendar=calendar, tz=tz
        )

    # pandas Series and one-dimensional indexes are accepted as input. Convert
    # them to a plain list so the existing string/temporal/numeric paths apply,
    # normalising pandas' missing sentinels to NaN along the way.
    pandas_values = _pandas_sequence(data)
    if pandas_values is not None:
        data = pandas_values

    if _is_missing(data):
        return _coerce_values([np.nan], discrete)

    # Strings and object arrays are parsed according to the chronon.
    if isinstance(data, (str, bytes)) or _is_string_sequence(data):
        return _parse_for_unit(data, unit, calendar=calendar, tz=tz, discrete=discrete)

    # Temporal scalars / arrays.
    if _is_temporal(data):
        values = _temporal_values(data, unit, calendar=calendar, tz=tz, discrete=discrete)
        return _coerce_values(values, discrete)

    # Numeric values are already chronon counts for linear/cyclical time.
    values = _as_values(data)
    return _coerce_values(values, discrete)


def _parse_for_unit(
    data: Any,
    unit: Unit,
    *,
    calendar: Calendar,
    tz: str | None,
    discrete: bool,
) -> np.ndarray:
    name = unit.name
    if name == "year":
        return _coerce_values(parse_year(data), True)
    if name == "quarter":
        return _coerce_values(parse_quarter(data), True)
    if name == "month":
        return _coerce_values(parse_month(data), True)
    if name == "day":
        return _coerce_values(parse_date(data, tz=tz), True)
    if name == "week":
        return _coerce_values(parse_week(data, tz=tz), True)
    if name in {"second", "millisecond", "microsecond", "nanosecond",
                "minute", "hour"}:
        seconds = parse_time(data, tz=tz)
        values = calendar.convert(seconds, calendar.second, unit, discrete=discrete)
        return _coerce_values(values, discrete)
    raise ValueError(f"String parsing is not implemented for {unit!r}")


def _temporal_values(
    data: Any,
    unit: Unit,
    *,
    calendar: Calendar,
    tz: str | None,
    discrete: bool,
) -> np.ndarray:
    if isinstance(data, (list, tuple, np.ndarray)):
        items = list(data)
    else:
        items = [data]

    if unit.name == "day":
        days = np.asarray(
            [_temporal_days(item, tz=tz) for item in items], dtype=np.float64
        )
        return _round_for_discrete(days / float(unit.n), discrete)
    if unit.name == "week":
        days = np.asarray(
            [_temporal_days(item, tz=tz) for item in items], dtype=np.float64
        )
        return _round_for_discrete((days + 3.0) / (7.0 * float(unit.n)), discrete)
    if unit.name in {"second", "millisecond", "microsecond", "nanosecond",
                     "minute", "hour"}:
        seconds = np.asarray(
            [datetime_to_seconds(item, tz=tz) for item in items], dtype=np.float64
        )
        return calendar.convert(seconds, calendar.second, unit, discrete=discrete)

    days = np.asarray(
        [_temporal_days(item, tz=tz) for item in items], dtype=np.float64
    )
    return calendar.convert(days, calendar.day, unit, discrete=discrete)


def _temporal_days(value: Any, *, tz: str | None) -> float:
    if _is_missing(value):
        return np.nan
    if (
        isinstance(value, _dt.datetime)
        or isinstance(value, np.datetime64)
        or type(value).__name__ == "Timestamp"
    ):
        return datetime_to_seconds(value, tz=tz) / 86_400.0
    return date_to_days(value, tz=tz)


def _round_for_discrete(values: np.ndarray, discrete: bool) -> np.ndarray:
    if discrete:
        return np.floor(values)
    return values


def _as_time_vector(value: Any) -> TimeVector:
    if isinstance(value, TimeVector):
        return value
    raise TypeError(f"Expected a TimeVector, got {type(value)!r}")


def _as_values(data: Any) -> np.ndarray:
    if isinstance(data, (list, tuple, np.ndarray)):
        arr = np.asarray(data, dtype=np.float64)
    else:
        arr = np.asarray([data], dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("time values must be one-dimensional")
    return arr


def _coerce_values(values: Any, discrete: bool) -> np.ndarray:
    arr = _as_values(values)
    if discrete and np.all(np.isfinite(arr)) and np.all(arr == np.floor(arr)):
        return arr.astype(np.int64)
    # Continuous chronons are doubles in R, even when the parsed value is
    # integral; keep the Python storage type consistent.
    return arr.astype(np.float64)


def _pandas_sequence(data: Any) -> list[Any] | None:
    """Convert a one-dimensional pandas Series/Index to a plain list.

    pandas is optional, so this deliberately checks the type without importing
    it. Missing sentinels such as ``pd.NA`` and ``pd.NaT`` become ``NaN`` so the
    ordinary missing-value path can handle them.
    """
    if not type(data).__module__.startswith("pandas"):
        return None
    if getattr(data, "ndim", None) != 1:
        return None
    try:
        items = list(data)
    except TypeError:
        return None
    return [np.nan if _is_missing(item) else item for item in items]


def _is_string_sequence(data: Any) -> bool:
    if isinstance(data, np.ndarray):
        return data.dtype.kind in {"U", "S", "O"}
    if isinstance(data, (list, tuple)) and data:
        return all(
            isinstance(item, (str, bytes)) or _is_missing(item) for item in data
        )
    return False


def _is_temporal(data: Any) -> bool:
    if isinstance(data, (_dt.date, _dt.datetime, np.datetime64)):
        return True
    if type(data).__module__.startswith("pandas") and type(data).__name__ in {
        "Timestamp",
        "Period",
    }:
        return True
    if isinstance(data, (list, tuple, np.ndarray)) and len(data) > 0:
        for item in data:
            if _is_missing(item):
                continue
            return isinstance(item, (_dt.date, _dt.datetime, np.datetime64)) or (
                type(item).__module__.startswith("pandas")
                and type(item).__name__ in {"Timestamp", "Period"}
            )
    return False


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    if isinstance(value, np.datetime64) and np.isnat(value):
        return True
    # pandas NA / NaT without importing pandas.
    return type(value).__name__ in {"NAType", "NaTType"}


def _is_time_scalar(value: Any) -> bool:
    from .scalar import TimeScalar

    return isinstance(value, TimeScalar)


def _is_integral(value: Any) -> bool:
    return bool(np.isfinite(value)) and float(value) == np.floor(float(value))


def _month_parts(value: Any) -> tuple[int, int]:
    index = int(np.floor(value))
    year = index // 12 + 1970
    month = index % 12 + 1
    return year, month


def _quarter_parts(value: Any) -> tuple[int, int]:
    index = int(np.floor(value))
    year = index // 4 + 1970
    quarter = index % 4 + 1
    return year, quarter


def _format_year_value(value: Any, *, continuous: bool) -> str:
    if _is_missing(value):
        return "NA"
    frac = float(value) - np.floor(float(value))
    if not continuous and frac == 0.0:
        return str(int(value) + 1970)
    base = (
        str(int(value) + 1970)
        if frac == 0.0
        else f"{float(value) + 1970:.15g}"
    )
    return f"{base} {frac * 100:.1f}%"


def _date_from_day(value: Any) -> _dt.date:
    return _EPOCH_DATE + _dt.timedelta(days=int(np.floor(value)))


def _datetime_from_seconds(value: Any, tz: str | None) -> _dt.datetime:
    zone = _dt.timezone.utc if _timezone_is_utc(tz) else ZoneInfo(tz)
    return _dt.datetime.fromtimestamp(float(value), tz=zone)


def _shift_count(values: np.ndarray, duration: Duration) -> int:
    time_count = len(values)
    duration_count = len(duration.values)
    if time_count == duration_count:
        return time_count
    if time_count == 1:
        return duration_count
    if duration_count == 1:
        return time_count
    raise ValueError(
        "duration length must be 1 or match the time vector length"
    )


def _duration_amounts(
    duration: Duration, count: int, sign: int
) -> np.ndarray:
    """Broadcast a duration's values and convert to its base unit."""
    values = np.asarray(duration.values, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(count, float(values[0]), dtype=float)
    elif values.size != count:
        raise ValueError(
            "duration length must be 1 or match the time vector length"
        )
    return values * float(duration.chronon.n) * float(sign)


def _calendar_month_amounts(
    duration: Duration, count: int, sign: int
) -> np.ndarray:
    unit = duration.chronon.name
    months = _duration_amounts(duration, count, sign) * float(
        _MONTHS_PER_UNIT[unit]
    )
    finite = np.isfinite(months)
    if np.any(finite & (months != np.floor(months))):
        raise NotImplementedError(
            "fractional month shifts are not implemented; the shift must be "
            "a whole number of months. Use a whole-month duration or a "
            "day/week duration."
        )
    return months


def _day_fraction(value: _dt.datetime) -> float:
    seconds = (
        value.hour * 3600
        + value.minute * 60
        + value.second
        + value.microsecond / 1_000_000.0
    )
    return seconds / 86_400.0


def _add_months_local(
    value: _dt.datetime, months: float
) -> _dt.datetime:
    """Add whole months, clamping the day to the target month."""
    total = value.year * 12 + (value.month - 1) + int(months)
    year, month_index = divmod(total, 12)
    month = month_index + 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _shift_date_values(
    values: np.ndarray, duration: Duration, sign: int, discrete: bool
) -> np.ndarray:
    count = _shift_count(values, duration)
    values = np.broadcast_to(values, count)
    unit = duration.chronon.name
    if unit in {"day", "week"}:
        amounts = _duration_amounts(duration, count, sign)
        day_factor = 7.0 if unit == "week" else 1.0
        months = None
    else:
        amounts = None
        day_factor = 1.0
        months = _calendar_month_amounts(duration, count, sign)

    out = np.empty(count, dtype=float)
    for index in range(count):
        value = values[index]
        if _is_missing(value):
            out[index] = np.nan
            continue
        base_day = int(np.floor(float(value)))
        fraction = float(value) - base_day
        local = _dt.datetime.combine(
            _EPOCH_DATE + _dt.timedelta(days=base_day), _dt.time()
        ) + _dt.timedelta(days=fraction)

        if months is None:
            amount = float(amounts[index])
            if not np.isfinite(amount):
                out[index] = np.nan
                continue
            local += _dt.timedelta(days=amount * day_factor)
        else:
            amount = float(months[index])
            if not np.isfinite(amount):
                out[index] = np.nan
                continue
            local = _add_months_local(local, amount)

        shifted = (local.date() - _EPOCH_DATE).days + _day_fraction(local)
        out[index] = np.floor(shifted) if discrete else shifted
    return out


def _local_naive_from_seconds(
    value: Any, tz: str | None
) -> _dt.datetime:
    return _datetime_from_seconds(value, tz).replace(tzinfo=None)


def _local_naive_to_seconds(
    value: _dt.datetime, tz: str | None
) -> float:
    """Resolve a naive local time to UTC seconds.

    With R's default ``clamp`` strategy, an ambiguous local time resolves to
    its first occurrence, while a local time skipped by a DST gap moves back by
    the size of the gap.
    """
    if _timezone_is_utc(tz):
        return value.replace(tzinfo=_dt.timezone.utc).timestamp()

    zone = ZoneInfo(tz)
    candidates: list[tuple[float, bool]] = []
    for fold in (0, 1):
        aware = value.replace(tzinfo=zone, fold=fold)
        seconds = aware.timestamp()
        round_trip = _dt.datetime.fromtimestamp(seconds, zone).replace(
            tzinfo=None
        )
        candidates.append((seconds, round_trip == value))

    valid = [seconds for seconds, ok in candidates if ok]
    if valid:
        return min(valid)
    return min(seconds for seconds, _ in candidates)


def _shift_datetime_values(
    values: np.ndarray,
    duration: Duration,
    sign: int,
    tz: str | None,
) -> np.ndarray:
    count = _shift_count(values, duration)
    values = np.broadcast_to(values, count)
    unit = duration.chronon.name
    if unit in {"day", "week"}:
        amounts = _duration_amounts(duration, count, sign)
        day_factor = 7.0 if unit == "week" else 1.0
        months = None
    else:
        amounts = None
        day_factor = 1.0
        months = _calendar_month_amounts(duration, count, sign)

    out = np.empty(count, dtype=float)
    for index in range(count):
        value = values[index]
        if _is_missing(value):
            out[index] = np.nan
            continue
        local = _local_naive_from_seconds(value, tz)

        if months is None:
            amount = float(amounts[index])
            if not np.isfinite(amount):
                out[index] = np.nan
                continue
            local += _dt.timedelta(days=amount * day_factor)
        else:
            amount = float(months[index])
            if not np.isfinite(amount):
                out[index] = np.nan
                continue
            local = _add_months_local(local, amount)

        out[index] = _local_naive_to_seconds(local, tz)
    return out


def _cycle_offset(
    value: Any,
    chronon: Unit,
    cycle: Unit,
    calendar: Calendar,
    tz: str | None,
) -> float:
    """Return the 0-based offset of ``value`` within ``cycle``."""
    c = chronon.name
    y = cycle.name
    if c == "month" and y == "year":
        return float(value) % 12.0
    if c == "quarter" and y == "year":
        return float(value) % 4.0
    if c == "week" and y == "year":
        # ISO week number within the ISO year, plus the fraction of the week.
        week_value = float(value)
        date_value = _date_from_day(week_value * 7.0)
        iso = date_value.isocalendar()
        return float(iso.week - 1) + (week_value - np.floor(week_value))
    if c == "day" and y == "year":
        days, mod = _rs_day_to_year(np.asarray([float(value)]))
        return float(mod[0])
    if c == "day" and y == "month":
        months, mod = _rs_day_to_month(np.asarray([float(value)]))
        return float(mod[0])
    if c == "day" and y == "week":
        day_fraction = float(value) - np.floor(float(value))
        return float(_date_from_day(value).weekday()) + day_fraction
    if c == "second" and y == "day":
        if tz not in (None, "", "UTC"):
            dt = _datetime_from_seconds(value, tz)
            return float(dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6)
        return float(value) % 86_400.0
    # Fallback: find the start of the enclosing cycle and subtract it. This
    # works for generic cycles such as hour-of-day or month-of-quarter, and
    # preserves a continuous value's fractional position.
    cycle_index = calendar.convert(
        np.asarray([float(value)]), chronon, cycle, discrete=True
    )[0]
    base = calendar.convert(
        np.asarray([float(cycle_index)]), cycle, chronon, discrete=True
    )[0]
    return float(value) - float(base)


def _rs_day_to_year(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mixtime_rs as rs

    return rs.day_to_year(values, 1.0, 1.0)


def _rs_day_to_month(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mixtime_rs as rs

    return rs.day_to_month(values, 1.0, 1.0)


def _format_duration_vector(
    values: np.ndarray, unit: Unit, discrete: bool
) -> list[str]:
    plural = unit.name + "s"
    scale = float(unit.n)
    if discrete:
        out: list[str] = []
        for value in values:
            if _is_missing(value):
                out.append("NA")
                continue
            amount = int(np.trunc(float(value) * scale))
            label = unit.name if amount == 1 else plural
            out.append(f"{amount} {label}")
        return out

    decimals = 1
    for value in values:
        if _is_missing(value):
            continue
        text = f"{abs(float(value) * scale):.7g}"
        if "e" in text or "E" in text:
            continue
        decimals = max(decimals, len(text.split(".", 1)[1]) if "." in text else 0)
    out = []
    for value in values:
        if _is_missing(value):
            out.append("NA")
            continue
        amount = float(value) * scale
        out.append(f"{amount:.{decimals}f} {plural}")
    return out


# -- format token handling --------------------------------------------------


def _format_tokens(value: Any, vector: TimeVector, fmt: str) -> str:
    components = _components_for(value, vector)
    out = fmt
    # R-style component tokens.
    out = re.sub(
        r"\{lin\((\w+)\)\}",
        lambda m: _component_text(components, m.group(1), cyclical=False),
        out,
    )
    out = re.sub(
        r"\{cyc\((\w+),\s*(\w+)(?:,[^}]*)?\)\}",
        lambda m: _component_text(components, m.group(1), cyclical=True),
        out,
    )
    # strftime-style tokens. Use a private sentinel so literal text survives.
    tokens = {
        "%Y": f"{components['year']:04d}",
        "%y": f"{components['year'] % 100:02d}",
        "%m": f"{components['month']:02d}",
        "%b": _MONTH_ABBR[components["month"] - 1],
        "%B": _MONTH_FULL[components["month"] - 1],
        "%d": f"{components['day']:02d}",
        "%q": str(components["quarter"]),
        "%H": f"{components['hour']:02d}",
        "%M": f"{components['minute']:02d}",
        "%S": f"{components['second']:02d}",
        "%f": f"{components['microsecond']:06d}",
        "%j": f"{components['day_of_year']:03d}",
        "%a": _WEEKDAY_ABBR[components["weekday"]],
        "%A": _WEEKDAY_FULL[components["weekday"]],
        "%%": "%",
    }
    # Replace in a single pass so ``%%Y`` is a literal ``%Y`` rather than a
    # year token that happens to be preceded by an escaped percent sign.
    out = re.sub(
        r"%(?:%|[YymbBdqHMSfjaA])",
        lambda match: tokens[match.group(0)],
        out,
    )
    return out


def _component_text(components: dict[str, int], name: str, *, cyclical: bool) -> str:
    if name == "year":
        return f"{components['year']:04d}"
    if name == "quarter":
        return str(components["quarter"])
    if name == "month":
        return _MONTH_ABBR[components["month"] - 1] if cyclical else f"{components['month']:02d}"
    if name == "day":
        return str(components["day"])
    return str(components.get(name, ""))


def _components_for(value: Any, vector: TimeVector) -> dict[str, int]:
    name = vector.chronon.name
    if name == "year":
        year = int(np.floor(value)) + 1970
        return _date_components(_dt.date(year, 1, 1))
    if name == "quarter":
        year, quarter = _quarter_parts(value)
        month = (quarter - 1) * 3 + 1
        return _date_components(_dt.date(year, month, 1)) | {"quarter": quarter}
    if name == "month":
        year, month = _month_parts(value)
        return _date_components(_dt.date(year, month, 1))
    if name == "week":
        return _date_components(_date_from_day(float(value) * 7.0))
    if name == "day":
        return _date_components(_date_from_day(value))
    if name in {"second", "millisecond", "microsecond", "nanosecond"}:
        dt = _datetime_from_seconds(value, vector.timezone)
        return _date_components(dt) | {
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second,
            "microsecond": dt.microsecond,
        }
    return _date_components(_EPOCH_DATE)


def _date_components(value: _dt.date | _dt.datetime) -> dict[str, int]:
    return {
        "year": value.year,
        "month": value.month,
        "day": value.day,
        "quarter": (value.month - 1) // 3 + 1,
        "day_of_year": value.timetuple().tm_yday,
        "weekday": value.weekday(),
        "hour": getattr(value, "hour", 0),
        "minute": getattr(value, "minute", 0),
        "second": getattr(value, "second", 0),
        "microsecond": getattr(value, "microsecond", 0),
    }
