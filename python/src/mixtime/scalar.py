"""Scalar values for :class:`~mixtime.time.TimeVector` and ``Mixtime``.

Pandas extension arrays need real scalar values: ``series.iloc[i]`` must return
one time point, not a length-one vector.  ``TimeScalar`` is that representation.
It deliberately keeps the same chronon metadata as the vector it came from so
round-tripping through a pandas column does not silently coerce granularity.

The core API continues to expose ``x[i]`` as a length-one vector.  Use
``x.scalar(i)`` when an actual scalar is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from .units import Calendar, Unit, cal_gregorian


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    # pandas.NA / pandas.NaT without importing pandas.
    return type(value).__name__ in {"NAType", "NaTType"}


def _unit_key(unit: Unit | None) -> tuple[str, float, str | None] | None:
    if unit is None:
        return None
    return (unit.name, float(unit.n), unit.timezone)


@dataclass(frozen=True)
class TimeScalar:
    """One value with its chronon metadata.

    ``value`` is ``None`` for missing values.  ``kind`` is one of ``"linear"``,
    ``"cyclical"``, or ``"duration"``.
    """

    value: float | None
    chronon: Unit
    kind: str = "linear"
    discrete: bool = True
    calendar: Calendar = cal_gregorian
    cycle: Unit | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        if self.value is not None and _is_missing_value(self.value):
            object.__setattr__(self, "value", None)
        elif self.value is not None:
            object.__setattr__(self, "value", float(self.value))
        if self.timezone is None and self.chronon.timezone is not None:
            object.__setattr__(self, "timezone", self.chronon.timezone)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_vector(cls, vector: Any, value: Any) -> "TimeScalar":
        """Build a scalar from a vector's metadata and one stored value."""
        missing = _is_missing_value(value)
        return cls(
            None if missing else float(value),
            vector.chronon,
            kind=vector.kind,
            discrete=vector.discrete,
            calendar=vector.calendar,
            cycle=vector.cycle,
            timezone=vector.timezone,
        )

    @classmethod
    def missing(cls, like: Any | None = None) -> "TimeScalar":
        """Return a missing scalar, optionally borrowing ``like``'s metadata."""
        if like is None:
            return cls(None, cal_gregorian.day)
        if isinstance(like, TimeScalar):
            return replace(like, value=None)
        parts = getattr(like, "parts", None)
        if parts is not None:
            if not parts:
                return cls(None, cal_gregorian.day)
            like = parts[0]
        return cls(
            None,
            like.chronon,
            kind=getattr(like, "kind", "linear"),
            discrete=getattr(like, "discrete", True),
            calendar=getattr(like, "calendar", cal_gregorian),
            cycle=getattr(like, "cycle", None),
            timezone=getattr(like, "timezone", None),
        )

    # -- properties --------------------------------------------------------
    def isna(self) -> bool:
        return self.value is None

    def notna(self) -> bool:
        return self.value is not None

    def _raw_value(self) -> float:
        return np.nan if self.value is None else float(self.value)

    def _meta_key(self) -> tuple[Any, ...]:
        return (
            self.kind,
            _unit_key(self.chronon),
            self.calendar.name,
            _unit_key(self.cycle),
            self.timezone,
            self.discrete,
        )

    # -- conversion --------------------------------------------------------
    def to_vector(self, values: Iterable[Any] | None = None) -> Any:
        """Return a length-one vector, or a vector using ``values``."""
        from .time import CyclicalTime, Duration, LinearTime

        if values is None:
            values = [self._raw_value()]
        else:
            values = list(values)
        kwargs = {
            "discrete": self.discrete,
            "calendar": self.calendar,
            "timezone": self.timezone,
        }
        if self.kind == "duration":
            return Duration(values, self.chronon, **kwargs)
        if self.kind == "cyclical":
            return CyclicalTime(
                values,
                self.chronon,
                self.cycle or self.chronon,
                **kwargs,
            )
        return LinearTime(values, self.chronon, cycle=self.cycle, **kwargs)

    def format(self, fmt: str | None = None) -> str:
        return self.to_vector().format(fmt)[0]

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"<TimeScalar {self.format()!r}>"

    # -- equality and ordering --------------------------------------------
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TimeScalar):
            return NotImplemented
        if self.isna() or other.isna():
            return self.isna() and other.isna()
        return (
            self.value == other.value and self._meta_key() == other._meta_key()
        )

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self) -> int:
        if self.isna():
            return hash(("TimeScalar", None))
        return hash(("TimeScalar", self.value, self._meta_key()))

    def _compare(self, other: Any, op: str) -> bool:
        if not isinstance(other, TimeScalar):
            return NotImplemented
        from .time import _compare_scalars

        return _compare_scalars(self, other, op)

    def __lt__(self, other: Any) -> bool:
        return self._compare(other, "<")

    def __le__(self, other: Any) -> bool:
        return self._compare(other, "<=")

    def __gt__(self, other: Any) -> bool:
        return self._compare(other, ">")

    def __ge__(self, other: Any) -> bool:
        return self._compare(other, ">=")


def _scalar_from_item(item: Any) -> TimeScalar:
    if isinstance(item, TimeScalar):
        return item
    if _is_missing_value(item):
        return TimeScalar.missing()
    # Accept length-one vectors/containers so pandas construction from a list
    # of ``x[i]`` values keeps working.
    if hasattr(item, "parts") and hasattr(item, "scalar"):
        if len(item) != 1:
            raise ValueError("a Mixtime item must have length 1")
        return item.scalar(0)
    if hasattr(item, "chronon") and hasattr(item, "values"):
        if len(item) != 1:
            raise ValueError("a TimeVector item must have length 1")
        return item.scalar(0)
    raise TypeError(f"expected TimeScalar or missing value, got {type(item)!r}")


def _is_generic_missing_scalar(value: Any) -> bool:
    """Return whether ``value`` is a metadata-free missing scalar."""
    return (
        isinstance(value, TimeScalar)
        and value.isna()
        and value._meta_key() == TimeScalar.missing()._meta_key()
    )


def _mixtime_from_scalars(
    scalars: Iterable[Any], *, inherit_missing: bool = True
) -> Any:
    """Group scalars into contiguous same-granularity parts."""
    from .time import Mixtime

    parts: list[Any] = []
    template: TimeScalar | None = None
    values: list[float] = []

    for item in scalars:
        scalar = _scalar_from_item(item)
        if scalar.isna() and template is not None and inherit_missing:
            # A generic missing scalar should inherit the surrounding part's
            # granularity rather than creating a separate one-element part.
            scalar = TimeScalar.missing(template)
        if template is None or scalar._meta_key() != template._meta_key():
            if template is not None:
                parts.append(template.to_vector(values))
            template = scalar
            values = [scalar._raw_value()]
        else:
            values.append(scalar._raw_value())

    if template is not None:
        parts.append(template.to_vector(values))
    return Mixtime(parts)
