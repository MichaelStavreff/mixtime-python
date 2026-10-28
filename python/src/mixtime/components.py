"""Component extraction and composition.

The Python API mirrors the intent of R mixtime's ``time_components()`` and
``time_compose()`` without copying its formula syntax. Components are described
with :func:`lin` and :func:`cyc`, and composition accepts those descriptions
paired with values::

    time_components(x, year=lin("year"), month=cyc("month", "year"))
    time_compose(lin("year", 1980), cyc("month", "year", 3))

``cyc()`` values are 1-indexed positions within the cycle, matching everyday
counting. ``time_components()`` columns, like R's, store absolute chronon counts;
formatting reduces them to the position within the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .predicates import _as_granule
from .time import (
    CyclicalTime,
    Duration,
    LinearTime,
    Mixtime,
    TimeVector,
    _cycle_offset,
)
from .units import Calendar, Unit, _nests_in, cal_gregorian

_UNSET = object()


@dataclass(frozen=True)
class ComponentSpec:
    """A ``lin()`` or ``cyc()`` component description."""

    kind: str
    granule: Unit | str | Duration
    cycle: Unit | str | Duration | None = None


@dataclass(frozen=True)
class ComponentValue:
    """A component description paired with values for ``time_compose()``."""

    spec: ComponentSpec
    value: Any


def lin(granule: Unit | str | Duration, value: Any = _UNSET) -> ComponentSpec | ComponentValue:
    """Describe a linear component, optionally paired with a value."""
    spec = ComponentSpec("linear", granule)
    return spec if value is _UNSET else ComponentValue(spec, value)


def cyc(
    granule: Unit | str | Duration,
    cycle: Unit | str | Duration,
    value: Any = _UNSET,
) -> ComponentSpec | ComponentValue:
    """Describe a cyclical component, optionally paired with a 1-indexed value."""
    spec = ComponentSpec("cyclical", granule, cycle)
    return spec if value is _UNSET else ComponentValue(spec, value)


def time_components(
    x: TimeVector | Mixtime, **components: ComponentSpec
) -> dict[str, TimeVector | Mixtime]:
    """Extract named linear and cyclical components from ``x``.

    For a ``Mixtime`` input each returned column is a ``Mixtime`` whose parts
    correspond to the input parts. For a single time vector each column is a
    ``LinearTime`` or ``CyclicalTime``.
    """
    if not components:
        raise ValueError("time_components() needs at least one component")
    for name, spec in components.items():
        if not isinstance(spec, ComponentSpec):
            raise TypeError(
                f"component {name!r} must be created with lin() or cyc()"
            )

    if isinstance(x, Mixtime):
        parts = list(x.parts)
        return {
            name: Mixtime([_component_one(part, spec) for part in parts])
            for name, spec in components.items()
        }
    if not isinstance(x, TimeVector) or isinstance(x, Duration):
        raise TypeError("time_components() expects a time vector or Mixtime")

    return {name: _component_one(x, spec) for name, spec in components.items()}


def time_compose(
    *components: ComponentValue | ComponentSpec | tuple[ComponentSpec, Any] | TimeVector,
    discrete: bool = True,
    calendar: Calendar = cal_gregorian,
) -> TimeVector:
    """Compose a time vector from linear and cyclical components.

    Each argument is a :class:`ComponentValue` (for example
    ``lin("year", 1980)``), a ``(spec, value)`` pair, or an already-tagged
    ``LinearTime``/``CyclicalTime``. Cyclical values are 1-indexed.
    """
    if not components:
        raise ValueError("time_compose() needs at least one component")

    normalized = [_compose_one(component, calendar) for component in components]
    arrays = [np.asarray(item["value"], dtype=float) for item in normalized]
    arrays = [value.reshape(1) if value.ndim == 0 else value for value in arrays]
    for value in arrays:
        if value.ndim != 1:
            raise ValueError("component values must be one-dimensional")
    arrays = np.broadcast_arrays(*arrays)
    for item, values in zip(normalized, arrays):
        item["value"] = values

    linear = [item for item in normalized if item["cycle"] is None]
    if len(linear) > 1:
        raise ValueError("time_compose() accepts at most one linear anchor")

    if linear:
        root_index = normalized.index(linear[0])
        root_cycle = None
    else:
        supplied = [item["chronon"] for item in normalized]
        roots = [
            index
            for index, item in enumerate(normalized)
            if not any(_same_unit(item["cycle"], unit) for unit in supplied)
        ]
        if len(roots) != 1:
            raise ValueError(
                "time_compose() needs a single connected chain of cyclical "
                "components, or one linear anchor"
            )
        root_index = roots[0]
        root_cycle = normalized[root_index]["cycle"]

    chain = [root_index]
    frontier = normalized[root_index]["chronon"]
    remaining = set(range(len(normalized))) - {root_index}
    while remaining:
        matches = [
            index
            for index in remaining
            if _same_unit(normalized[index]["cycle"], frontier)
        ]
        if not matches:
            raise ValueError(
                "time_compose() components do not form one connected chain"
            )
        if len(matches) > 1:
            raise ValueError(
                "time_compose() found multiple components for the same cycle"
            )
        index = matches[0]
        chain.append(index)
        remaining.remove(index)
        frontier = normalized[index]["chronon"]

    anchor = normalized[chain[0]]
    running_count = np.asarray(anchor["value"], dtype=float).copy()
    running_chronon = anchor["chronon"]
    keep = np.isfinite(running_count)

    for index in chain[1:]:
        link = normalized[index]
        chronon = link["chronon"]
        base = calendar.convert(
            running_count[keep], running_chronon, chronon, discrete=True
        )
        end = calendar.convert(
            running_count[keep] + 1.0, running_chronon, chronon, discrete=True
        )
        cardinality = np.asarray(end, dtype=float) - np.asarray(base, dtype=float)
        offset = np.asarray(link["value"], dtype=float)[keep]
        finite = np.isfinite(offset)
        bad = finite & ((offset < 0) | (offset >= cardinality))
        if np.any(bad):
            raise ValueError(
                f"Invalid position for {chronon.name} granules within "
                f"{running_chronon.name}"
            )

        new_count = running_count.copy()
        new_count[keep] = np.asarray(base) + offset
        next_keep = keep.copy()
        next_keep[keep] = finite
        running_count = new_count
        running_chronon = chronon
        keep = next_keep

    if discrete:
        rounded = np.where(np.isfinite(running_count), np.rint(running_count), np.nan)
        values: np.ndarray = (
            rounded.astype(np.int64)
            if np.all(np.isfinite(rounded))
            else rounded
        )
    else:
        values = running_count

    if root_cycle is None:
        return LinearTime(
            values, running_chronon, discrete=discrete, calendar=calendar
        )
    return CyclicalTime(
        values,
        running_chronon,
        root_cycle,
        discrete=discrete,
        calendar=calendar,
    )


def _component_one(x: TimeVector, spec: ComponentSpec) -> TimeVector:
    if isinstance(x, Duration):
        raise TypeError("durations have no calendar components")
    unit = _as_granule(spec.granule, x.calendar)
    cycle = (
        _as_granule(spec.cycle, x.calendar)
        if spec.kind == "cyclical" and spec.cycle is not None
        else None
    )
    if spec.kind not in {"linear", "cyclical"}:
        raise ValueError(f"Unknown component kind: {spec.kind!r}")
    if spec.kind == "cyclical" and cycle is None:
        raise ValueError("cyc() needs a cycle")

    values = np.asarray(x.values, dtype=float)
    special = ~np.isfinite(values)
    keep = ~special
    determinable = (not x.discrete) or _nests_in(x.chronon, unit)

    out = np.full(len(x), np.nan, dtype=float)
    if determinable and np.any(keep):
        if _same_unit(x.chronon, unit):
            converted = values[keep]
        else:
            # R's component decomposition floors when it changes granules,
            # even for continuous input; a same-unit component keeps its
            # fractional position.
            converted = x.calendar.convert(
                values[keep], x.chronon, unit, discrete=True
            )
        if spec.kind == "linear":
            out = values.copy()
        out[keep] = converted

    if spec.kind == "linear":
        return LinearTime(
            out,
            unit,
            discrete=x.discrete,
            calendar=x.calendar,
            timezone=x.timezone,
        )
    assert cycle is not None
    return CyclicalTime(
        out,
        unit,
        cycle,
        discrete=x.discrete,
        calendar=x.calendar,
        timezone=x.timezone,
    )


def _compose_one(component: Any, calendar) -> dict[str, Any]:
    if isinstance(component, ComponentValue):
        spec = component.spec
        value = component.value
        if isinstance(value, TimeVector):
            value = value.values
        chronon = _as_granule(spec.granule, calendar)
        if spec.kind == "linear":
            return {
                "chronon": chronon,
                "cycle": None,
                "value": np.asarray(value, dtype=float) - _chronon_epoch(chronon),
            }
        if spec.kind == "cyclical":
            if spec.cycle is None:
                raise ValueError("cyc() needs a cycle")
            cycle = _as_granule(spec.cycle, calendar)
            return {
                "chronon": chronon,
                "cycle": cycle,
                "value": np.asarray(value, dtype=float) - 1.0,
            }
        raise ValueError(f"Unknown component kind: {spec.kind!r}")

    if isinstance(component, ComponentSpec):
        raise TypeError("time_compose() components need values; use lin(..., value)")

    if isinstance(component, tuple) and len(component) == 2:
        spec, value = component
        if not isinstance(spec, ComponentSpec):
            raise TypeError("expected a (lin()/cyc() spec, value) pair")
        return _compose_one(ComponentValue(spec, value), calendar)

    if isinstance(component, Mixtime):
        if len(component.parts) != 1:
            raise ValueError(
                "each time_compose() component must have one consistent chronon"
            )
        component = component.parts[0]

    if isinstance(component, Duration):
        raise TypeError("durations are not time_compose() components")

    if isinstance(component, (LinearTime, CyclicalTime)):
        values = np.asarray(component.values, dtype=float)
        if isinstance(component, LinearTime):
            return {
                "chronon": component.chronon,
                "cycle": None,
                "value": values,
            }
        assert component.cycle is not None
        offsets = np.full(len(values), np.nan, dtype=float)
        keep = np.isfinite(values)
        offsets[keep] = [
            _cycle_offset(
                value,
                component.chronon,
                component.cycle,
                component.calendar,
                component.timezone,
            )
            for value in values[keep]
        ]
        return {
            "chronon": component.chronon,
            "cycle": component.cycle,
            "value": offsets,
        }

    raise TypeError(
        "time_compose() components must be lin()/cyc() values, (spec, value) "
        "pairs, or linear/cyclical time vectors"
    )


def _same_unit(left: Unit | None, right: Unit | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.name == right.name and float(left.n) == float(right.n)


def _chronon_epoch(unit: Unit) -> float:
    # R's Gregorian year chronon displays real-world years; every other
    # chronon's linear count starts at zero.
    return 1970.0 if unit.name == "year" else 0.0
