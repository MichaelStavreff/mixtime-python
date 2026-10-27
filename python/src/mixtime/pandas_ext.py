"""Optional pandas extension dtype for mixtime vectors.

The core package does not import pandas.  Import this module (or use
``TimeVector.to_pandas_series`` / ``Mixtime.to_pandas_series``) only when pandas
is available.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .scalar import (
    TimeScalar,
    _is_missing_value,
    _mixtime_from_scalars,
    _scalar_from_item,
)
from .time import Mixtime, TimeVector

try:  # pragma: no cover - the import path is environment-dependent
    import pandas as pd
    from pandas.api.extensions import (
        ExtensionArray,
        ExtensionDtype,
        register_extension_dtype,
    )

    try:
        from pandas._libs.lib import no_default as _no_default
    except ImportError:  # pragma: no cover - older pandas
        _no_default = object()
except ImportError as exc:  # pragma: no cover - depends on environment
    pd = None
    _PANDAS_IMPORT_ERROR = exc
    ExtensionArray = object  # type: ignore[assignment,misc]
    ExtensionDtype = object  # type: ignore[assignment,misc]
    _no_default = object()

    def register_extension_dtype(cls):  # type: ignore[misc]
        return cls

else:
    _PANDAS_IMPORT_ERROR = None


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover - depends on environment
        raise ImportError(
            "pandas is required for mixtime's pandas extension dtype; "
            "install mixtime[pandas]"
        ) from _PANDAS_IMPORT_ERROR


def _coerce_fill_scalar(value: Any) -> TimeScalar:
    if isinstance(value, TimeScalar):
        return value
    if _is_missing_value(value):
        return TimeScalar.missing()
    try:
        return _scalar_from_item(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "fill value must be a TimeScalar or a missing value"
        ) from exc


def _fill_values(value: Any, length: int) -> list[TimeScalar] | None:
    if isinstance(value, MixtimeArray):
        values = [value[i] for i in range(len(value))]
    elif isinstance(value, (TimeVector, Mixtime)):
        values = [value.scalar(i) for i in range(len(value))]
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = [_coerce_fill_scalar(item) for item in value]
    else:
        return None
    if len(values) != length:
        raise ValueError("fill value has the wrong length")
    return values


@register_extension_dtype
class MixtimeDtype(ExtensionDtype):
    """pandas dtype for a column of :class:`~mixtime.scalar.TimeScalar`."""

    name = "mixtime"
    type = TimeScalar
    kind = "O"
    na_value = TimeScalar.missing()

    @classmethod
    def construct_from_string(cls, string: str) -> "MixtimeDtype":
        if string == cls.name:
            return cls()
        raise TypeError(f"cannot construct a {cls.__name__} from {string!r}")

    @classmethod
    def construct_array_type(cls) -> type["MixtimeArray"]:
        return MixtimeArray

    def __repr__(self) -> str:
        return "MixtimeDtype()"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MixtimeDtype)

    def __hash__(self) -> int:
        return hash(self.name)


class MixtimeArray(ExtensionArray):
    """A pandas ExtensionArray backed by a :class:`~mixtime.time.Mixtime`."""

    def __init__(self, data: Any, copy: bool = False) -> None:
        _require_pandas()
        if isinstance(data, MixtimeArray):
            self._data = (
                Mixtime(
                    [
                        part._new(np.array(part.values, copy=True))
                        for part in data._data.parts
                    ]
                )
                if copy
                else data._data
            )
        elif isinstance(data, Mixtime):
            self._data = data
        elif isinstance(data, TimeVector):
            self._data = Mixtime([data])
        else:
            self._data = _mixtime_from_scalars(list(data))

    # -- pandas ExtensionArray protocol -----------------------------------
    @property
    def dtype(self) -> MixtimeDtype:
        return MixtimeDtype()

    @property
    def nbytes(self) -> int:
        return int(sum(np.asarray(part.values).nbytes for part in self._data.parts))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            return MixtimeArray(self._data[item])
        if isinstance(item, (list, tuple, np.ndarray)):
            positions = np.asarray(item)
            if positions.dtype == bool:
                positions = np.flatnonzero(positions)
            return MixtimeArray(self._data.take(positions))
        return self._data.scalar(int(item))

    def __setitem__(self, key: Any, value: Any) -> None:
        scalars = [self._data.scalar(i) for i in range(len(self))]
        if isinstance(key, (int, np.integer)):
            indices = [int(key)]
        elif isinstance(key, slice):
            indices = list(range(len(self)))[key]
        else:
            positions = np.asarray(key)
            if positions.dtype == bool:
                indices = np.flatnonzero(positions).tolist()
            else:
                indices = positions.astype(np.int64).ravel().tolist()

        replacements = _fill_values(value, len(indices))
        if replacements is None:
            fill = _coerce_fill_scalar(value)
            replacements = [fill] * len(indices)
        for index, replacement in zip(indices, replacements):
            scalars[index] = replacement
        self._data = _mixtime_from_scalars(scalars)

    def isna(self) -> np.ndarray:
        return self._data.isna()

    def fillna(
        self,
        value: Any,
        limit: int | None = None,
        copy: bool = True,
    ) -> "MixtimeArray":
        mask = self.isna()
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        replacements = _fill_values(value, len(self))
        fill = None if replacements is not None else _coerce_fill_scalar(value)

        scalars: list[TimeScalar] = []
        filled = 0
        for index, scalar in enumerate(self):
            should_fill = bool(mask[index]) and (
                limit is None or filled < limit
            )
            if should_fill:
                scalars.append(
                    replacements[index] if replacements is not None else fill
                )
                filled += 1
            else:
                scalars.append(scalar)
        return MixtimeArray(_mixtime_from_scalars(scalars))

    def take(
        self,
        indices: Any,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> "MixtimeArray":
        positions = np.asarray(indices, dtype=np.int64).ravel()
        if not allow_fill:
            return MixtimeArray(self._data.take(positions))

        if fill_value is None or fill_value is _no_default:
            fill = TimeScalar.missing()
        elif isinstance(fill_value, TimeScalar):
            fill = fill_value
        elif _is_missing_value(fill_value):
            fill = TimeScalar.missing()
        else:
            raise TypeError(
                "fill_value must be a TimeScalar or a missing value"
            )
        return MixtimeArray(
            self._data.take(positions, allow_fill=True, fill_value=fill)
        )

    def copy(self) -> "MixtimeArray":
        return MixtimeArray(self, copy=True)

    @classmethod
    def _from_sequence(
        cls,
        scalars: Any,
        *,
        dtype: Any = None,
        copy: bool = False,
    ) -> "MixtimeArray":
        if dtype is not None and not isinstance(dtype, MixtimeDtype):
            if dtype != MixtimeDtype.name:
                raise TypeError(f"cannot create a MixtimeArray with dtype {dtype!r}")
        return cls(scalars, copy=copy)

    @classmethod
    def _from_factorized(
        cls, values: Any, original: "MixtimeArray"
    ) -> "MixtimeArray":
        return cls(values)

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence["MixtimeArray"]
    ) -> "MixtimeArray":
        parts: list[TimeVector] = []
        for array in to_concat:
            parts.extend(array._data.parts)
        return cls(Mixtime(parts))

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        values = np.asarray(
            [self._data.scalar(i) for i in range(len(self))], dtype=object
        )
        if copy:
            values = values.copy()
        if dtype is None:
            return values
        return values.astype(dtype, copy=False)

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        if isinstance(dtype, MixtimeDtype) or dtype == MixtimeDtype.name:
            return self.copy() if copy else self
        if dtype is object or np.dtype(dtype) == np.dtype(object):
            return np.asarray(self, dtype=object)
        return np.asarray(self, dtype=dtype)

    def to_numpy(
        self,
        dtype: Any = None,
        copy: bool = False,
        na_value: Any = _no_default,
    ) -> np.ndarray:
        if dtype is None or np.dtype(dtype) == np.dtype(object):
            values = np.asarray(self, dtype=object)
            if na_value is not _no_default:
                values = np.asarray(
                    [
                        na_value if _is_missing_value(value) else value
                        for value in values
                    ],
                    dtype=object,
                )
            return values.copy() if copy else values
        return np.asarray(self, dtype=dtype).copy() if copy else np.asarray(
            self, dtype=dtype
        )

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        return np.asarray(self, dtype=object), self.dtype.na_value

    def _values_for_argsort(self) -> np.ndarray:
        scalars = [self._data.scalar(i) for i in range(len(self))]
        nonmissing = [scalar for scalar in scalars if not scalar.isna()]
        if not nonmissing:
            return np.arange(len(self), dtype=np.int64)
        metadata = nonmissing[0]._meta_key()
        if all(scalar._meta_key() == metadata for scalar in nonmissing):
            return np.asarray(
                [np.nan if scalar.isna() else scalar.value for scalar in scalars],
                dtype=float,
            )
        # Mixed granularities have no single natural numeric order.  Keep the
        # existing order rather than inventing one that disagrees with the
        # vector's part-based ordering.
        return np.arange(len(self), dtype=np.int64)

    def _formatter(self, boxed: bool = False):
        def format_value(value: Any) -> str:
            if isinstance(value, TimeScalar):
                return value.format()
            return str(value)

        return format_value

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, MixtimeArray):
            if len(self) != len(other):
                return NotImplemented
            other_values = [other._data.scalar(i) for i in range(len(other))]
        elif isinstance(other, TimeScalar):
            other_values = [other] * len(self)
        elif isinstance(other, (TimeVector, Mixtime)):
            if len(self) != len(other):
                return NotImplemented
            other_values = [other.scalar(i) for i in range(len(other))]
        else:
            return NotImplemented
        return np.asarray(
            [
                left == right
                for left, right in zip(
                    (self._data.scalar(i) for i in range(len(self))),
                    other_values,
                )
            ],
            dtype=bool,
        )

    def __ne__(self, other: Any) -> Any:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return np.logical_not(result)


if pd is None:  # pragma: no cover - depends on environment
    # Keep the names importable so callers get a useful error at construction
    # time rather than a confusing import failure.
    class MixtimeDtype:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_pandas()

    class MixtimeArray:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_pandas()
