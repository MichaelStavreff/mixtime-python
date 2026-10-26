"""Python wrapper around the Rust port of mixtime's Gregorian chronon arithmetic.

The functions mirror `chronon_divmod()` from mixtime 0.4.0 (`R/cal_gregorian.R`).
Values are chronon counts relative to the Gregorian epoch, 1970-01-01.

Every function takes a one-dimensional ``float64`` NumPy array and returns two
one-dimensional ``float64`` NumPy arrays. Both directions are zero-copy: the
input buffer is read in place, and the Rust kernels fill freshly allocated
NumPy output buffers directly. A list or any other dtype is rejected by the
extension module.
"""

from __future__ import annotations

import numpy as np

import mixtime_rs as _rs

__all__ = [
    "day_to_month",
    "day_to_quarter",
    "day_to_year",
    "month_to_day",
    "year_to_day",
]


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
