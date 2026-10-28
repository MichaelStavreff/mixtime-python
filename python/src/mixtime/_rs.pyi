"""Type stubs for the Rust chronon kernel (``mixtime._rs``).

The kernels are NumPy-in/NumPy-out and read the input buffer in place. Use the
documented wrappers in :mod:`mixtime` rather than importing this module
directly.
"""

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]

def day_to_month(x: _F64, from_n: float = 1.0, to_n: float = 1.0) -> tuple[_F64, _F64]:
    """Split day chronons into the containing month.

    Returns ``(div, mod)``: the containing month block and the day offset from
    the start of that block.
    """
    ...

def day_to_year(x: _F64, from_n: float = 1.0, to_n: float = 1.0) -> tuple[_F64, _F64]:
    """Split day chronons into the containing year.

    Returns ``(div, mod)``: the containing year and the day offset from its
    start.
    """
    ...

def month_to_day(x: _F64, from_n: float = 1.0, to_n: float = 1.0) -> tuple[_F64, _F64]:
    """Convert month chronons to the first day of the containing month.

    Returns ``(div, mod)``: the day count of the month start and the residual.
    """
    ...

def year_to_day(x: _F64, from_n: float = 1.0, to_n: float = 1.0) -> tuple[_F64, _F64]:
    """Convert year chronons to the first day of the containing year.

    Returns ``(div, mod)``: the day count of the year start and the residual.
    """
    ...
