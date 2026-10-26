"""Fixture tests: the Rust port must match R's `chronon_divmod` output.

Inputs are passed as NumPy arrays, which is the supported boundary: the
extension reads the array's buffer in place and never converts a Python list.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import mixtime

FIXTURE_DIR = Path(
    os.environ.get(
        "MIXTIME_FIXTURES",
        Path(__file__).resolve().parents[2] / "fixtures",
    )
)
if FIXTURE_DIR.is_file():
    FIXTURE_DIR = FIXTURE_DIR.parent

with (FIXTURE_DIR / "fixtures.json").open(encoding="utf-8") as fh:
    FIXTURES = json.load(fh)
with (FIXTURE_DIR / "crosscheck.json").open(encoding="utf-8") as fh:
    CROSSCHECK = json.load(fh)

CASES = FIXTURES["cases"]


def as_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def run_case(case: dict) -> tuple[list[float], list[float]]:
    x = as_array(case["input"])
    from_unit, from_n = case["from"]["unit"], case["from"]["n"]
    to_unit, to_n = case["to"]["unit"], case["to"]["n"]

    if from_unit == "day" and to_unit == "month":
        if to_n == 3:
            return mixtime.day_to_quarter(x, from_n=from_n)
        return mixtime.day_to_month(x, from_n=from_n, to_n=to_n)
    if from_unit == "day" and to_unit == "year":
        return mixtime.day_to_year(x, from_n=from_n, to_n=to_n)
    if from_unit == "month" and to_unit == "day":
        return mixtime.month_to_day(x, from_n=from_n, to_n=to_n)
    if from_unit == "year" and to_unit == "day":
        return mixtime.year_to_day(x, from_n=from_n, to_n=to_n)

    raise AssertionError(f"Unhandled fixture pair: {from_unit} -> {to_unit}")


def run_crosscheck_case(case: dict) -> tuple[list[float], list[float]]:
    x = as_array(case["input"])
    from_n, to_n = case["from_n"], case["to_n"]

    if case["name"] == "day_to_quarter":
        return mixtime.day_to_quarter(x, from_n=from_n)
    if case["name"] in ("day_to_month", "twoday_to_month"):
        return mixtime.day_to_month(x, from_n=from_n, to_n=to_n)
    if case["name"] in ("day_to_year", "twoday_to_year"):
        return mixtime.day_to_year(x, from_n=from_n, to_n=to_n)
    if case["name"] in ("month_to_day", "bimonth_to_day", "month_to_2day"):
        return mixtime.month_to_day(x, from_n=from_n, to_n=to_n)
    if case["name"] in ("year_to_day", "year_to_2day"):
        return mixtime.year_to_day(x, from_n=from_n, to_n=to_n)

    raise AssertionError(f"Unhandled crosscheck case: {case['name']}")


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_matches_r_fixture(case: dict) -> None:
    div, mod = run_case(case)
    expected = case["expected"]
    expected_div = expected["div"]
    # R returns `mod = 0L` for the inverse conversions (a length-1 vector in
    # JSON) and recycles it across the result vector.
    expected_mod = (
        expected["mod"] * len(expected_div)
        if len(expected["mod"]) == 1 and len(expected_div) > 1
        else expected["mod"]
    )

    assert len(div) == len(expected_div)
    assert len(mod) == len(expected_mod)
    np.testing.assert_allclose(div, expected_div, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mod, expected_mod, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "case", CROSSCHECK["cases"], ids=[case["name"] for case in CROSSCHECK["cases"]]
)
def test_matches_r_randomized_crosscheck(case: dict) -> None:
    div, mod = run_crosscheck_case(case)
    expected_div = case["div"]
    expected_mod = (
        case["mod"] * len(expected_div)
        if len(case["mod"]) == 1 and len(expected_div) > 1
        else case["mod"]
    )

    assert len(div) == len(expected_div)
    assert len(mod) == len(expected_mod)
    np.testing.assert_allclose(div, expected_div, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mod, expected_mod, rtol=1e-12, atol=1e-12)


def test_outputs_are_float64_numpy_arrays() -> None:
    x = np.array([0.0, 1.0, 31.25])
    div, mod = mixtime.day_to_month(x)

    for out in (div, mod):
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.ndim == 1
        assert out.shape == (len(x),)
        assert out.flags.writeable
        assert not np.shares_memory(out, x)


def test_crosscheck_fixture_shape() -> None:
    assert len(CROSSCHECK["inputs"]["days"]) == 120
    assert len(CROSSCHECK["inputs"]["months"]) == 120
    assert len(CROSSCHECK["inputs"]["years"]) == 120
    assert len(CROSSCHECK["cases"]) == 10


def test_list_input_is_rejected() -> None:
    """The boundary is a NumPy buffer; a list is not silently converted."""
    with pytest.raises(TypeError):
        mixtime.day_to_month([0.0, 1.0])
