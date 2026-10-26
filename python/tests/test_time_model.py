"""Tests for the Pythonic time-vector model against R-generated fixtures."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

import mixtime as mt


FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "time_model.json"
FIXTURES = json.loads(FIXTURE_PATH.read_text())


def _listify(value):
    if isinstance(value, list):
        return value
    return [value]


def _expected_values(case):
    return _listify(case["values"])


def _assert_values(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=object)
    assert actual.shape == expected.shape
    for got, want in zip(actual, expected):
        if want is None:
            assert np.isnan(got)
        else:
            assert np.isclose(float(got), float(want), rtol=0, atol=1e-12)


def _assert_formatted(actual, expected):
    assert list(actual) == _listify(expected)


def _assert_metadata(actual, case):
    assert actual.chronon.name == case["chronon"]
    assert float(actual.chronon.n) == float(case["n"])
    if case.get("cycle") not in (None, "isoyear"):
        assert actual.cycle is not None
        assert actual.cycle.name == case["cycle"]
    if case["chronon"] != "week":
        assert actual.calendar.name == "gregorian"
    assert actual.discrete == case["discrete"]


CONSTRUCTOR_INPUTS = {
    "year_strings": (mt.year, ["2024", "2025"], True),
    "yearquarter_strings": (mt.yearquarter, ["2024-Q1", "2024-Q2", "2023-Q4"], True),
    "yearmonth_strings": (mt.yearmonth, ["2024-01", "2024-02", "2024-03"], True),
    "yearweek_strings": (mt.yearweek, ["2024-W01", "2024-W03", "2024-W09"], True),
    "date_strings": (mt.date, ["2024-01-15", "2024-02-29", "1970-01-01"], True),
    "datetime_strings": (
        mt.datetime,
        ["2024-01-15 12:34:56", "2024-01-15 00:00:00"],
        True,
    ),
    "yearmonth_numeric_double": (mt.yearmonth, 648.0, None),
    "yearmonth_numeric_integer": (mt.yearmonth, 648, None),
    "yearmonth_continuous": (mt.yearmonth, dt.date(2024, 1, 15), False),
    "yearquarter_continuous": (mt.yearquarter, dt.date(2024, 2, 15), False),
    "year_continuous": (mt.year, dt.date(2024, 7, 1), False),
    "date_continuous": (mt.date, dt.datetime(2024, 1, 15, 12), False),
    "datetime_continuous": (
        mt.datetime,
        dt.datetime(2024, 1, 15, 12, 0, 0, 250000),
        False,
    ),
}


@pytest.mark.parametrize("case", FIXTURES["constructors"], ids=lambda c: c["name"])
def test_constructors(case):
    constructor, data, discrete = CONSTRUCTOR_INPUTS[case["name"]]
    actual = constructor(data, discrete=discrete)
    _assert_values(actual.values, _expected_values(case))
    _assert_formatted(actual.format(), case["formatted"])
    _assert_metadata(actual, case)


PARSING_INPUTS = {
    "yearmonth_2024_jan": (mt.yearmonth, "2024 Jan"),
    "yearmonth_jan_2024": (mt.yearmonth, "Jan 2024"),
    "yearmonth_slash": (mt.yearmonth, "2024/1"),
    "yearmonth_dash": (mt.yearmonth, "2024-1"),
    "yearquarter_2024_q1": (mt.yearquarter, "2024 Q1"),
    "yearquarter_compact": (mt.yearquarter, "2024Q1"),
    "yearquarter_q1_2024": (mt.yearquarter, "Q1 2024"),
    "yearquarter_slash": (mt.yearquarter, "2024/1"),
    "yearquarter_month": (mt.yearquarter, "2024-01"),
    "date_day_month_name": (mt.date, "15 Jan 2024"),
    "date_month_name_day": (mt.date, "Jan 15, 2024"),
    "date_slash": (mt.date, "2024/01/15"),
    "date_short_digits": (mt.date, "2024-1-5"),
    "datetime_iso_t": (mt.datetime, "2024-01-15T12:34:56"),
}


@pytest.mark.parametrize("case", FIXTURES["parsing"], ids=lambda c: c["name"])
def test_parsing(case):
    constructor, data = PARSING_INPUTS[case["name"]]
    actual = constructor(data)
    _assert_values(actual.values, _expected_values(case))
    _assert_formatted(actual.format(), case["formatted"])
    _assert_metadata(actual, case)


CYCLICAL_INPUTS = {
    "month_of_year": (
        mt.month_of_year,
        ["2024-01-15", "2024-02-29", "2024-12-31"],
        True,
    ),
    "day_of_year": (
        mt.day_of_year,
        ["2024-01-15", "2024-02-29", "2024-12-31"],
        True,
    ),
    "day_of_month": (
        mt.day_of_month,
        ["2024-01-15", "2024-02-29", "2024-12-31"],
        True,
    ),
    "day_of_week": (
        mt.day_of_week,
        ["2024-01-15", "2024-02-29", "2024-12-31"],
        True,
    ),
    "week_of_year": (
        mt.week_of_year,
        ["2024-01-15", "2024-02-29", "2024-12-31"],
        True,
    ),
    "time_of_day": (
        mt.time_of_day,
        ["2024-01-15 12:34:56", "2024-01-15 00:00:00"],
        True,
    ),
    "month_of_year_continuous": (mt.month_of_year, dt.date(2024, 1, 15), False),
    "day_of_year_continuous": (mt.day_of_year, dt.date(2024, 1, 15), False),
    "day_of_month_continuous": (mt.day_of_month, dt.date(2024, 1, 15), False),
    "day_of_week_continuous": (mt.day_of_week, dt.date(2024, 1, 15), False),
    "day_of_week_continuous_datetime": (
        mt.day_of_week,
        dt.datetime(2024, 1, 15, 12),
        False,
    ),
    "week_of_year_continuous": (mt.week_of_year, dt.date(2024, 1, 15), False),
    "time_of_day_continuous": (
        mt.time_of_day,
        dt.datetime(2024, 1, 15, 12, 34, 56, 250000),
        False,
    ),
}


@pytest.mark.parametrize("case", FIXTURES["cyclical"], ids=lambda c: c["name"])
def test_cyclical(case):
    constructor, data, discrete = CYCLICAL_INPUTS[case["name"]]
    actual = constructor(data, discrete=discrete)
    _assert_values(actual.values, _expected_values(case))
    _assert_formatted(actual.format(), case["formatted"])
    assert actual.kind == "cyclical"
    _assert_metadata(actual, case)


def _duration_months_3(data, **kwargs):
    return mt.duration(data, mt.cal_gregorian.month(3), **kwargs)


DURATION_INPUTS = {
    "years": (mt.years, [1, 2, 0.5]),
    "quarters": (mt.quarters, [1, 2, 0.5]),
    "months": (mt.months, [1, 2, 0.5]),
    "weeks": (mt.weeks, [1, 2, 0.5]),
    "days": (mt.days, [1, 2, 0.5]),
    "hours": (mt.hours, [1, 2, 0.5]),
    "minutes": (mt.minutes, [1, 2, 0.5]),
    "seconds": (mt.seconds, [1, 2, 0.5]),
    "milliseconds": (mt.milliseconds, [1, 2, 0.5]),
    "years_fractional": (mt.years, [1 / 3, 1.5, 2]),
    "months_fractional": (mt.months, [0.967741935483871]),
    "days_fractional": (mt.days, [1 / 3, -0.5]),
    "seconds_fractional": (mt.seconds, [0.123456789]),
    "milliseconds_fractional": (mt.milliseconds, [1.2345]),
    "years_discrete": (mt.years, [1, 2, 0, -1]),
    "days_discrete": (mt.days, [1, 2, 0, -1]),
    "months_3_discrete": (_duration_months_3, [1, 2]),
    "months_3_continuous": (_duration_months_3, [1.5, 2.5]),
}


@pytest.mark.parametrize("case", FIXTURES["durations"], ids=lambda c: c["name"])
def test_durations(case):
    constructor, data = DURATION_INPUTS[case["name"]]
    actual = constructor(data, discrete=case["discrete"])
    _assert_values(actual.values, _expected_values(case))
    _assert_formatted(actual.format(), case["formatted"])
    assert actual.kind == "duration"
    _assert_metadata(actual, case)


OPERATION_INPUTS = {
    "floor_year": lambda: mt.time_floor(
        mt.yearmonth(["2024-01", "2024-02", "2024-07", "2024-12"]), "year"
    ),
    "ceiling_year": lambda: mt.time_ceiling(
        mt.yearmonth(["2024-01", "2024-02", "2024-07", "2024-12"]), "year"
    ),
    "round_year": lambda: mt.time_round(
        mt.yearmonth(["2024-01", "2024-02", "2024-07", "2024-12"]), "year"
    ),
    "seq_by_two": lambda: mt.seq(
        mt.yearmonth("2024-01"), mt.yearmonth("2024-06"), by=2
    ),
    "add_months": lambda: mt.yearmonth(
        ["2024-01", "2024-02", "2024-07", "2024-12"]
    )
    + mt.months(1),
    "subtract_months": lambda: mt.yearmonth(
        ["2024-01", "2024-02", "2024-07", "2024-12"]
    )
    - mt.months(1),
    "difference_months": lambda: mt.yearmonth(
        ["2024-01", "2024-02", "2024-07", "2024-12"]
    )
    - mt.yearmonth("2023-12"),
    "difference_cross": lambda: mt.yearmonth(
        dt.date(2024, 1, 31), discrete=False
    )
    - mt.date(dt.date(2024, 1, 1), discrete=False),
    "difference_quarter_month": lambda: mt.yearquarter(
        dt.date(2024, 1, 31), discrete=False
    )
    - mt.yearmonth(dt.date(2024, 1, 1), discrete=False),
    "difference_month_week": lambda: mt.yearmonth(
        dt.date(2024, 1, 31), discrete=False
    )
    - mt.yearweek(dt.date(2024, 1, 1), discrete=False),
    "add_continuous_months": lambda: mt.yearmonth(
        dt.date(2024, 1, 31), discrete=False
    )
    + mt.months(1),
    "duration_common_weeks_days": lambda: mt.weeks(1) + mt.days(7),
    "duration_common_months_years": lambda: mt.months(1) + mt.years(1),
    "duration_discrete_add": lambda: mt.years(1, discrete=True)
    + mt.months(1, discrete=True),
    "duration_discrete_sub": lambda: mt.years(1, discrete=True)
    - mt.months(1, discrete=True),
    "duration_discrete_scalar_add": lambda: mt.years(1, discrete=True) + 1,
    "duration_discrete_scalar_sub": lambda: mt.years(1, discrete=True) - 0.5,
    "duration_discrete_scalar_mul": lambda: mt.years(1, discrete=True) * 2,
    "duration_discrete_scalar_div": lambda: mt.years(1, discrete=True) / 2,
    "time_shift_months": lambda: mt.time_shift(
        mt.yearmonth(["2024-01", "2024-02", "2024-07", "2024-12"]),
        mt.months(1),
    ),
    "time_shift_days": lambda: mt.time_shift(
        mt.date(["2024-01-15", "2024-02-29"]), mt.days(1)
    ),
    "date_timezone_utc": lambda: mt.date(
        dt.datetime(
            2024,
            1,
            15,
            23,
            tzinfo=dt.timezone(dt.timedelta(hours=-5)),
        ),
        tz="UTC",
    ),
    "time_shift_continuous_months": lambda: mt.time_shift(
        mt.yearmonth(dt.date(2024, 1, 31), discrete=False), mt.months(1)
    ),
    "continuous_string_month": lambda: mt.yearmonth("2024-01", discrete=False),
    "continuous_string_quarter": lambda: mt.yearquarter(
        "2024-Q1", discrete=False
    ),
    "continuous_string_week": lambda: mt.yearweek("2024-W03", discrete=False),
    "continuous_string_year": lambda: mt.year("2024", discrete=False),
    "continuous_string_date": lambda: mt.date("2024-01-15", discrete=False),
    "continuous_string_datetime": lambda: mt.datetime(
        "2024-01-15 12:34:56", discrete=False
    ),
    "seq_length_out_fractional": lambda: mt.seq(
        mt.yearmonth("2024-01"),
        mt.yearmonth("2024-12"),
        length_out=5,
    ),
    "seq_length_out_integral": lambda: mt.seq(
        mt.date("2024-01-01"),
        mt.date("2024-01-10"),
        length_out=4,
    ),
    "seq_fractional_step_month": lambda: mt.seq(
        mt.yearmonth("2024-01"), mt.yearmonth("2024-02"), by=0.5
    ),
    "seq_fractional_step_date": lambda: mt.seq(
        mt.date("2024-01-01"), mt.date("2024-01-02"), by=0.5
    ),
    "seq_fractional_step_date_06": lambda: mt.seq(
        mt.date("2024-01-01"), mt.date("2024-01-02"), by=0.6
    ),
}


@pytest.mark.parametrize("case", FIXTURES["operations"], ids=lambda c: c["name"])
def test_operations(case):
    actual = OPERATION_INPUTS[case["name"]]()
    _assert_values(actual.values, _expected_values(case))
    _assert_formatted(actual.format(), case["formatted"])
    _assert_metadata(actual, case)


COMPARISON_INPUTS = {
    "month_vs_day_start": lambda: mt.yearmonth("2024-01")
    == mt.date("2024-01-01"),
    "month_vs_day_end": lambda: mt.yearmonth("2024-01") == mt.date("2024-01-31"),
    "month_vs_month": lambda: mt.yearmonth("2024-01")
    == mt.yearmonth("2024-01"),
    "continuous_month_vs_day": lambda: mt.yearmonth(
        dt.date(2024, 1, 15), discrete=False
    )
    == mt.date(dt.date(2024, 1, 15), discrete=False),
    "span_vs_instant_same_value": lambda: mt.yearmonth("2024-01")
    == mt.yearmonth(dt.date(2024, 1, 1), discrete=False),
    "duration_discrete_vs_continuous": lambda: mt.years(1, discrete=True)
    == mt.years(1),
    "month_vs_day_gt": lambda: mt.yearmonth("2024-01") > mt.date("2024-01-01"),
    "month_vs_day_lt": lambda: mt.yearmonth("2024-01") < mt.date("2024-01-01"),
    "month_vs_day_ge": lambda: mt.yearmonth("2024-01") >= mt.date("2024-01-01"),
}


@pytest.mark.parametrize("case", FIXTURES["comparisons"], ids=lambda c: c["name"])
def test_comparisons(case):
    actual = COMPARISON_INPUTS[case["name"]]()
    assert bool(np.all(actual)) == case["result"]


@pytest.mark.parametrize("case", FIXTURES["mixed"], ids=lambda c: c["name"])
def test_mixed(case):
    actual = mt.mixtime(
        mt.yearmonth(["2024-01", "2024-02"]),
        mt.yearquarter(["2024-Q2", "2024-Q3"]),
    )
    _assert_values(actual.values, _expected_values(case))
    _assert_formatted(actual.format(), case["formatted"])
    assert len(actual) == len(case["values"])


# -- Python API behaviours not directly represented in the R fixture --------


def test_indexing_returns_length_one_vectors():
    x = mt.yearmonth(["2024-01", "2024-02"])
    assert len(x[0]) == 1
    assert x[0].format() == ["2024 Jan"]
    assert x[-1].format() == ["2024 Feb"]


def test_mixed_indexing_including_negative_index():
    m = mt.mixtime(mt.yearmonth(["2024-01"]), mt.yearquarter(["2024-Q2"]))
    assert m[0].format() == ["2024 Jan"]
    assert m[-1].format() == ["2024 Q2"]
    assert m[1:].format() == ["2024 Q2"]


def test_format_tokens():
    x = mt.yearmonth(["2024-01", "2024-02"])
    assert x.format("%Y-%m") == ["2024-01", "2024-02"]
    assert x.format("{lin(year)}-{cyc(month,year)}") == ["2024-Jan", "2024-Feb"]
    assert x.format("%%Y") == ["%Y", "%Y"]
    assert mt.yearweek("2024-W03").format("%Y-%m-%d") == ["2024-01-18"]


def test_arithmetic_and_comparison():
    x = mt.yearmonth(["2024-01", "2024-02"])
    y = x + mt.months(1)
    assert y.format() == ["2024 Feb", "2024 Mar"]
    assert np.array_equal(y > x, [True, True])
    assert np.array_equal(y - mt.months(1) == x, [True, True])


def test_timezone_aware_datetime():
    x = mt.datetime("2024-01-15 12:34:56+00:00", tz="UTC")
    assert x.format() == ["2024-01-15 12:34:56"]


def test_offset_date_strings_convert_to_requested_timezone():
    assert mt.date("2024-01-15T23:00:00-05:00").format() == ["2024-01-16"]
    assert mt.date(
        "2024-01-15T23:00:00-05:00", tz="America/New_York"
    ).format() == ["2024-01-15"]
    assert mt.yearweek("2024-01-15T23:00:00-05:00").format() == ["2024 W03"]


def test_pandas_is_optional():
    pytest.importorskip("pandas")
    x = mt.yearmonth(["2024-01", "2024-02"])
    assert list(x.to_pandas().astype(str)) == ["2024-01", "2024-02"]
    assert list(mt.date(["2024-01-15", "2024-02-29"]).to_pandas().astype(str)) == [
        "2024-01-15",
        "2024-02-29",
    ]
    assert list(
        mt.datetime(["2024-01-15 12:34:56"]).to_pandas().astype(str)
    ) == ["2024-01-15 12:34:56"]
    assert list(
        mt.yearweek(["2024-W01", "2024-W03"]).to_pandas().astype(str)
    ) == ["2024-01-01/2024-01-07", "2024-01-15/2024-01-21"]


def test_discrete_shift_rejects_indeterminate_duration():
    quarters = mt.yearquarter(["2024-Q2", "2024-Q3"])
    with pytest.raises(ValueError, match="temporally indeterminate"):
        quarters + mt.months(1)
    assert (quarters + mt.months(3)).format() == ["2024 Q3", "2024 Q4"]

    dates = mt.date(["2024-01-15", "2024-02-29"])
    with pytest.raises(ValueError, match="temporally indeterminate"):
        dates + mt.months(1)
    assert (dates + mt.weeks(1)).format() == ["2024-01-22", "2024-03-07"]


def test_different_chronon_difference_requires_continuous():
    with pytest.raises(ValueError, match="indeterminate"):
        mt.yearmonth("2024-01") - mt.date("2024-01-01")


def test_mixed_rounding_applies_to_each_part():
    m = mt.mixtime(
        mt.yearmonth(["2024-01", "2024-02"]),
        mt.yearquarter(["2024-Q2", "2024-Q3"]),
    )
    assert mt.time_floor(m, "year").format() == [
        "2024 Jan",
        "2024 Jan",
        "2024 Q1",
        "2024 Q1",
    ]


def test_mixed_time_shift_applies_to_each_part():
    m = mt.mixtime(mt.yearmonth(["2024-01"]), mt.yearquarter(["2024-Q2"]))
    assert mt.time_shift(m, mt.months(3)).format() == [
        "2024 Apr",
        "2024 Q3",
    ]


def test_sequence_rejects_indeterminate_discrete_step():
    start = mt.yearquarter("2024-Q1")
    end = mt.yearquarter("2024-Q4")
    with pytest.raises(ValueError, match="temporally indeterminate"):
        mt.seq(start, end, by=mt.months(1))
    assert mt.seq(start, end, by=mt.months(3)).format() == [
        "2024 Q1",
        "2024 Q2",
        "2024 Q3",
        "2024 Q4",
    ]


def test_rounding_rejects_zero_granule_duration():
    with pytest.raises(ValueError, match="must not be zero"):
        mt.time_floor(mt.yearmonth("2024-01"), mt.months(0))


def test_week_granule_uses_monday_start():
    dates = mt.date(["2024-01-15", "2024-01-17", "2024-01-18", "2024-01-21"])
    assert mt.time_floor(dates, "week").format() == ["2024-01-15"] * 4
    assert mt.time_ceiling(dates, "week").format() == ["2024-01-22"] * 4
    assert mt.time_round(dates, "week").format() == [
        "2024-01-15",
        "2024-01-15",
        "2024-01-15",
        "2024-01-22",
    ]


def test_week_duration_is_seven_days():
    assert (mt.date("2024-01-15") + mt.weeks(1)).format() == ["2024-01-22"]
    assert (mt.date("2024-01-15") + mt.days(7)).format() == ["2024-01-22"]
    assert mt.weeks(1).astype("day").values.tolist() == [7]


def test_yearweek_to_date_uses_monday_start():
    weeks = mt.yearweek(["2024-W01", "2024-W03"])
    assert mt.date(weeks).format() == ["2024-01-01", "2024-01-15"]
    assert mt.date(weeks).values.tolist() == [19723, 19737]


def test_continuous_cross_category_shifts_are_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="Continuous shifts"):
        mt.date(dt.date(2024, 1, 31), discrete=False) + mt.months(1)
    with pytest.raises(NotImplementedError, match="Continuous shifts"):
        mt.yearmonth(dt.date(2024, 1, 31), discrete=False) + mt.days(1)
    with pytest.raises(ValueError, match="calendar and fixed"):
        mt.months(1).astype("day")
    with pytest.raises(ValueError, match="calendar and fixed"):
        mt.months(1) + mt.days(31)


def test_duration_and_time_point_comparisons_raise():
    with pytest.raises(TypeError, match="duration with a time point"):
        mt.years(1) == mt.yearmonth("2024-01")
    with pytest.raises(TypeError, match="duration with a time point"):
        mt.years(1) > mt.yearmonth("2024-01")


def test_duration_comparison_uses_fixed_week_length():
    assert np.array_equal(mt.weeks(1) == mt.days(7), [True])
    assert np.array_equal(mt.weeks(1) > mt.days(6), [True])
    assert (mt.weeks(1) - mt.days(7)).format() == ["0.0 days"]
    assert (mt.weeks(1) + mt.days(7)).format() == ["14.0 days"]
    assert (mt.months(1) + mt.years(1)).format() == ["13.0 months"]
    assert (mt.quarters(1) + mt.years(1)).format() == ["5.0 quarters"]


def test_duration_defaults_to_continuous_and_supports_discrete():
    assert mt.years(1).discrete is False
    assert mt.years(1).format() == ["1.0 years"]
    assert mt.years(1, discrete=True).discrete is True
    assert mt.years(1, discrete=True).format() == ["1 year"]
    assert mt.years([1, 2]).format() == ["1.0 years", "2.0 years"]
    assert mt.years([1, 2], discrete=True).format() == ["1 year", "2 years"]


def test_sequence_by_week_name_adds_seven_days():
    start = mt.date("2024-01-15")
    end = mt.date("2024-02-05")
    assert mt.seq(start, end, by="week").format() == [
        "2024-01-15",
        "2024-01-22",
        "2024-01-29",
        "2024-02-05",
    ]


def test_duration_rejects_time_point_input():
    with pytest.raises(ValueError, match="duration vectors"):
        mt.duration(mt.yearmonth("2024-01"), "month")


def test_duration_conversion_uses_fixed_ratios_without_an_anchor():
    assert mt.duration(mt.weeks(1), "day").format() == ["7.0 days"]
    assert mt.duration(mt.days(4), "week", discrete=True).format() == ["0 weeks"]
    assert mt.duration(mt.days(4), "week", discrete=False).format() == [
        "0.5714286 weeks"
    ]
    with pytest.raises(ValueError, match="calendar and fixed"):
        mt.duration(mt.months(1), "day")


def test_discrete_duration_conversion_truncates_toward_zero():
    assert mt.days(-4).astype("week", discrete=True).format() == ["0 weeks"]
    assert mt.duration(mt.days(-4), "week", discrete=True).format() == ["0 weeks"]
    assert mt.days(-4).astype("week", discrete=False).format() == [
        "-0.5714286 weeks"
    ]


def test_sequence_rejects_incompatible_chronons():
    with pytest.raises(ValueError, match="Incompatible time chronons"):
        mt.seq(mt.yearmonth("2024-01"), mt.yearquarter("2024-Q4"))
    with pytest.raises(ValueError, match="Incompatible time chronons"):
        mt.seq(mt.date("2024-01-01"), mt.yearmonth("2024-12"))


def test_duration_arithmetic_returns_continuous_durations():
    x = mt.years(1, discrete=True) + mt.months(1, discrete=True)
    assert x.discrete is False
    assert x.format() == ["13.0 months"]
    y = mt.years(1, discrete=True) - mt.months(1, discrete=True)
    assert y.discrete is False
    assert y.format() == ["11.0 months"]


def test_discrete_duration_scalar_arithmetic_is_continuous():
    x = mt.years(1, discrete=True)
    assert (x + 1).format() == ["2.0 years"]
    assert (x - 0.5).format() == ["0.5 years"]
    assert (x * 2).format() == ["2.0 years"]
    assert (x / 2).format() == ["0.5 years"]
    assert (-x).format() == ["-1.0 years"]


def test_continuous_values_are_stored_as_double():
    x = mt.yearmonth("2024-01", discrete=False)
    assert x.values.dtype == np.float64
    assert mt.yearmonth("2024-01").values.dtype == np.int64


def test_time_constructors_reject_duration_input():
    with pytest.raises(ValueError, match="duration to a time point"):
        mt.yearmonth(mt.months(1))
    with pytest.raises(ValueError, match="duration to a time point"):
        mt.date(mt.days(1))


def test_inequality_is_elementwise():
    x = mt.yearmonth(["2024-01", "2024-02"])
    y = mt.yearmonth(["2024-01", "2024-03"])
    assert np.array_equal(x != y, [False, True])


def test_sequence_rejects_combining_by_and_length_out():
    with pytest.raises(ValueError, match="combining by and length_out"):
        mt.seq(
            mt.yearmonth("2024-01"),
            mt.yearmonth("2024-12"),
            by=2,
            length_out=5,
        )


def test_discrete_inference_from_input_type():
    assert mt.yearmonth(648).discrete is True
    assert mt.yearmonth(648).format() == ["2024 Jan"]
    assert mt.yearmonth(648.0).discrete is False
    assert mt.yearmonth(648.0).format() == ["2024 Jan 0.0%"]
    assert mt.yearmonth(648.5).discrete is False
    assert mt.yearmonth(648.5).format() == ["2024 Jan 50.0%"]
    assert mt.yearmonth([648, 649]).discrete is True
    assert mt.yearmonth([648.0, 649.0]).discrete is False
    assert mt.date(dt.date(2024, 1, 15)).discrete is True
    assert mt.month_of_year("2024-01-15").discrete is True


def test_sequence_validates_length_one_and_finite_steps():
    with pytest.raises(ValueError, match="length-one"):
        mt.seq(
            mt.yearmonth(["2024-01", "2024-02"]),
            mt.yearmonth("2024-06"),
        )
    with pytest.raises(ValueError, match="finite non-zero"):
        mt.seq(mt.yearmonth("2024-01"), mt.yearmonth("2024-06"), by=0)
    with pytest.raises(ValueError, match="finite non-zero"):
        mt.seq(mt.yearmonth("2024-01"), mt.yearmonth("2024-06"), by=np.inf)


def test_time_shift_requires_a_duration():
    with pytest.raises(TypeError, match="Duration"):
        mt.time_shift(mt.yearmonth("2024-01"), mt.yearmonth("2024-02"))


def test_duration_conversion_rejects_cross_category_even_when_anchored():
    with pytest.raises(ValueError, match="Cannot convert a duration"):
        mt.cal_gregorian.convert(
            np.asarray([1.0]),
            mt.cal_gregorian.month,
            mt.cal_gregorian.day,
            duration=True,
            anchored=True,
        )
