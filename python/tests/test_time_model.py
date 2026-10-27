"""Tests for the Pythonic time-vector model against R-generated fixtures."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

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
    "time_shift_date_months_clamp": lambda: mt.time_shift(
        mt.date("2024-01-31"), mt.months(1)
    ),
    "time_shift_date_years_leap": lambda: mt.time_shift(
        mt.date("2024-02-29"), mt.years(1)
    ),
    "time_shift_datetime_days_dst_forward": lambda: mt.time_shift(
        mt.datetime("2024-03-09 12:00:00", tz="America/New_York"),
        mt.days(1),
    ),
    "time_shift_datetime_hours_dst_forward": lambda: mt.time_shift(
        mt.datetime("2024-03-09 12:00:00", tz="America/New_York"),
        mt.hours(24),
    ),
    "time_shift_datetime_days_dst_backward": lambda: mt.time_shift(
        mt.datetime("2024-11-02 12:00:00", tz="America/New_York"),
        mt.days(1),
    ),
    "time_shift_datetime_days_vector": lambda: mt.time_shift(
        mt.datetime(
            ["2024-03-09 12:00:00", "2024-11-02 12:00:00"],
            tz="America/New_York",
        ),
        mt.days(1),
    ),
    "time_shift_datetime_months_clamp_tz": lambda: mt.time_shift(
        mt.datetime("2021-01-31 09:30:00", tz="America/New_York"),
        mt.months(1),
    ),
    "time_shift_datetime_years_leap_tz": lambda: mt.time_shift(
        mt.datetime("2024-02-29 09:30:00", tz="America/New_York"),
        mt.years(1),
    ),
    "time_shift_datetime_quarters_tz": lambda: mt.time_shift(
        mt.datetime("2024-01-31 09:30:00", tz="America/New_York"),
        mt.quarters(1),
    ),
    "time_shift_melbourne_gap": lambda: mt.time_shift(
        mt.datetime("2025-10-04 02:30:00", tz="Australia/Melbourne"),
        mt.days(1),
    ),
    "time_shift_melbourne_ambiguous": lambda: mt.time_shift(
        mt.datetime("2025-04-05 02:30:00", tz="Australia/Melbourne"),
        mt.days(1),
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


PREDICATE_INPUTS = {
    "linear_yearmonth": lambda: mt.time_is_linear(
        mt.yearmonth(["2024-01-15", "2024-02-29"])
    ),
    "linear_month_of_year": lambda: mt.time_is_linear(
        mt.month_of_year(["2024-01-15", "2024-02-29"])
    ),
    "cyclical_yearmonth": lambda: mt.time_is_cyclical(
        mt.yearmonth(["2024-01-15", "2024-02-29"])
    ),
    "cyclical_month_of_year": lambda: mt.time_is_cyclical(
        mt.month_of_year(["2024-01-15", "2024-02-29"])
    ),
    "duration_months": lambda: mt.time_is_duration(mt.months([1, 2])),
    "duration_yearmonth": lambda: mt.time_is_duration(
        mt.yearmonth(["2024-01-15", "2024-02-29"])
    ),
    "determinate_year_month": lambda: mt.time_is_determinate_at(
        mt.year(2020), mt.cal_gregorian.month(1)
    ),
    "determinate_continuous_year_month": lambda: mt.time_is_determinate_at(
        mt.year(2020, discrete=False), mt.cal_gregorian.month(1)
    ),
    "determinate_month_year": lambda: mt.time_is_determinate_at(
        mt.yearmonth("2020-02"), mt.cal_gregorian.year(1)
    ),
    "determinate_month_week": lambda: mt.time_is_determinate_at(
        mt.yearmonth("2020-01"), mt.cal_gregorian.week(1)
    ),
    "determinate_day_month": lambda: mt.time_is_determinate_at(
        mt.date("2020-01-01"), mt.cal_gregorian.month(1)
    ),
    "determinate_day_week": lambda: mt.time_is_determinate_at(
        mt.date("2020-01-01"), mt.cal_gregorian.week(1)
    ),
    "determinate_week_month": lambda: mt.time_is_determinate_at(
        mt.yearweek("2020-W01"), mt.cal_gregorian.month(1)
    ),
    "determinate_quarter_month": lambda: mt.time_is_determinate_at(
        mt.yearquarter("2020-Q1"), mt.cal_gregorian.month(1)
    ),
    "determinate_datetime_hour": lambda: mt.time_is_determinate_at(
        mt.datetime("2020-01-01 00:00:00"), mt.cal_gregorian.hour(1)
    ),
    "complete_year_partial": lambda: mt.time_is_complete_at(
        mt.yearmonth("2020-01") + mt.months(np.arange(10)),
        mt.cal_gregorian.year(1),
    ),
    "complete_year_full": lambda: mt.time_is_complete_at(
        mt.yearmonth("2020-01") + mt.months(np.arange(15)),
        mt.cal_gregorian.year(1),
    ),
    "complete_quarter_full": lambda: mt.time_is_complete_at(
        mt.yearquarter(["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4"]),
        mt.cal_gregorian.year(1),
    ),
    "complete_month_missing": lambda: mt.time_is_complete_at(
        mt.yearmonth(["2020-01", "2020-03"]),
        mt.cal_gregorian.year(1),
    ),
    "complete_days_month": lambda: mt.time_is_complete_at(
        mt.date("2020-01-01") + mt.days(np.arange(31)),
        mt.cal_gregorian.month(1),
    ),
    "complete_days_week": lambda: mt.time_is_complete_at(
        mt.date("2020-01-01") + mt.days(np.arange(7)),
        mt.cal_gregorian.week(1),
    ),
}


@pytest.mark.parametrize("case", FIXTURES["predicates"], ids=lambda c: c["name"])
def test_predicates(case):
    actual = PREDICATE_INPUTS[case["name"]]()
    assert list(actual) == _listify(case["result"])


def test_predicates_missing_and_multi_unit():
    x = mt.date([None, "2024-01-01"])
    assert list(mt.time_is_determinate_at(x, "month")) == [None, True]
    assert list(mt.time_is_complete_at(x, "year")) == [None, False]
    assert list(
        mt.time_is_determinate_at(mt.date("2020-01-01"), mt.cal_gregorian.month(3))
    ) == [True]
    assert list(
        mt.time_is_determinate_at(mt.date("2020-01-01"), mt.cal_gregorian.hour(5))
    ) == [False]

    m = mt.mixtime(mt.year(2020), mt.date("2020-01-01"))
    assert list(mt.time_is_determinate_at(m, "month")) == [False, True]
    assert list(mt.time_is_linear(m)) == [True, True]
    assert list(mt.time_is_duration(m)) == [False, False]
    with pytest.raises(ValueError, match="mixed-granularity"):
        mt.time_is_complete_at(m, "year")


COMPONENT_INPUTS = {
    "yearmonth": lambda: mt.time_components(
        mt.yearmonth(
            [
                dt.date(2026, 2, 14) + dt.timedelta(days=offset)
                for offset in (0, 40, 400)
            ]
        ),
        yr=mt.lin("year"),
        mth=mt.cyc("month", "year"),
    ),
    "date_week": lambda: mt.time_components(
        mt.date(
            [
                dt.date(2025, 12, 15) + dt.timedelta(days=offset)
                for offset in range(4)
            ]
        ),
        dow=mt.cyc("day", "week"),
        doy=mt.cyc("day", "year"),
    ),
    "datetime": lambda: mt.time_components(
        mt.datetime("2024-03-15 12:34:56"),
        yr=mt.lin("year"),
        mth=mt.cyc("month", "year"),
        dom=mt.cyc("day", "month"),
        hod=mt.cyc("hour", "day"),
        min=mt.cyc("minute", "hour"),
        sec=mt.cyc("second", "minute"),
        mq=mt.cyc("month", "quarter"),
        qy=mt.cyc("quarter", "year"),
    ),
    "indeterminate": lambda: mt.time_components(
        mt.yearmonth("2024-01"), day=mt.cyc("day", "month")
    ),
    "continuous": lambda: mt.time_components(
        mt.yearmonth(dt.date(2024, 1, 15), discrete=False),
        yr=mt.lin("year"),
        mth=mt.cyc("month", "year"),
        dom=mt.cyc("day", "month"),
    ),
}


@pytest.mark.parametrize("case", FIXTURES["components"], ids=lambda c: c["name"])
def test_components(case):
    actual = COMPONENT_INPUTS[case["name"]]()
    assert list(actual) == list(case["columns"])
    for name, expected in case["columns"].items():
        column = actual[name]
        _assert_values(column.values, _expected_values(expected))
        _assert_formatted(column.format(), expected["formatted"])
        assert column.chronon.name == expected["chronon"]["name"]
        assert float(column.chronon.n) == float(expected["chronon"]["n"])
        if expected["cycle"] is None:
            assert column.cycle is None
        else:
            assert column.cycle is not None
            assert column.cycle.name == expected["cycle"]["name"]
            assert float(column.cycle.n) == float(expected["cycle"]["n"])


COMPOSE_INPUTS = {
    "year_month_day": lambda: mt.time_compose(
        mt.lin("year", 1980),
        mt.cyc("month", "year", 3),
        mt.cyc("day", "month", 15),
    ),
    "year": lambda: mt.time_compose(mt.lin("year", 1980)),
    "month": lambda: mt.time_compose(mt.cyc("month", "year", 3)),
    "day_month": lambda: mt.time_compose(
        mt.cyc("day", "month", 15), mt.cyc("month", "year", 3)
    ),
    "multi_unit": lambda: mt.time_compose(
        mt.lin(mt.cal_gregorian.month(3), 3),
        mt.cyc(mt.cal_gregorian.month(1), mt.cal_gregorian.month(3), 3),
    ),
    "tagged": lambda: _compose_tagged(),
}


def _compose_tagged():
    parts = mt.time_components(
        mt.date(dt.date(2024, 3, 15)),
        yr=mt.lin("year"),
        mth=mt.cyc("month", "year"),
        dom=mt.cyc("day", "month"),
    )
    return mt.time_compose(parts["yr"], parts["mth"], parts["dom"])


@pytest.mark.parametrize("case", FIXTURES["compose"], ids=lambda c: c["name"])
def test_compose(case):
    actual = COMPOSE_INPUTS[case["name"]]()
    _assert_values(actual.values, _expected_values(case))
    _assert_formatted(actual.format(), case["formatted"])
    assert actual.chronon.name == case["chronon"]["name"]
    assert float(actual.chronon.n) == float(case["chronon"]["n"])
    if case["cycle"] is None:
        assert actual.cycle is None
    else:
        assert actual.cycle is not None
        assert actual.cycle.name == case["cycle"]["name"]
        assert float(actual.cycle.n) == float(case["cycle"]["n"])


def test_components_and_compose_python_api():
    mixed = mt.mixtime(
        mt.yearmonth(["2026-02", "2026-03"]),
        mt.date([dt.date(2026, 2, 14), dt.date(2026, 3, 15)]),
    )
    components = mt.time_components(
        mixed, yr=mt.lin("year"), mth=mt.cyc("month", "year")
    )
    assert isinstance(components["yr"], mt.Mixtime)
    assert components["yr"].format() == ["2026", "2026", "2026", "2026"]
    assert components["mth"].format() == ["Feb", "Mar", "Feb", "Mar"]

    composed = mt.time_compose(
        mt.lin("year", [1980, 1981]), mt.cyc("month", "year", 3)
    )
    assert composed.format() == ["1980 Mar", "1981 Mar"]

    with pytest.raises(ValueError, match="Invalid position"):
        mt.time_compose(mt.lin("year", 1980), mt.cyc("month", "year", 13))
    with pytest.raises(ValueError, match="at least one component"):
        mt.time_components(mt.date("2024-01-01"))
    with pytest.raises(TypeError, match="need values"):
        mt.time_compose(mt.lin("year"))


GRANULE_INPUTS = {
    "yearmonth_date": lambda: (
        mt.yearmonth("2024-01"),
        mt.date("2024-01-01"),
    ),
    "week_month": lambda: (
        mt.yearweek("2024-W01"),
        mt.yearmonth("2024-01"),
    ),
    "week_hour": lambda: (
        mt.yearweek("2024-W01"),
        mt.linear_time(1, "hour"),
    ),
    "second_minute": lambda: (
        mt.linear_time(1, "second"),
        mt.linear_time(1, "minute"),
    ),
    "year_quarter": lambda: (
        mt.year("2024"),
        mt.yearquarter("2024-Q1"),
    ),
    "month_2_3": lambda: (
        mt.cal_gregorian.month(2),
        mt.cal_gregorian.month(3),
    ),
    "year_2_month": lambda: (
        mt.cal_gregorian.year(2),
        mt.cal_gregorian.month(1),
    ),
    "week_day": lambda: (
        mt.yearweek("2024-W01"),
        mt.date("2024-01-01"),
    ),
    "year_week": lambda: (
        mt.year("2024"),
        mt.yearweek("2024-W01"),
    ),
}


def _assert_unit(actual, expected):
    assert actual.name == expected["name"]
    assert float(actual.n) == float(expected["n"])


@pytest.mark.parametrize("case", FIXTURES["granules"], ids=lambda c: c["name"])
def test_chronon_bounds(case):
    inputs = GRANULE_INPUTS[case["name"]]()
    _assert_unit(mt.chronon_glb(*inputs), case["glb"])
    if case["lub"] is None:
        with pytest.raises(ValueError, match="common coarser chronon"):
            mt.chronon_lub(*inputs)
    else:
        _assert_unit(mt.chronon_lub(*inputs), case["lub"])


@pytest.mark.parametrize("case", FIXTURES["circsum"], ids=lambda c: c["name"])
def test_circsum(case):
    result = mt.circsum(case["x"], case["size"], case["step"])
    _assert_values(result, case["result"])


def test_circsum_default_step_and_chronon_helpers():
    _assert_values(mt.circsum([1, 2, 3, 4, 5], 3), [6, 10, 9, 8, 12])

    mixed = mt.mixtime(mt.yearmonth("2024-01"), mt.date("2024-01-01"))
    _assert_unit(mt.chronon_glb(mixed), {"name": "day", "n": 1})
    _assert_unit(mt.chronon_lub(mixed), {"name": "month", "n": 1})
    _assert_unit(mt.chronon_common(mixed), {"name": "day", "n": 1})

    _assert_unit(
        mt.chronon_glb(mt.yearmonth("2024-01"), ptype="day"),
        {"name": "day", "n": 1},
    )
    _assert_unit(
        mt.chronon_lub(mt.yearmonth("2024-01"), ptype="year"),
        {"name": "year", "n": 1},
    )
    with pytest.raises(ValueError, match="at least one chronon"):
        mt.chronon_glb()
    with pytest.raises(ValueError, match="at least one chronon"):
        mt.chronon_lub()


# -- Python API behaviours not directly represented in the R fixture --------


def test_indexing_returns_length_one_vectors():
    x = mt.yearmonth(["2024-01", "2024-02"])
    assert len(x[0]) == 1
    assert x[0].format() == ["2024 Jan"]
    assert x[-1].format() == ["2024 Feb"]


def test_scalar_access_preserves_metadata():
    x = mt.yearmonth(["2024-01", "2024-02"])
    scalar = x.scalar(0)
    assert isinstance(scalar, mt.TimeScalar)
    assert scalar.format() == "2024 Jan"
    assert scalar.chronon.name == "month"
    assert scalar.kind == "linear"
    assert scalar.discrete is True
    # Existing extraction behavior is intentionally unchanged.
    assert x[0].format() == ["2024 Jan"]

    cyclical = mt.month_of_year(["2024-01-15"]).scalar(0)
    assert cyclical.kind == "cyclical"
    assert cyclical.cycle is not None
    assert cyclical.cycle.name == "year"

    duration = mt.days(2).scalar(0)
    assert duration.kind == "duration"
    assert duration.format() == "2.0 days"


def test_missing_scalars_and_masks():
    x = mt.date(["2024-01-01", None])
    assert x.isna().tolist() == [False, True]
    assert x.notna().tolist() == [True, False]
    assert x.scalar(1).isna()
    assert x.scalar(1).format() == "NA"
    assert mt.TimeScalar.missing(x.scalar(0)).format() == "NA"
    assert x.scalar(0) == x.scalar(0)
    assert hash(x.scalar(0)) == hash(x.scalar(0))
    assert x.scalar(0) != x.scalar(1)


def test_mixed_scalar_take_and_shift():
    m = mt.mixtime(
        mt.yearmonth(["2024-01", "2024-02"]),
        mt.date(["2024-03-01", None]),
    )
    assert m.scalar(1).chronon.name == "month"
    assert m.scalar(2).chronon.name == "day"
    assert m.isna().tolist() == [False, False, False, True]
    assert (m == m).tolist() == [True, True, True, True]
    assert (m != m).tolist() == [False, False, False, False]

    taken = m.take([2, -1, 0], allow_fill=True)
    assert taken.format() == ["2024-03-01", "NA", "2024 Jan"]
    assert [part.chronon.name for part in taken.parts] == ["day", "month"]

    shifted = mt.mixtime(
        mt.yearmonth(["2024-01", "2024-02"]),
        mt.yearmonth(["2024-03"]),
    ) + mt.months(1)
    assert shifted.format() == ["2024 Feb", "2024 Mar", "2024 Apr"]


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


def test_non_utc_datetime_format_includes_abbreviation():
    x = mt.datetime("2024-01-15 12:34:56", tz="America/New_York")
    assert x.format() == ["2024-01-15 12:34:56 EST"]


def test_fractional_calendar_shifts_are_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="fractional month"):
        mt.date("2024-01-31") + mt.months(1.5)
    with pytest.raises(NotImplementedError, match="fractional month"):
        mt.datetime("2024-01-31 09:30:00", tz="America/New_York") + mt.quarters(0.5)


def test_offset_date_strings_convert_to_requested_timezone():
    assert mt.date("2024-01-15T23:00:00-05:00").format() == ["2024-01-16"]
    assert mt.date(
        "2024-01-15T23:00:00-05:00", tz="America/New_York"
    ).format() == ["2024-01-15"]
    assert mt.yearweek("2024-01-15T23:00:00-05:00").format() == ["2024 W03"]


METADATA_INPUTS = {
    "yearmonth": lambda: mt.yearmonth("2024-01"),
    "yearquarter": lambda: mt.yearquarter("2024-Q1"),
    "yearweek": lambda: mt.yearweek("2024-W03"),
    "date": lambda: mt.date("2024-01-15"),
    "datetime": lambda: mt.datetime("2024-01-15 12:00:00", tz="UTC"),
    "month_of_year": lambda: mt.month_of_year(dt.date(2024, 1, 15)),
    "day_of_week": lambda: mt.day_of_week(dt.date(2024, 1, 15)),
    "time_of_day": lambda: mt.time_of_day(
        dt.datetime(2024, 1, 15, 12, 34, 56, tzinfo=ZoneInfo("UTC"))
    ),
    "duration_months": lambda: mt.months(2),
    "mixed": lambda: mt.mixtime(
        mt.yearmonth("2024-01"), mt.date("2024-01-15"), mt.months(2)
    ),
}


def _unit_text(unit):
    return "NA" if unit is None else repr(unit)


@pytest.mark.parametrize("case", FIXTURES["metadata"], ids=lambda c: c["name"])
def test_metadata(case):
    x = METADATA_INPUTS[case["name"]]()
    chronon = mt.time_chronon(x)
    if isinstance(chronon, list):
        assert [_unit_text(unit) for unit in chronon] == _listify(
            case["chronon"]
        )
    else:
        assert [_unit_text(chronon)] == _listify(case["chronon"])

    cycle = mt.time_cycle(x)
    if isinstance(cycle, list):
        assert [_unit_text(unit) for unit in cycle] == _listify(case["cycle"])
    else:
        assert [_unit_text(cycle)] == _listify(case["cycle"])

    calendar = mt.time_calendar(x)
    assert not isinstance(calendar, list)
    # Python has one Gregorian calendar with a week unit rather than R's
    # separate cal_isoweek calendar.
    expected_calendar = (
        "gregorian" if case["calendar"] == "isoweek" else case["calendar"]
    )
    assert calendar.name == expected_calendar
    assert mt.is_mixtime(x) is (case["name"] == "mixed")


def test_metadata_accepts_python_temporal_values():
    assert mt.time_chronon(dt.date(2024, 1, 15)) == mt.cal_gregorian.day
    assert (
        mt.time_chronon(dt.datetime(2024, 1, 15))
        == mt.cal_gregorian.second
    )
    assert mt.time_cycle(dt.date(2024, 1, 15)) is None
    assert mt.time_calendar(dt.date(2024, 1, 15)) is mt.cal_gregorian
    unit = mt.cal_gregorian.month(3)
    assert mt.time_chronon(unit) == unit
    assert mt.time_cycle(unit) is None
    assert mt.time_calendar(unit) is mt.cal_gregorian
    assert mt.is_mixtime(unit) is False


TIMEZONE_INPUTS = {
    "naive_datetime": lambda: mt.datetime("2024-01-15 12:00:00"),
    "utc_datetime": lambda: mt.datetime("2024-01-15 12:00:00", tz="UTC"),
    "ny_winter_datetime": lambda: mt.datetime(
        "2024-01-15 12:00:00", tz="America/New_York"
    ),
    "ny_summer_datetime": lambda: mt.datetime(
        "2024-07-15 12:00:00", tz="America/New_York"
    ),
    "sydney_datetime": lambda: mt.datetime(
        "2024-01-15 12:00:00", tz="Australia/Sydney"
    ),
    "naive_date": lambda: mt.date("2024-01-15"),
    "ny_date": lambda: mt.date("2024-01-15", tz="America/New_York"),
    "ny_yearmonth": lambda: mt.yearmonth(
        "2024-01", tz="America/New_York"
    ),
    "ny_yearquarter": lambda: mt.yearquarter(
        "2024-Q1", tz="America/New_York"
    ),
    "ny_year": lambda: mt.year("2024", tz="America/New_York"),
    "ny_week": lambda: mt.yearweek(
        "2024-W03", tz="America/New_York"
    ),
    "ny_hour": lambda: mt.linear_time(
        1, mt.cal_gregorian.hour(1, tz="America/New_York")
    ),
    "mixed": lambda: mt.mixtime(
        mt.datetime("2024-01-15 12:00:00", tz="America/New_York"),
        mt.date("2024-01-15", tz="America/New_York"),
        mt.yearmonth("2024-01", tz="America/New_York"),
    ),
}


@pytest.mark.parametrize("case", FIXTURES["timezones"], ids=lambda c: c["name"])
def test_timezones(case):
    x = TIMEZONE_INPUTS[case["name"]]()
    assert mt.tz_name(x) == _listify(case["tz_name"])
    assert mt.tz_abbreviation(x) == _listify(case["abbreviation"])

    offset = mt.tz_offset(x)
    if isinstance(offset, mt.Mixtime):
        _assert_values(
            np.concatenate([part.values for part in offset.parts]),
            _listify(case["offset_values"]),
        )
    else:
        _assert_values(offset.values, _listify(case["offset_values"]))
        assert offset.chronon.name == case["offset_chronon"]["name"]
        assert float(offset.chronon.n) == float(case["offset_chronon"]["n"])
    _assert_formatted(offset.format(), case["offset_formatted"])


TRANSITION_INPUTS = {
    "ny_2024": lambda: (
        dt.datetime(2024, 1, 1, tzinfo=ZoneInfo("America/New_York")),
        dt.datetime(2024, 12, 31, tzinfo=ZoneInfo("America/New_York")),
    ),
    "london_2024": lambda: (
        dt.datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/London")),
        dt.datetime(2024, 12, 31, tzinfo=ZoneInfo("Europe/London")),
    ),
    "utc_2024": lambda: (
        dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc),
    ),
}


@pytest.mark.parametrize(
    "case", FIXTURES["transitions"], ids=lambda c: c["name"]
)
def test_tz_transitions(case):
    start, end = TRANSITION_INPUTS[case["name"]]()
    result = mt.tz_transitions(start, end)
    assert set(result) == {"time", "offset_before", "offset_after"}
    _assert_values(result["time"].values, _listify(case["time"]))
    _assert_values(
        result["offset_before"].values, _listify(case["offset_before"])
    )
    _assert_values(
        result["offset_after"].values, _listify(case["offset_after"])
    )
    assert result["time"].discrete is False
    assert result["time"].timezone == "UTC"
    assert result["offset_before"].chronon.name == "second"
    assert result["offset_after"].chronon.name == "second"


def test_timezone_helpers_accept_python_temporal_values():
    aware = dt.datetime(
        2024, 1, 15, 12, tzinfo=ZoneInfo("America/New_York")
    )
    assert mt.tz_name(aware) == ["America/New_York"]
    assert mt.tz_abbreviation(aware) == ["EST"]
    assert mt.tz_offset(aware).format() == ["-18000.0 seconds"]
    assert mt.tz_offset(dt.date(2024, 1, 15)).format() == ["0.0 days"]
    naive = dt.datetime(2024, 1, 15, 12)
    assert mt.tz_abbreviation(
        naive, tz="America/New_York"
    ) == ["EST"]
    assert mt.tz_offset(
        naive, tz="America/New_York"
    ).format() == ["-18000.0 seconds"]
    assert mt.tz_name(
        mt.cal_gregorian.hour(1, tz="America/New_York")
    ) == "America/New_York"


def test_tz_transitions_accept_time_vectors():
    start = mt.datetime("2024-01-01", tz="America/New_York")
    end = mt.datetime("2024-12-31", tz="America/New_York")
    result = mt.tz_transitions(start, end)
    assert result["time"].format() == [
        "2024-03-10 07:00:00 0.0%",
        "2024-11-03 06:00:00 0.0%",
    ]


def test_tz_transitions_accept_naive_values_with_explicit_tz():
    result = mt.tz_transitions(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 12, 31),
        tz="America/New_York",
    )
    assert result["time"].values.tolist() == [1710054000.0, 1730613600.0]


def test_pandas_extension_series_roundtrip():
    pd = pytest.importorskip("pandas")

    x = mt.yearmonth(["2024-01", "2024-02"])
    series = x.to_pandas_series(name="when")

    assert series.dtype == mt.MixtimeDtype()
    assert isinstance(series.array, mt.MixtimeArray)
    assert series.name == "when"
    assert series.iloc[0] == x.scalar(0)
    assert [value.format() for value in series] == ["2024 Jan", "2024 Feb"]
    assert series.isna().tolist() == [False, False]

    constructed = pd.Series(mt.MixtimeArray(x))
    assert constructed.dtype == mt.MixtimeDtype()
    assert [value.format() for value in constructed] == ["2024 Jan", "2024 Feb"]

    explicit = pd.Series([x.scalar(0), x.scalar(1)], dtype=mt.MixtimeDtype())
    assert explicit.dtype == mt.MixtimeDtype()
    assert explicit.iloc[1] == x.scalar(1)


def test_pandas_mixed_extension_dataframe():
    pd = pytest.importorskip("pandas")

    m = mt.mixtime(
        mt.yearmonth(["2024-01", "2024-02"]),
        mt.date(["2024-03-01", None]),
    )
    series = m.to_pandas_series()
    assert series.dtype == mt.MixtimeDtype()
    assert series.iloc[1].chronon.name == "month"
    assert series.iloc[2].chronon.name == "day"
    assert series.iloc[3].isna()

    frame = pd.DataFrame({"when": series, "value": [1, 2, 3, 4]})
    assert frame["when"].dtype == mt.MixtimeDtype()
    assert frame["when"].iloc[2].format() == "2024-03-01"
    assert frame["when"].isna().tolist() == [False, False, False, True]


def test_pandas_extension_take_concat_and_numpy():
    pd = pytest.importorskip("pandas")

    x = mt.date(["2024-01-01", None, "2024-01-02"])
    array = mt.MixtimeArray(x)
    taken = array.take([0, -1, 1], allow_fill=True)
    assert taken.isna().tolist() == [False, True, True]
    assert taken[0].format() == "2024-01-01"

    left = pd.Series(array)
    right = pd.Series(array)
    joined = pd.concat([left, right], ignore_index=True)
    assert joined.dtype == mt.MixtimeDtype()
    assert len(joined) == 6

    values = left.to_numpy()
    assert all(isinstance(value, mt.TimeScalar) for value in values)
    assert left.astype(object).iloc[0] == x.scalar(0)
    assert left.astype("mixtime").dtype == mt.MixtimeDtype()
    assert pd.array([x.scalar(0), x.scalar(2)], dtype="mixtime").dtype == mt.MixtimeDtype()

    filled = left.fillna(x.scalar(0))
    assert filled.isna().tolist() == [False, False, False]
    assert filled.iloc[1].format() == "2024-01-01"
    assert left.dropna().tolist() == [x.scalar(0), x.scalar(2)]
    assert left.where(left.notna(), x.scalar(0)).isna().tolist() == [
        False,
        False,
        False,
    ]


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


def test_pandas_input_coercion():
    pd = pytest.importorskip("pandas")

    stamps = pd.to_datetime(["2024-01-15", "2024-02-29"])
    assert mt.date(pd.Series(stamps)).format() == ["2024-01-15", "2024-02-29"]
    assert mt.date(pd.DatetimeIndex(stamps)).format() == [
        "2024-01-15",
        "2024-02-29",
    ]
    assert mt.yearmonth(
        pd.Series(pd.period_range("2024-01", periods=2, freq="M"))
    ).format() == ["2024 Jan", "2024 Feb"]
    assert mt.yearmonth(pd.Series([1, 2])).values.tolist() == [1, 2]
    assert mt.yearmonth(pd.Series([1.5, 2.5])).discrete is False
    assert mt.date(pd.Series(["2024-01-15", None])).format() == [
        "2024-01-15",
        "NA",
    ]
    assert mt.date(pd.Series([pd.Timestamp("2024-01-15"), pd.NaT])).format() == [
        "2024-01-15",
        "NA",
    ]
    aware = pd.Series(pd.to_datetime(["2024-01-15 12:00:00+00:00"]))
    assert mt.datetime(aware, tz="America/New_York").format() == [
        "2024-01-15 07:00:00 EST"
    ]
    assert mt.date(pd.NaT).format() == ["NA"]
    assert mt.date(np.datetime64("NaT", "ns")).format() == ["NA"]
    assert mt.date(
        np.array(["2024-01-15", "NaT"], dtype="datetime64[ns]")
    ).format() == ["2024-01-15", "NA"]


def test_discrete_shift_rejects_indeterminate_duration():
    quarters = mt.yearquarter(["2024-Q2", "2024-Q3"])
    with pytest.raises(ValueError, match="temporally indeterminate"):
        quarters + mt.months(1)
    assert (quarters + mt.months(3)).format() == ["2024 Q3", "2024 Q4"]

    dates = mt.date(["2024-01-15", "2024-02-29"])
    assert (dates + mt.months(1)).format() == ["2024-02-15", "2024-03-29"]
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


def test_continuous_calendar_shifts_and_unsupported_cross_category_shifts():
    continuous_date = mt.date(dt.date(2024, 1, 31), discrete=False)
    assert (continuous_date + mt.months(1)).format() == ["2024-02-29 0.0%"]
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
