# mixtime (Python port)

A Pythonic time-vector package built on the Rust port of `mixtime`'s Gregorian
chronon arithmetic. The Rust crate lives in `../rust` and is built with maturin;
this package provides the time model, parsing, formatting, timezones, and the
low-level NumPy kernel.

## Layers

- `mixtime.yearmonth`, `yearquarter`, `yearweek`, `date`, `datetime` create
  linear time vectors.
- `mixtime.month_of_year`, `day_of_year`, `day_of_month`, `day_of_week`,
  `week_of_year`, `time_of_day` create cyclical time vectors.
- `mixtime.years`, `quarters`, `months`, `weeks`, `days`, `hours`, `minutes`,
  `seconds`, `milliseconds` create durations.
- `mixtime.mixtime(...)` combines vectors of different granularities.
- `mixtime.time_floor`, `time_ceiling`, `time_round`, `seq`, and `time_shift`
  provide calendar-aware operations.
- `mixtime.time_is_linear`, `time_is_cyclical`, `time_is_duration`,
  `time_is_determinate_at`, and `time_is_complete_at` inspect vector type and
  granularity.
- `mixtime.day_to_month`, `day_to_quarter`, `day_to_year`, `month_to_day`,
  `year_to_day` expose the low-level Rust kernel.

Time points are stored as chronon counts since 1970-01-01. Durations store plain
counts of their unit. `values` is a NumPy array; `format()` returns a list of
strings. Constructors accept one-dimensional pandas `Series`/`Index` objects
and `to_pandas()` is available when pandas is installed.

## Example

```python
import mixtime as mt

x = mt.yearmonth(["2024-01", "2024-02"])
print(x.values)          # [648 649]
print(x.format())        # ['2024 Jan', '2024 Feb']
print((x + mt.months(1)).format())

when = mt.date(["2024-01-15", "2024-02-29"])
print(mt.month_of_year(when).format())  # ['Jan', 'Feb']
print(mt.day_of_week(when).format())    # ['Mon', 'Thu']
print(mt.time_is_determinate_at(when, "week"))  # [True True]
print(mt.time_is_complete_at(when, "year"))     # [False False]
```

## Divergences from R

These are intentional Pythonic choices, not accidental behavior changes. The
R-generated fixtures in `../fixtures/time_model.json` still pin the shared
Gregorian chronon values and default formatted output.

- **Discrete inference.** Time constructors infer discrete chronons from the
  input type: integer counts, strings, and temporal objects are discrete;
  floating-point counts are continuous. R makes the same integer/double
  distinction, but Python integer literals are not R's default doubles. Pass
  `discrete=True` or `discrete=False` to override the inference.
- **Superset parsing.** R's constructors apply one candidate format to a whole
  character vector. The Python constructors parse each element independently,
  so a vector can mix `"2024 Jan"` and `"Jan 2024"`. Python also accepts full
  date strings where R's `yearmonth()`/`yearquarter()` parsers reject them.
- **Format tokens.** Both R-style `{lin(year)}` / `{cyc(month,year)}` tokens and
  strftime-style `%Y-%m-%d` tokens are supported. R's `format()` treats `%`
  strings as literal text.
- **ISO week metadata.** `yearweek()` and `week_of_year()` reproduce R's week
  values and formatting, but the Python objects use the Gregorian calendar with
  a `week` chronon and a `year` cycle. R reports `cal_isoweek` and `isoyear`.
- **ISO week to date.** Week chronons are Monday-start blocks. `date(yearweek(...))`
  yields the Monday start, matching R's value, but Python formats it as an ISO
  date (`2024-01-15`) rather than R's ISO-week label (`2024-W03-Mon`).
- **Time-vector indexing.** `x[i]` returns a length-one time vector, not a
  Python scalar, mirroring R's length-one result. Use `.values[0]` for the
  scalar chronon count.
- **Mutable values.** `TimeVector.values` and `np.asarray(vector)` expose the
  underlying NumPy buffer. In-place edits mutate the vector, unlike R's
  copy-on-modify; copy first when sharing or modifying values.
- **Different-chronon subtraction.** Subtracting two discrete time vectors of
  different chronons raises, matching R's indeterminacy rule. Continuous
  differences use the finest common chronon (for example, month − day is a
  duration in days) and always return a continuous duration.
- **Discrete shifts.** Adding a duration that does not nest within the time
  chronon raises. For example, `yearquarter(...) + months(1)` is indeterminate,
  while `+ months(3)` is allowed.
- **Continuous cross-category shifts.** Continuous shifts within one category
  are supported: calendar + calendar (`yearmonth + months(1)`) and fixed +
  fixed (`date + days(1)`). Continuous shifts between calendar and fixed units
  (`date + months(1)`, `yearmonth + days(1)`) raise `NotImplementedError`
  because R clamps to calendar fields at the shifted date. Use discrete
  calendar chronons or convert explicitly.
- **Duration defaults.** Duration helpers are continuous by default, matching
  R's double-valued helpers (`years(1)` formats as `1.0 years`). Pass
  `discrete=True` for whole-granule durations (`years(1, discrete=True)`
  formats as `1 year`). Duration conversion between calendar and fixed units
  raises, as it does in R without a time context.
- **Sequences.** `seq()` rejects incompatible chronons instead of converting
  the endpoint. It supports `from`/`to` with `by` or `length_out`; R's
  `from` + `length_out` + `by` form is not implemented. Calendar steps on
  day/second time (for example, `seq(date, by="month")`) raise; use calendar
  chronons or explicit dates.
- **Rounding granules.** `time_floor(x, mt.year(1))` treats the vector's chronon
  as the granule. R's `year(1L)` is a linear time value and is easy to misuse as
  a granule; use `"year"`, `mt.cal_gregorian.year`, or `mt.years(1)` in Python.
- **Python date range.** Date formatting uses `datetime.date`, so the supported
  display range is 0001-01-01 through 9999-12-31. The Rust kernel remains usable
  outside that range.
- **Offset-only datetimes.** A datetime string with a UTC offset but no `tz=`
  argument is normalized to UTC for storage and display. Pass `tz=` to display
  it in a named zone.
- **pandas week output.** `yearweek(...).to_pandas()` returns a Monday–Sunday
  `PeriodIndex` with `freq="W-SUN"`, not an ISO week label.
- **pandas input.** Constructors accept one-dimensional `Series` and `Index`
  objects; `DataFrame` and other columnar inputs are not coerced. `pd.NA` and
  `pd.NaT` become `NaN`, so the usual missing-value rules apply.
- **Predicate missing values.** `time_is_determinate_at()` and
  `time_is_complete_at()` return object arrays with `None` for missing or
  infinite values, whereas R returns logical `NA`.
- **Completeness with mixed granularity.** `time_is_complete_at()` rejects a
  `Mixtime` with more than one part, matching R's current restriction; a
  single-part `Mixtime` is accepted.
- **Comparison operands.** Equality for linear time points uses R's half-open
  interval rule, so a discrete month is not equal to its first day. Ordering
  between different chronons is position-based and can differ from R's interval
  relation; compare after `astype()` for exact ordering. Bare chronon counts
  are not accepted as comparison operands.
- **Missing values.** Missing time values are represented by `NaN` in a float
  array and formatted as `"NA"`. R distinguishes typed `NA` variants.

## Upstream sync

The numbered procedure (install → replace upstream → regenerate fixtures →
rerun both suites → reconcile) lives in
[`../CONTRIBUTING.md#upstream-sync`](../CONTRIBUTING.md#upstream-sync).
