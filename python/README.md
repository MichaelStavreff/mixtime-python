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
- `TimeVector.scalar(i)` and `Mixtime.scalar(i)` return a `TimeScalar` with its
  chronon metadata. `x[i]` still returns a length-one vector.
- Optional pandas interop: `to_pandas_series()` and `MixtimeArray` expose time
  columns through the `"mixtime"` extension dtype, including Arrow/Parquet
  round-trips; pandas and pyarrow are not required by the core package.
- `mixtime.time_floor`, `time_ceiling`, `time_round`, `seq`, and `time_shift`
  provide calendar-aware operations, including DST-aware day/week shifts,
  clamped month/quarter/year shifts, and calendar-aware sequences on `date`
  and `datetime`.
- `mixtime.time_is_linear`, `time_is_cyclical`, `time_is_duration`,
  `time_is_determinate_at`, and `time_is_complete_at` inspect vector type and
  granularity.
- `mixtime.lin`, `cyc`, `time_components`, and `time_compose` extract and
  rebuild Gregorian linear/cyclical components.
- `mixtime.chronon_glb`, `chronon_lub`, and `chronon_common` select a common
  Gregorian chronon; `mixtime.circsum` sums circular windows.
- `mixtime.time_chronon`, `time_cycle`, `time_calendar`, and `is_mixtime`
  expose chronon, cycle, calendar, and container metadata.
- `mixtime.tz_name`, `tz_abbreviation`, `tz_offset`, and `tz_transitions`
  inspect timezones and daylight-saving transitions.
- `mixtime.day_to_month`, `day_to_quarter`, `day_to_year`, `month_to_day`,
  `year_to_day` expose the low-level Rust kernel.

Time points are stored as chronon counts since 1970-01-01. Durations store plain
counts of their unit. `values` is a NumPy array; `format()` returns a list of
strings. Constructors accept one-dimensional pandas `Series`/`Index` objects.
`to_pandas()` returns a pandas `PeriodIndex`/`DatetimeIndex` for linear time;
`to_pandas_series()` returns a Series using mixtime's optional extension dtype.

## Example

```python
import mixtime as mt

x = mt.yearmonth(["2024-01", "2024-02"])
print(x.values)          # [648 649]
print(x.format())        # ['2024 Jan', '2024 Feb']
print((x + mt.months(1)).format())

mixed = mt.mixtime(
    mt.yearmonth(["2024-02"]),
    mt.year(["2025"]),
)
earlier = mt.mixtime(
    mt.yearmonth(["2024-01"]),
    mt.year(["2024"]),
)
print((mixed - earlier).format())  # ['1.0 months', '1.0 years']

when = mt.date(["2024-01-15", "2024-02-29"])
print(mt.month_of_year(when).format())  # ['Jan', 'Feb']
print(mt.day_of_week(when).format())    # ['Mon', 'Thu']
print(mt.time_is_determinate_at(when, "week"))  # [True True]
print(mt.time_is_complete_at(when, "year"))     # [False False]

parts = mt.time_components(when, yr=mt.lin("year"), mth=mt.cyc("month", "year"))
print(parts["mth"].format())  # ['Jan', 'Feb']
print(mt.time_compose(mt.lin("year", 1980), mt.cyc("month", "year", 3)).format())
# ['1980 Mar']

print(mt.chronon_glb(mt.yearmonth("2024-01"), mt.date("2024-01-01")))  # 1 day
print(mt.chronon_lub(mt.yearmonth("2024-01"), mt.date("2024-01-01")))  # 1 month
print(mt.circsum([1, 2, 3, 4], 2, 2))  # [3. 7.]

when = mt.month_of_year("2024-03")
print(mt.time_chronon(when))  # 1 month
print(mt.time_cycle(when))    # 1 year
print(mt.time_calendar(when).name)  # gregorian

ny = mt.datetime("2024-07-15 12:00:00", tz="America/New_York")
print(mt.tz_name(ny))             # ['America/New_York']
print(mt.tz_abbreviation(ny))     # ['EDT']
print(mt.tz_offset(ny).format())  # ['-14400.0 seconds']
changes = mt.tz_transitions(
    mt.datetime("2024-01-01", tz="America/New_York"),
    mt.datetime("2024-12-31", tz="America/New_York"),
)
print(changes["time"].format())
# ['2024-03-10 07:00:00 0.0%', '2024-11-03 06:00:00 0.0%']
```

## Comparison and ordering

Time points compare as half-open intervals, matching R: a discrete value spans
its whole granule, while a continuous value is an instant. For two time points
`a` and `b`, with `start`/`end` measured in their finest shared chronon:

| Operator | Holds when |
| --- | --- |
| `a == b` | `start(a) == start(b)` and `end(a) == end(b)` |
| `a != b` | not `a == b` |
| `a < b` | `end(a) <= start(b)` |
| `a > b` | `start(a) >= end(b)` |
| `a <= b` | `end(a) <= end(b)` |
| `a >= b` | `start(a) >= start(b)` |

`<=`/`>=` are therefore not shorthand for `(< or ==)`/`(> or ==)` across
different granularities. `Mixtime` comparisons apply the same rules
element-wise; a length-one operand is broadcast and other length mismatches
raise instead of silently returning `False`.

```python
import mixtime as mt

mixed = mt.mixtime(
    mt.yearmonth(["2024-01", "2024-02"]),
    mt.year(["2024", "2025"]),
)
print(mixed.sort().format())   # ['2024 Jan', '2024 Feb', '2024', '2025']
print(mixed.min().format())    # 2024 Jan
print(mixed.max().format())    # 2025 Jan

months = mt.month_of_year(["2024-12-15", "2025-01-15", "2024-06-15"])
print(months.sort().format())  # ['Jan', 'Jun', 'Dec']
```

`TimeVector.sort()`/`argsort()` order one granularity directly; cyclical
vectors order by position within the cycle. `Mixtime.sort()`/`argsort()` use
R's mixed order proxy: every part is converted to the finest shared chronon,
discrete values are nudged by half a chronon so a span sorts at its midpoint,
and cyclical parts order by position within their shared cycle. `sort()` drops
missing values by default (as R does); pass `na_position="first"` or `"last"`
to keep them. `TimeVector.min()`/`max()` and `Mixtime.min()`/`max()` return a
`TimeScalar`; `Mixtime` converts to the common chronon first. They skip missing
values by default; pass `skipna=False` for R's `na.rm = FALSE` behaviour.

## pandas interop

pandas is optional. Install `mixtime[pandas]` and use `to_pandas_series()` when
a column should carry mixtime's `"mixtime"` extension dtype:

```python
import pandas as pd
import mixtime as mt

when = mt.mixtime(
    mt.yearmonth(["2024-01", "2024-02"]),
    mt.date(["2024-03-01", None]),
)
series = when.to_pandas_series(name="when")
frame = pd.DataFrame({"when": series, "value": [1, 2, 3, 4]})

print(series.dtype)       # mixtime
print(series.iloc[0])     # 2024 Jan (a TimeScalar)
print(series.iloc[0].chronon.name)  # month
print(frame["when"].isna().tolist())  # [False, False, False, True]
```

`pd.Series(mt.MixtimeArray(x))` is equivalent. The extension array supports
`isna`, `take`, `fillna`, `where`, `concat`, slicing, `shift()`/`diff()`,
`cummin()`/`cummax()`, `astype(object)`, `astype(str)`/`astype("string")`,
duration shifts
(`series + months(1)`), mixed time-point differences (`series - series`), and
`to_numpy()` (object array of `TimeScalar`). String conversion uses
`TimeScalar.format()`; pandas string dtypes keep missing values as `NA`, while
NumPy unicode arrays use the literal `"NA"` because they cannot represent
missing values. `to_csv()` and `to_json()` use the same formatted strings, with
missing values as an empty field or `null`. Columns sort chronologically even when granularities are
mixed, and comparisons, `between()`, `min()`/`max()`,
`idxmin()`/`idxmax()`, `groupby()`, `merge()`, `drop_duplicates()`, `unique()`,
and `value_counts()` follow the same interval rules as the core vectors.
`MixtimeArray.nlargest()`/`nsmallest()` are available directly; pandas'
`Series.nlargest()`/`nsmallest()` reject non-numeric extension dtypes before
dispatching, so use `series.sort_values(ascending=False).head(n)` instead.
`diff()` follows mixed subtraction: discrete values with different chronons
raise rather than guessing an anchor. `to_pandas()` remains available for a
`PeriodIndex`/`DatetimeIndex`; it is a separate API and is not changed by the
extension dtype.

Arrow and Parquet export require `pyarrow` (install `mixtime[pandas,arrow]` or
`pyarrow` separately). `MixtimeArray.__arrow_array__()` writes a struct with
the value plus chronon, kind, discreteness, cycle, calendar, and timezone
metadata, and `MixtimeDtype.__from_arrow__()` rebuilds the extension array.
`DataFrame.to_parquet()` and `pd.read_parquet()` therefore round-trip mixtime
columns without reducing them to formatted strings.

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
  Python scalar, mirroring R's length-one result. Use `.scalar(i)` for a
  `TimeScalar` with chronon metadata, or `.values[0]` for the scalar chronon
  count. pandas `series.iloc[i]` returns a `TimeScalar`. `.take(indices)` is
  the positional bulk form; with `allow_fill=True`, `-1` inserts a missing
  value. `Mixtime.take` borrows the surrounding part's metadata unless an
  explicit `TimeScalar` fill is given, and `TimeVector.take` requires an
  explicit fill to share the vector's chronon metadata.
- **Mutable values.** `TimeVector.values` and `np.asarray(vector)` expose the
  underlying NumPy buffer. In-place edits mutate the vector, unlike R's
  copy-on-modify; copy first when sharing or modifying values.
- **Different-chronon subtraction.** Subtracting two discrete time vectors of
  different chronons raises, matching R's indeterminacy rule. Continuous
  differences use the finest common chronon (for example, month − day is a
  duration in days) and always return a continuous duration.
- **Discrete shifts.** Adding a duration that does not nest within the time
  chronon raises. For example, `yearquarter(...) + months(1)` is indeterminate,
  while `+ months(3)` is allowed. `date` and `datetime` are exceptions for
  day-or-longer durations because they use calendar-field arithmetic.
- **Calendar shifts and DST.** `date`/`datetime` plus `days`, `weeks`,
  `months`, `quarters`, or `years` uses local wall-clock arithmetic for
  datetimes: New York `2024-03-09 12:00` plus one day stays at 12:00 after the
  DST change, while plus 24 hours is one hour later. Month, quarter, and year
  shifts clamp the day to the target month (`2024-01-31 + months(1)` becomes
  `2024-02-29`). A local time skipped by a DST gap clamps backward by the gap,
  and an ambiguous local time resolves to its first occurrence. These are R's
  defaults (`on_invalid = "clamp"`); the other `on_invalid` modes are not
  exposed. Shifts that require a fractional month (including fractional
  quarters) raise `NotImplementedError`; fractional day/week shifts are
  supported, and fractional years that resolve to whole months are allowed
  (`years(1.5)` is 18 months).
- **Continuous cross-category shifts.** Fixed-to-calendar shifts on
  `date`/`datetime` are supported with clamping. Shifts from a calendar chronon
  to a finer fixed unit (`yearmonth + days(1)`) still raise
  `NotImplementedError` because R's continuous result needs calendar
  decomposition that is not yet ported. `date + hours(1)` remains
  indeterminate.
- **Duration defaults.** Duration helpers are continuous by default, matching
  R's double-valued helpers (`years(1)` formats as `1.0 years`). Pass
  `discrete=True` for whole-granule durations (`years(1, discrete=True)`
  formats as `1 year`). Duration conversion between calendar and fixed units
  raises, as it does in R without a time context. Durations scale with `*` and
  `/` by numbers, and `duration / duration` returns a numeric ratio in the
  common chronon (raising across calendar/fixed units).
- **Raw numeric arithmetic.** `time ± number` advances raw chronon counts,
  matching R's implicit-chronon arithmetic, and `number - time` is supported.
  Python `int` operands map to R integers and keep the vector discrete; Python
  `float` operands map to R doubles and make it continuous.
- **Sequences.** `seq()` rejects incompatible chronons instead of converting
  the endpoint. It supports `from`/`to` with `by` or `length_out`; R's
  `from` + `length_out` + `by` form is not implemented. For `date` and
  `datetime`, day/week/month/quarter/year steps are supported as a duration,
  unit, or string. Day/week sequences use local anchor shifts and preserve
  wall-clock time across DST; month-like sequences use anchor shifts with
  month-end clamping. Month-like endpoint alignment follows R, so a clamped
  endpoint can be excluded when the end is earlier in its month/quarter/year
  than the start. R's clamping warning is not emitted. Counted unit strings
  accept fractional counts (`by="1.5 years"` is 18 months); R's parser
  truncates the numeric prefix to an integer.
- **Rounding granules.** `time_floor(x, mt.year(1))` treats the vector's chronon
  as the granule. R's `year(1L)` is a linear time value and is easy to misuse as
  a granule; use `"year"`, `mt.cal_gregorian.year`, or `mt.years(1)` in Python.
- **Python date range.** Date formatting uses `datetime.date`, so the supported
  display range is 0001-01-01 through 9999-12-31. The Rust kernel remains usable
  outside that range.
- **Offset-only datetimes.** A datetime string with a UTC offset but no `tz=`
  argument is normalized to UTC for storage and display. Pass `tz=` to display
  it in a named zone. Non-UTC named zones append their abbreviation to default
  `datetime` formatting, matching R; UTC and naive datetimes do not.
- **Metadata accessors.** R's `time_chronon()` and `time_cycle()` return
  duration vectors; this port returns the underlying `Unit` metadata (or a
  list of units for a `Mixtime`). `time_calendar()` returns a `Calendar`; a
  `Mixtime` with differing calendars returns a list. Python has one Gregorian
  calendar with a week unit, so `time_calendar(yearweek(...))` is Gregorian
  rather than R's separate `cal_isoweek`.
- **Timezone helpers.** `tz_name()` returns `None` for naive time (R returns
  `NA_character_`), while `tz_abbreviation()` returns `""` there, as R does.
  `tz_offset()` returns a `Duration` in the element's chronon and scales by R's
  nominal cardinalities (31 days per month, 365/4 days per quarter, 365 days
  per year), not by the actual local interval. `tz_transitions()` returns a
  dictionary of `time`, `offset_before`, and `offset_after` vectors instead of
  an R data frame; a naive start requires an explicit `tz=` because Python has
  no portable `Sys.timezone()` equivalent. Transition scanning uses one-hour
  steps, so two offset changes inside the same hour could be missed.
- **pandas week output.** `yearweek(...).to_pandas()` returns a Monday–Sunday
  `PeriodIndex` with `freq="W-SUN"`, not an ISO week label.
- **pandas input.** Constructors accept one-dimensional `Series` and `Index`
  objects; `DataFrame` and other columnar inputs are not coerced. `pd.NA` and
  `pd.NaT` become `NaN`, so the usual missing-value rules apply.
- **pandas extension dtype.** `to_pandas_series()` and
  `pd.Series(mt.MixtimeArray(x))` produce a `"mixtime"` column. Direct
  `pd.Series(x)` still treats the vector as its numeric chronon buffer, because
  the core vector is not a pandas `ExtensionArray` and pandas remains optional.
  Use the explicit series helper when column semantics are wanted; when passing
  `dtype=` directly, use `dtype=mt.MixtimeDtype()` so the optional module is
  loaded. `to_pandas()` continues to return a
  `PeriodIndex`/`DatetimeIndex` and is not replaced.
- **pandas `isna` on scalars.** pandas has no protocol for custom scalar
  missingness, so `pd.isna(series.iloc[i])` is `False` even when the
  `TimeScalar` is missing. Use `series.iloc[i].isna()` for one scalar or
  `series.isna()` for the column mask; the extension array reports missing
  values correctly.
- **pandas scalar equality.** `TimeScalar` equality is strict about chronon,
  cycle, calendar, timezone, and discreteness so pandas factorization and joins
  keep different chronons distinct, matching R's `unique()`. Scalar ordering
  and vector/pandas mask comparisons use R's half-open interval semantics
  instead, so different chronons can be ordered without being equal. Missing
  values compare equal to other missing values in equality masks, while
  ordering comparisons involving a missing value are `False`; R's three-valued
  `NA` propagation is not reproduced in boolean masks.
- **Predicate missing values.** `time_is_determinate_at()` and
  `time_is_complete_at()` return object arrays with `None` for missing or
  infinite values, whereas R returns logical `NA`.
- **Completeness with mixed granularity.** `time_is_complete_at()` rejects a
  `Mixtime` with more than one part, matching R's current restriction; a
  single-part `Mixtime` is accepted.
- **Chronon bounds.** `chronon_glb()`/`chronon_lub()` implement the Gregorian
  partial order directly. Week and calendar units have no common coarser
  chronon, so `chronon_lub()` raises `ValueError` for that combination, as R
  does. A supplied `ptype` is returned unchecked, matching R's current TODO.
  `circsum()` returns a NumPy array rather than R's numeric vector.
- **Component API.** R uses `time_components(x, yr = lin(year))` and formula
  composition such as `time_compose(lin(year) ~ 1980)`. Python uses
  `time_components(x, yr=lin("year"))`, returns a dictionary of columns rather
  than a data frame, and composes with `lin("year", 1980)` (or a
  `(spec, value)` pair). `cyc()` composition values are 1-indexed, matching R.
  `time_components()` columns store absolute chronon counts; formatting reduces
  them to the position within the cycle. Discrete components that cannot be
  resolved are `NA`; continuous components follow R's flooring when the target
  granule differs from the input chronon. `time_compose()` accepts one
  linear anchor and a connected cyclical chain, and rejects mixed-chronon
  `Mixtime` components.
- **Comparison operands.** Time points use R's half-open interval rule, so a
  discrete month is not equal to its first day and ordering is interval-based
  across chronons. `Mixtime` comparisons are element-wise, broadcast a
  length-one operand, and raise on other length mismatches. Mixed `Mixtime`
  ordering follows R's order proxy, including the half-chronon offset for
  discrete parts and position-within-cycle ordering for cyclical parts. Bare
  chronon counts and character operands are not accepted.
- **Missing comparisons.** Missing values compare equal to other missing values
  in `==` masks and unequal in `!=`; ordering operators involving a missing
  value are `False`. R returns a logical `NA` instead.
- **min/max missing values.** `Mixtime.min()`/`max()` default to
  `skipna=True`; R's `min()`/`max()` default to `na.rm = FALSE` and return
  `NA` when any value is missing. Pass `skipna=False` for R's default.
- **Missing values.** Missing time values are represented by `NaN` in a float
  array and formatted as `"NA"`; `TimeScalar` stores missing as `None`.
  pandas `pd.NA` and `NaT` normalize to missing. R distinguishes typed `NA`
  variants.

## Upstream sync

The numbered procedure (install → replace upstream → regenerate fixtures →
rerun both suites → reconcile) lives in
[`../CONTRIBUTING.md#upstream-sync`](../CONTRIBUTING.md#upstream-sync).
