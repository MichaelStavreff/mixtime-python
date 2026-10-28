# mixtime port — decision record

**Date:** 2026-10-23  
**Updated:** 2026-10-28 (numeric arithmetic and pandas protocol gaps)
**Pinned upstream:** CRAN `mixtime` 0.4.0 (`upstream/mixtime`)  
**Ports:** Rust core + PyO3/Python wrapper  
**Status:** **Niche MVP selected.** The kernel-only scope was correct as a
performance exercise but insufficient as a product. The port now includes a
Pythonic Gregorian/ISO-week time model on top of the Rust kernel.

## 2026-10-28 update: numeric arithmetic and remaining pandas protocol gaps

Time points now support R's implicit-chronon arithmetic: `time ± number` and
`number - time` operate on raw chronon counts. Python `int` operands follow R
integer semantics and preserve discreteness; Python `float` operands follow R
double semantics and make the result continuous. Durations divide by numbers,
and `duration / duration` returns a numeric ratio in the common chronon,
raising across calendar/fixed units just as R does without a time context.
`TimeVector.take()` joins the existing `Mixtime.take()` for positional
selection with optional `-1` filling.

The optional pandas array also implements cumulative `min`/`max`, formatted
CSV/JSON export, and metadata-preserving `shift()`/`diff()`. `TimeVector.take`
keeps an explicit fill value's chronon metadata, while `Mixtime.take` borrows
the surrounding part's metadata for inserted missing values. Empty time
constructors default to discrete, matching R's integer-typed empty vectors.
`Series.interpolate()` stays unsupported because interpolating across chronons
has no well-defined R equivalent; `Series.quantile()` is likewise unsupported.
`TimeVector.diff(lag, differences)` matches
R's `diff.mt_time` for single-chronon vectors, and `Mixtime` subtraction is
symmetric: `TimeVector - Mixtime` and `TimeScalar - MixtimeArray` mirror their
forward forms. Vector and `Mixtime` equality now raises `TypeError` for
unsupported operands instead of returning scalar `False`, and treats `None`,
`NaN`, and `pd.NA` as element-wise missing comparisons. `value_counts()`
honours `dropna` for mixtime missing values, which the generic object-array
fallback could not detect. `TimeVector.unique()`/`duplicated()` cover
single-chronon deduplication; mixed `Mixtime` deduplication stays out because
R's vecvec implementation drops parts. Empty `Mixtime` reductions now raise a
consistent `ValueError` instead of returning a missing scalar when the
container has empty parts.

## 2026-10-27 update: mixed comparison, ordering, and pandas operations

Phase 1 adds R-compatible comparison and ordering for mixed chronons plus the
pandas operations that depend on them. Time points compare as half-open
intervals, matching `compare.R`: a discrete value spans its granule and a
continuous value is an instant. `Mixtime` comparisons are element-wise,
broadcast a length-one operand, and raise on other length mismatches instead of
silently returning `False`.

Mixed ordering follows R's `vec_proxy_order`: convert every part to the finest
shared chronon, add half a chronon to discrete parts so a span sorts at its
midpoint, and order cyclical parts by position within their shared cycle.
`Mixtime.sort`/`argsort`/`min`/`max` expose this. `TimeVector.sort`/`argsort`/
`min`/`max` provide the same operations for one granularity; cyclical vectors
order by cycle position, while reductions use the absolute chronon value as
`Mixtime` does. `min`/`max` convert to the common chronon and return a
`TimeScalar`; R returns a length-one mixtime with the same chronon.

pandas columns now sort, filter, use `between`, reduce with `min`/`max`, and
participate in `groupby`, `merge`, `concat`, `drop_duplicates`, `unique`, and
`value_counts` with the same interval rules. `TimeScalar` equality stays strict
about metadata so factorization and joins keep different chronons distinct.
Missing values compare equal in `==` masks; ordering comparisons involving a
missing value are `False`. This is a documented divergence from R's
three-valued `NA`. pandas string conversion now uses `TimeScalar.format()`,
including `astype(str)`/`astype("string")`, and pandas columns support
duration shifts (`series + months(1)`). Broader vctrs parity remains deferred.

## 2026-10-27 update: mixed time-point subtraction

Mixed time-point differences are elementwise. `Mixtime - Mixtime` returns a
`Mixtime` whose parts are continuous `Duration` vectors: each pair uses its
finest common chronon, a length-one operand recycles, and discrete pairs with
different chronons raise. Missing results keep the pair's common chronon, so a
missing month paired with a month stays a missing month duration. The optional
pandas extension exposes the same operation as `series - series`.

## 2026-10-27 update: pandas columns and scalars

The core `TimeVector`/`Mixtime` API remains primary. `TimeScalar` and
`.scalar(i)` provide explicit scalar access without changing the existing
length-one `x[i]` behavior. An optional pandas `ExtensionArray`/`ExtensionDtype`
(`MixtimeArray`, `MixtimeDtype`, `to_pandas_series()`) makes mixed-granularity
values usable as Series/DataFrame columns. pandas stays an optional extra; the
core package does not import it. Direct `pd.Series(vector)` is not the
extension-dtype path because the core vector is not a pandas array.
Arrow/Parquet serialization is optional as well (`mixtime[arrow]`); it writes a
pyarrow struct carrying the value and chronon metadata, and pandas restores the
`"mixtime"` extension dtype on read instead of formatting values as strings.
pandas `shift()`/`diff()` preserve the neighbouring part's chronon metadata for
newly introduced missing values, so a shifted month column stays a month
column.

## 2026-10-27 update: DST-aware calendar shifts

`date` and `datetime` shifts by day-or-longer durations now use calendar-field
arithmetic. For timezone-aware `datetime`, day and week shifts preserve the
local wall-clock reading across DST changes, while sub-day durations add
elapsed time. Month, quarter, and year shifts clamp the day to the target
month (`2024-01-31 + months(1)` becomes `2024-02-29`). A local time skipped by
a DST gap clamps backward by the gap, and an ambiguous local time resolves to
its first occurrence. This matches R's default `on_invalid = "clamp"`; the
alternative resolutions are not exposed. Shifts that require a fractional
month (including fractional quarters) raise `NotImplementedError`, while
fractional day/week shifts are supported and fractional years that resolve to
whole months are allowed.

## 2026-10-27 update: calendar-aware sequences

`seq()` now accepts day, week, month, quarter, and year steps for `date` and
`datetime`, supplied as a duration, unit, or string. Counted strings such as
`"1 month"` and `"3 days"` are supported. Day/week sequences use local anchor
shifts, so they preserve wall-clock time across DST changes. Month-like
sequences use anchor shifts with month-end clamping and reproduce R's endpoint
alignment, including R's rule that can exclude a clamped endpoint when the end
is earlier in its month/quarter/year than the start. R's clamping warning is
not emitted. Counted strings honor fractional counts (`"1.5 years"` is 18
months); R's parser truncates the numeric prefix to an integer.

## 2026-10-26 update: why kernel-only was not enough

The kernel-only port reproduced the four Gregorian `chronon_divmod` splits and
was faster than R at every measured size. That is table stakes, not a reason to
use a separate package: users can already do fast date arithmetic with NumPy,
pandas, or the standard library.

The niche value of mixtime is the combination of:

- mixed granularities in one vector (`yearmonth` next to `yearquarter`),
- cyclical components (`month_of_year`, `day_of_week`, ...),
- calendar-aware durations and arithmetic,
- parsing and formatting of temporal values,
- timezone-aware civil time, and
- interop with the Python data stack.

The selected niche MVP implements those capabilities for the Gregorian calendar
and ISO weeks, while keeping the Rust kernel for chronon arithmetic. The Python
layer owns classes, metadata, parsing, formatting, timezones, pandas interop,
type/determinacy/completeness predicates, Gregorian component extraction /
composition, common-chronon selection, metadata accessors, and timezone
name/abbreviation/offset helpers with transition discovery. Rust remains the
arithmetic kernel. Durations follow R's double-valued helpers by default
(continuous), with `discrete=True` for whole granules. Time-point
differences use the finest common chronon. Date/datetime day-or-longer shifts
use calendar-field arithmetic, including DST handling and month/year clamping;
shifts from a calendar chronon to a finer fixed unit still raise rather than
approximating R's continuous decomposition.

Deferred:

- Symmetry454, civil/lunar/solar/astronomical calendars,
- the full R format-token and labelled-parsing language,
- the complete vctrs/S7 coercion surface (a Gregorian `time_components()` /
  `time_compose()` subset is implemented),
- ggplot2 and tsibble integration,
- complete mixed-vector arithmetic beyond duration shifts and time-point
  differences; full non-pandas columnar coercion and other mixed-vector
  operators remain partial,
- shifts from calendar chronons to finer fixed units (for example,
  `yearmonth + days(1)`),
- shifts requiring a fractional month and R's alternative `on_invalid` DST
  resolutions,
- R's `seq(from, length_out=, by=)` form.

This update supersedes the "Option 1 selected" status below. The option analysis
is retained as the historical record of the kernel-only decision.

## Kernel layer

The four Gregorian `chronon_divmod` splits from `R/cal_gregorian.R`:

- `day -> month` (including `to_n = 3`, i.e. quarters)
- `day -> year`
- `month -> day` (including multi-month `from_n`)
- `year -> day` (including multi-year `from_n` and `to_n` scaling)

The Rust kernels operate on `ndarray` views of `f64` and reproduce R's
floor-division, floor-modulo, epoch (1970-01-01), leap-year, and month-window
semantics. The Python package accepts one-dimensional `float64` NumPy arrays
and returns two `float64` NumPy arrays: the input buffer is read in place, and
the kernels fill freshly allocated NumPy output buffers directly, so there is
no `Vec` or list intermediate in either direction. Parsing, formatting,
timezones, astronomical calendars, other calendar systems, cardinality
algebra, and BC/AD labels are excluded from the Rust kernel. The Python time
model now covers Gregorian parsing, formatting, and timezones; the generic
`chronon_divmod` dispatch in `R/00_chronon_divmod.R` is still not ported.

## Equivalence evidence

- **Kernel fixtures:** `fixtures/fixtures.json` has 6 cases generated by real
  R 0.4.0 runs. Rust unit tests cover the same values; the kernel pytest suite
  covers the 6 golden cases, 10 replayed R cross-check cases, plus
  fixture-shape, output-type, and list-rejection tests.
- **Randomized R cross-check:** `fixtures/crosscheck.json` has 10 cases × 120 values
  (100 integer + 20 fractional samples per unit), including the original four
  splits plus `day(2)`, `month(2)`, `month(3)`, and `to_n = 2` variants. Rust
  and pytest each replay all 1,200 R outputs with zero mismatches.
- **Property test:** four `proptest` properties × 480 cases over the original
  ranges plus explicit month-boundary, quarter, multi-granule, leap-year, and
  fractional edge values. They compare against an independent `chrono` oracle,
  not the port's own arithmetic. There is also an error-path test and an LCG pin
  proving the benchmark vectors match R. `cargo test` is 14/14.
- **Upstream sync:** GitHub `0.4.0.9000` was installed and run through the full
  loop. `cases` and `inputs` were identical to 0.4.0; only `package_version`
  changed. The sync suite passed at the time. The current suite includes 14 Rust
  tests and the expanded Python time-model tests; see `COVERAGE.md`.

## Performance envelope

Benchmarks were rerun after both directions of the Python boundary became
zero-copy. Input is a `PyReadonlyArray1<f64>` read in place; output is two
`PyArray1<f64>` buffers filled by the kernel, with no `Vec` or list
intermediate. Both sides use the identical vector
`np.arange(n, dtype=float64)` / `as.numeric(seq_len(n) - 1)`, and R is timed
with `microbenchmark`; the raw tables are in the accompanying reply.

- **End-to-end `day_to_month`, median (Python→Rust vs R):** 1k 117.87 µs vs
  242.64 µs (2.1× faster); 100k 14.161 ms vs 19.865 ms (1.4×); 1M 145.206 ms
  vs 316.738 ms (2.2×). By min: 115.54 vs 223.37 µs; 14.103 vs 19.001 ms;
  143.891 vs 299.651 ms.
- **Crossover:** none. Rust is faster at every measured size from 0 to
  2,097,152 elements, by both min and median. The list boundary crossed at
  ~1,288 (min) / ~2,326 (median); zero-copy output removes the remaining
  conversion entirely.
- **Boundary:** an empty call costs 0.73 µs (it allocates two zero-length
  NumPy arrays); R's empty call costs ~44 µs. There is no per-element
  conversion on either side.
- **Pure Rust kernel (criterion, median):** `day_to_month` is 0.1346 /
  14.1511 / 150.8496 ms at 1k / 100k / 1M. The other splits scale the same
  way: `day_to_year` 0.0873 / 8.7696 / 98.1068, `month_to_day` 0.0367 /
  4.0702 / 51.5983, `year_to_day` 0.0242 / 2.5778 / 35.6178. The boundary no
  longer adds a per-element cost, so end-to-end is kernel-bound rather than
  boundary-bound.
- **C++ is not a valid comparison.** The C++ slice is string formatting,
  tzdb, and astronomy — a different workload. Benchmarking it against the
  vectorized split kernel would be meaningless, so it was not done.

## Sync cost

The full loop from a clean `env -i` shell took **24.41 s**: install + replace
9.57 s, regenerate fixtures 1.15 s, Rust + Python suites 13.61 s, reconcile
0.08 s. It needs an isolated `R_LIBS` (`/path/to/Rlib`), `just` on `PATH`,
`HOME` for cargo/uv caches and R's user library, plus R 4.6.1, Rust 1.98.1,
Python 3.14.7, uv 0.13.4, and a C++20 compiler. A fresh machine needs the same toolchain; CI can skip R unless it
regenerates fixtures.

## Options considered (historical, kernel-only decision)

**1. Kernel-only product — selected.** The vectorized split is the whole
product; the excluded surface is not needed. Supporting evidence: the kernel is correct,
both boundary directions are zero-copy, the API is small, the sync loop is
cheap, and the port is faster than R at every measured size from 0 to
2,097,152 elements (1.4×–2.2× at 100k/1M, much more at small sizes). The
option is selected on performance grounds. Against: users who need
parsing/formatting/timezones get nothing. Distinguishing evidence: a call-site
audit showing users only call splits.

**2. Full-surface port.** The value is in parsing, formatting, timezones, and
astronomy; the split kernel is table stakes. Supporting evidence: those are
most of `cal_gregorian.R` and most real workflows need them. Against: a much
larger correctness surface, tz/astronomy risk, and a larger sync burden.
Distinguishing evidence: issue/feature-request data, an end-to-end workflow
benchmark, and a count of which R functions callers actually use.

**3. Hybrid.** Keep the Rust kernel and bind the excluded surface to R or to
existing Python libraries (zoneinfo, astropy, etc.). Supporting evidence: low
implementation effort and reuse of mature tz/astronomy code. Against: two
runtimes, deployment complexity, and an R dependency if R is the backend.
Distinguishing evidence: deployment constraints, acceptable latency, and
whether R is already present in the target environment.

**4. Defer/stop.** Kernel-only is not enough to justify a port; wait for a
concrete workload. Supporting evidence: no user data has been collected, and
the excluded surface may still be required. Against: the port is already
correct, uniformly faster than R at every measured size, and the sync cost is
only ~24 s. Distinguishing evidence: a named user or workload that needs the
splits, or a decision that the excluded surface is out of scope permanently.

## Open validation items (historical)

- **Call-site audit:** do callers use only the four splits, or do they
  parse/format strings, cross timezones, or use astronomical calendars?
- **API surface:** callers get NumPy arrays in and out; whether downstream
  code needs Python scalars/lists is the remaining integration question.
- **End-to-end benchmarks:** kernel throughput is not the same as a real
  workflow; the current benchmark is a single split, not a full pipeline.
- **Maintenance budget:** a 24.41 s sync is cheap; a full-surface port is not.
- **Correctness risk:** arithmetic is testable against R; timezone and
  astronomy behavior is much harder.

At the time of the kernel-only decision, Option 1 was selected: the port was
correct against the R original, both boundary directions were zero-copy, and it
was faster than R at every measured size. The 2026-10-26 update supersedes that
status with the niche MVP, because the call-site/product question was answered
in favor of the higher-level time model.
