# mixtime (Rust port)

Rust port of mixtime 0.4.0's Gregorian `chronon_divmod` methods, from
`R/cal_gregorian.R`. Built as a PyO3 extension by maturin; the Python wrapper
lives in `../python`.

Scope: `day -> month/quarter/year` and `month/year -> day` chronon splits.
Not ported: parsing/formatting, timezones, astronomical calendars, and the
remaining calendar systems.

The kernels take `ndarray` views. The PyO3 layer binds `PyReadonlyArray1<f64>`
and passes `as_array()` straight through, then fills two `PyArray1<f64>`
outputs in place, so NumPy input and output are both zero-copy.

## Upstream sync

The numbered procedure (install → replace upstream → regenerate fixtures →
rerun both suites → reconcile) lives in
[`../CONTRIBUTING.md#upstream-sync`](../CONTRIBUTING.md#upstream-sync).
