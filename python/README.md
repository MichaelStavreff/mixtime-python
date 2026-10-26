# mixtime (Python port)

Python bindings for the Rust port of `mixtime`'s Gregorian `chronon_divmod`
methods. The Rust crate lives in `../rust` and is built with maturin; this
package re-exports it with type hints.

Scope: `day -> month/quarter/year` and `month/year -> day` chronon splits,
matching mixtime 0.4.0 exactly (see `../fixtures/fixtures.json`).

The API takes one-dimensional `float64` NumPy arrays and returns two `float64`
NumPy arrays. The Rust extension reads the input buffer in place and fills the
output buffers directly (no `Vec` or list intermediate). A Python list is
rejected rather than converted.

## Upstream sync

The numbered procedure (install → replace upstream → regenerate fixtures →
rerun both suites → reconcile) lives in
[`../README.md#upstream-sync`](../README.md#upstream-sync).
