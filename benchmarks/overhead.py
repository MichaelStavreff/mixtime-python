#!/usr/bin/env python3
"""Python->Rust per-call overhead for mixtime.day_to_month.

The input is `numpy.arange(n, dtype=np.float64)`, which the R side reproduces
exactly as `as.numeric(seq_len(n) - 1)`, so both benchmarks see identical
vectors. The array is passed straight to the extension, which reads its buffer
in place and fills two freshly allocated NumPy output arrays, so no list
conversion happens in either direction.

Uses adaptive iteration counts: one warm-up, then calibrate `number` so each
repeat takes roughly 0.2 s, then 7 repeats. Writes
/tmp/opencode/python_small.csv.
"""

from __future__ import annotations

import statistics
import timeit
from importlib.metadata import version

import numpy as np

import mixtime

SIZES = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    1024,
    4096,
    16384,
    65536,
    100000,
    262144,
    524288,
    1000000,
    1048576,
    2097152,
]


def arange_vec(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64)


def bench(x: np.ndarray) -> tuple[float, float, float, float, int]:
    timer = timeit.Timer(lambda: mixtime.day_to_month(x))
    timer.timeit(number=1)  # warm-up

    # Calibrate the iteration count.
    probe = min(timer.repeat(repeat=3, number=1))
    number = max(1, int(0.2 / probe))
    samples = timer.repeat(repeat=7, number=number)
    per_call_us = sorted(t / number * 1e6 for t in samples)
    return (
        per_call_us[0],
        statistics.median(per_call_us),
        per_call_us[-1],
        statistics.stdev(per_call_us) if len(per_call_us) > 1 else 0.0,
        number,
    )


def main() -> None:
    print(f"Python: {__import__('sys').version.split()[0]}")
    print(f"numpy: {np.__version__}")
    print(f"mixtime: {version('mixtime')}")
    print()
    print("=== Python->Rust day_to_month on np.arange(n) (microseconds per call) ===")
    print(f"{'size':>8} {'min_us':>12} {'median_us':>12} {'max_us':>12} {'sd_us':>10} {'number':>10}")

    rows = []
    for n in SIZES:
        x = arange_vec(n)
        lo, med, hi, sd, number = bench(x)
        rows.append((n, lo, med, hi, sd, number))
        print(f"{n:>8} {lo:>12.4f} {med:>12.4f} {hi:>12.4f} {sd:>10.4f} {number:>10}")

    with open("/tmp/opencode/python_small.csv", "w", encoding="utf-8") as fh:
        fh.write("size,min_us,median_us,max_us,sd_us,number\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    print("\nWrote /tmp/opencode/python_small.csv")


if __name__ == "__main__":
    main()
