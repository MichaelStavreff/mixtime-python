#!/usr/bin/env python3
"""Compare R and Python->Rust per-call times on identical arange vectors.

Both sides run `microbenchmark`-style repeated timings on the same input,
`numpy.arange(n, dtype=float64)` / `as.numeric(seq_len(n) - 1)`:

  - r_small.csv:      R `microbenchmark`, microseconds
  - python_small.csv: Python->Rust `timeit`, microseconds

No `bench::mark` data is used here: R's GC variance made it unsuitable for
locating a crossover, and the requested comparison is against `microbenchmark`.
"""

from __future__ import annotations

import csv
import math


def load(path: str) -> dict[int, dict[str, float]]:
    with open(path, encoding="utf-8") as fh:
        return {int(row["size"]): row for row in csv.DictReader(fh)}


r = {
    n: {
        "min": float(row["min_us"]),
        "median": float(row["median_us"]),
        "max": float(row["max_us"]),
    }
    for n, row in load("/tmp/opencode/r_small.csv").items()
}
rust = load("/tmp/opencode/python_small.csv")

sizes = sorted(set(r) & set(rust))
print(
    f"{'size':>8} {'R_min':>10} {'R_med':>10} {'Rust_min':>10} {'Rust_med':>10} "
    f"{'win_min':>8} {'win_med':>8}"
)
for n in sizes:
    rr = r[n]
    pp = rust[n]
    print(
        f"{n:>8} {rr['min']:>10.2f} {rr['median']:>10.2f} "
        f"{float(pp['min_us']):>10.2f} {float(pp['median_us']):>10.2f} "
        f"{'rust' if float(pp['min_us']) < rr['min'] else 'R':>8} "
        f"{'rust' if float(pp['median_us']) < rr['median'] else 'R':>8}"
    )


def crossings(stat: str) -> list[tuple[int, int, float, str, str]]:
    winners = [
        "rust" if float(rust[n][f"{stat}_us"]) < r[n][stat] else "R"
        for n in sizes
    ]
    out = []
    for i in range(1, len(sizes)):
        if winners[i] == winners[i - 1]:
            continue
        n0, n1 = sizes[i - 1], sizes[i]
        f0 = math.log(r[n0][stat]) - math.log(float(rust[n0][f"{stat}_us"]))
        f1 = math.log(r[n1][stat]) - math.log(float(rust[n1][f"{stat}_us"]))
        t = f0 / (f0 - f1)
        crossing = math.exp(math.log(n0) + t * (math.log(n1) - math.log(n0)))
        out.append((n0, n1, crossing, winners[i - 1], winners[i]))
    return out


for stat in ("min", "median"):
    print(f"\nCrossings by {stat}:")
    found = crossings(stat)
    if not found:
        print("  none: Rust is faster at every measured size")
    for n0, n1, crossing, before, after in found:
        print(
            f"  between {n0} and {n1}: n ~= {crossing:.0f} "
            f"({before} faster -> {after} faster)"
        )
