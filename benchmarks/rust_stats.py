#!/usr/bin/env python3
"""Print median/mean/std-dev from criterion's estimates.json files."""

from __future__ import annotations

import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "rust")
pattern = os.path.join(ROOT, "target", "criterion", "mixtime_kernel", "*", "*", "new", "estimates.json")

rows = []
for path in sorted(glob.glob(pattern)):
    parts = path.split(os.sep)
    size = int(parts[-3])
    name = parts[-4]
    with open(path, encoding="utf-8") as fh:
        est = json.load(fh)
    median_ns = est["median"]["point_estimate"]
    mean_ns = est["mean"]["point_estimate"]
    sd_ns = est["std_dev"]["point_estimate"]
    throughput = size / (median_ns / 1e9) / 1e6  # Melem/s
    rows.append((name, size, median_ns / 1e6, mean_ns / 1e6, sd_ns / 1e6, throughput))

print(f"{'benchmark':>14} {'size':>8} {'median_ms':>10} {'mean_ms':>10} {'sd_ms':>9} {'Melem/s':>9}")
for name, size, med, mean, sd, thr in rows:
    print(f"{name:>14} {size:>8} {med:>10.4f} {mean:>10.4f} {sd:>9.4f} {thr:>9.3f}")
