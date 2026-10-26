#!/usr/bin/env Rscript
# R side of the mixtime chronon_divmod benchmark.
#
# Uses the same deterministic LCG as the Rust criterion bench:
#   v = ((i * 1103515245 + 12345) mod 2^31) mod modulus - offset
#
# Run:
#   R_LIBS=/tmp/opencode/Rlib Rscript benchmarks/bench.R
#
# Outputs the microbenchmark tables and writes /tmp/opencode/r_small.csv for
# the Python/Rust crossover calculation.

suppressPackageStartupMessages(library(mixtime))
suppressPackageStartupMessages(library(microbenchmark))

cat("R:", R.version.string, "\n")
cat("mixtime:", as.character(packageVersion("mixtime")), "\n")
cat("platform:", R.version$platform, "\n\n")

lcg_vec <- function(n, modulus, offset) {
  if (n == 0) {
    return(numeric(0))
  }
  i <- 0:(n - 1)
  (((i * 1103515245 + 12345) %% 2^31) %% modulus) - offset
}

# Precompute granules: the Rust side receives precomputed from_n/to_n, so the
# R side should not include granule construction in the timed kernel.
day1 <- cal_gregorian$day(1L)
day2 <- cal_gregorian$day(2L)
month1 <- cal_gregorian$month(1L)
month3 <- cal_gregorian$month(3L)
year1 <- cal_gregorian$year(1L)

cat("First LCG values (days, months, years):\n")
print(lcg_vec(5, 200001, 100000))
print(lcg_vec(5, 10001, 5000))
print(lcg_vec(5, 1001, 500))
cat("\n")

sizes <- c(1000L, 100000L, 1000000L)
for (n in sizes) {
  days <- lcg_vec(n, 200001, 100000)
  months <- lcg_vec(n, 10001, 5000)
  years <- lcg_vec(n, 1001, 500)

  # Warm-up: one full evaluation of every expression before timing.
  invisible(chronon_divmod(day1, month1, days))
  invisible(chronon_divmod(day1, year1, days))
  invisible(chronon_divmod(month1, day1, months))
  invisible(chronon_divmod(year1, day1, years))

  times <- if (n >= 1000000L) 10L else if (n >= 100000L) 30L else 100L
  cat("=== kernel size:", n, "| microbenchmark times:", times, "===\n")
  bench <- microbenchmark(
    day_to_month = chronon_divmod(day1, month1, days),
    day_to_year = chronon_divmod(day1, year1, days),
    month_to_day = chronon_divmod(month1, day1, months),
    year_to_day = chronon_divmod(year1, day1, years),
    times = times,
    unit = "ms"
  )
  print(bench, signif = 5)
  cat("\n")
}

# --- Small-size end-to-end comparison (day_to_month only) -------------------
# The Python side (`overhead.py`) passes `numpy.arange(n, dtype=float64)`.
# `as.numeric(seq_len(n) - 1)` produces exactly the same vector, so both
# benchmarks operate on identical inputs.
arange_vec <- function(n) {
  if (n == 0) {
    return(numeric(0))
  }
  as.numeric(seq_len(n) - 1)
}

small_sizes <- c(
  0L, 1L, 2L, 4L, 8L, 16L, 32L, 64L, 128L, 256L, 512L, 1000L, 1024L,
  4096L, 16384L, 65536L, 100000L, 262144L, 524288L, 1000000L, 1048576L,
  2097152L
)

rows <- vector("list", length(small_sizes))
for (i in seq_along(small_sizes)) {
  n <- small_sizes[[i]]
  x <- arange_vec(n)
  times <- if (n <= 16L) 10000L else if (n <= 1024L) 1000L else if (n <= 65536L) 100L else 10L
  invisible(chronon_divmod(day1, month1, x))
  b <- microbenchmark(
    day_to_month = chronon_divmod(day1, month1, x),
    times = times,
    unit = "us"
  )
  s <- summary(b)
  rows[[i]] <- data.frame(
    size = n,
    min_us = s$min,
    median_us = s$median,
    max_us = s$max,
    neval = s$neval
  )
}

res <- do.call(rbind, rows)
cat("=== end-to-end comparison: R day_to_month on seq_len(n)-1 (microseconds per call) ===\n")
print(res, row.names = FALSE, digits = 5)
write.csv(res, "/tmp/opencode/r_small.csv", row.names = FALSE)
cat("\nWrote /tmp/opencode/r_small.csv\n")
