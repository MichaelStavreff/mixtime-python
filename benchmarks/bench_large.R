#!/usr/bin/env Rscript
# Focused large-vector R benchmark using bench::mark.
# Writes /tmp/opencode/r_large.csv with min/median/max over the raw iterations.

suppressPackageStartupMessages(library(mixtime))
suppressPackageStartupMessages(library(bench))

lcg_vec <- function(n, modulus, offset) {
  if (n == 0) return(numeric(0))
  i <- 0:(n - 1)
  (((i * 1103515245 + 12345) %% 2^31) %% modulus) - offset
}

day1 <- cal_gregorian$day(1L)
month1 <- cal_gregorian$month(1L)

sizes <- c(100000L, 262144L, 524288L, 1048576L, 2097152L)
rows <- vector("list", length(sizes))
for (i in seq_along(sizes)) {
  n <- sizes[[i]]
  x <- lcg_vec(n, 200001, 100000)
  invisible(chronon_divmod(day1, month1, x))
  b <- bench::mark(
    day_to_month = chronon_divmod(day1, month1, x),
    iterations = 30,
    check = FALSE,
    memory = FALSE,
    filter_gc = TRUE,
    time_unit = "ms"
  )
  # bench_time stores seconds; convert to milliseconds for the CSV.
  times <- as.numeric(b$time[[1]]) * 1000
  rows[[i]] <- data.frame(
    size = n,
    min_ms = min(times),
    median_ms = median(times),
    max_ms = max(times),
    n_itr = length(times),
    n_gc = sum(b$n_gc)
  )
  cat("=== bench::mark size:", n, "===\n")
  print(rows[[i]], row.names = FALSE, digits = 6)
  cat("\n")
}

res <- do.call(rbind, rows)
write.csv(res, "/tmp/opencode/r_large.csv", row.names = FALSE)
cat("Wrote /tmp/opencode/r_large.csv\n")
