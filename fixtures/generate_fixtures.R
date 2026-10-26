#!/usr/bin/env Rscript
# Regenerate the golden fixtures for the mixtime Rust/Python port.
#
# Requires the CRAN package `mixtime` 0.4.0 to be installed, e.g.:
#   R_LIBS=/tmp/opencode/Rlib Rscript fixtures/generate_fixtures.R
#
# Output: fixtures/fixtures.json and crosscheck.json (written
# next to this script). crosscheck.json replays the randomized R cross-check.

suppressPackageStartupMessages(library(mixtime))
suppressPackageStartupMessages(library(jsonlite))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}
out_file <- file.path(script_dir, "fixtures.json")

# Hardcoded sample: chronon offsets relative to the Gregorian epoch 1970-01-01.
# Includes pre-epoch values, leap days, century boundaries, and fractions.
day_offsets <- c(-1, 0, 1, 59, 60, 90, 365, 366, 19723, 19782, 20700, 31.25)
month_offsets <- c(-1, 0, 1, 12, 13, 600, 660.5)
year_offsets <- c(-1, 0, 1, 54, 55, 56.5)

# Each case records the R call's inputs and the exact `div`/`mod` output.
case <- function(name, from, to, input, result) {
  list(
    name = name,
    from = list(unit = from[[1]], n = as.numeric(from[[2]])),
    to = list(unit = to[[1]], n = as.numeric(to[[2]])),
    input = I(as.numeric(input)),
    expected = list(
      div = I(as.numeric(result$div)),
      mod = I(as.numeric(result$mod))
    )
  )
}

cases <- list(
  case(
    "day_to_month",
    c("day", 1L), c("month", 1L), day_offsets,
    with(cal_gregorian, chronon_divmod(day(1L), month(1L), day_offsets))
  ),
  case(
    "day_to_quarter",
    c("day", 1L), c("month", 3L), day_offsets,
    with(cal_gregorian, chronon_divmod(day(1L), month(3L), day_offsets))
  ),
  case(
    "day_to_year",
    c("day", 1L), c("year", 1L), day_offsets,
    with(cal_gregorian, chronon_divmod(day(1L), year(1L), day_offsets))
  ),
  case(
    "month_to_day",
    c("month", 1L), c("day", 1L), month_offsets,
    with(cal_gregorian, chronon_divmod(month(1L), day(1L), month_offsets))
  ),
  case(
    "bimonth_to_day",
    c("month", 2L), c("day", 1L), month_offsets,
    with(cal_gregorian, chronon_divmod(month(2L), day(1L), month_offsets))
  ),
  case(
    "year_to_day",
    c("year", 1L), c("day", 1L), year_offsets,
    with(cal_gregorian, chronon_divmod(year(1L), day(1L), year_offsets))
  )
)

fixtures <- list(
  package = "mixtime",
  package_version = as.character(packageVersion("mixtime")),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  calendar = "cal_gregorian",
  epoch = "1970-01-01",
  cases = cases
)

write_json(
  fixtures,
  path = out_file,
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE
)

cat("Wrote", out_file, "with", length(cases), "cases\n")

# --- Randomized cross-check -------------------------------------------------
# Same domain as the original 480-value check: 100 integer samples plus 20
# fractional samples per unit, over the original ranges. Extra multi-granule
# cases (2-day, 2-month, 3-month, 2-day target) exercise granule scaling.
set.seed(42)
cross_days <- c(sample(-100000:100000, 100), runif(20, -1000, 1000))
cross_months <- c(sample(-5000:5000, 100), runif(20, -100, 100))
cross_years <- c(sample(-500:500, 100), runif(20, -10, 10))

cross_case <- function(name, from, to, from_n, to_n, input, result) {
  list(
    name = name,
    from = from,
    to = to,
    from_n = as.numeric(from_n),
    to_n = as.numeric(to_n),
    input = I(as.numeric(input)),
    div = I(as.numeric(result$div)),
    mod = I(as.numeric(result$mod))
  )
}

cross_cases <- list(
  cross_case(
    "day_to_month", "day", "month", 1L, 1L, cross_days,
    with(cal_gregorian, chronon_divmod(day(1L), month(1L), cross_days))
  ),
  cross_case(
    "day_to_quarter", "day", "month", 1L, 3L, cross_days,
    with(cal_gregorian, chronon_divmod(day(1L), month(3L), cross_days))
  ),
  cross_case(
    "day_to_year", "day", "year", 1L, 1L, cross_days,
    with(cal_gregorian, chronon_divmod(day(1L), year(1L), cross_days))
  ),
  cross_case(
    "twoday_to_month", "day", "month", 2L, 1L, cross_days,
    with(cal_gregorian, chronon_divmod(day(2L), month(1L), cross_days))
  ),
  cross_case(
    "twoday_to_year", "day", "year", 2L, 1L, cross_days,
    with(cal_gregorian, chronon_divmod(day(2L), year(1L), cross_days))
  ),
  cross_case(
    "month_to_day", "month", "day", 1L, 1L, cross_months,
    with(cal_gregorian, chronon_divmod(month(1L), day(1L), cross_months))
  ),
  cross_case(
    "bimonth_to_day", "month", "day", 2L, 1L, cross_months,
    with(cal_gregorian, chronon_divmod(month(2L), day(1L), cross_months))
  ),
  cross_case(
    "month_to_2day", "month", "day", 1L, 2L, cross_months,
    with(cal_gregorian, chronon_divmod(month(1L), day(2L), cross_months))
  ),
  cross_case(
    "year_to_day", "year", "day", 1L, 1L, cross_years,
    with(cal_gregorian, chronon_divmod(year(1L), day(1L), cross_years))
  ),
  cross_case(
    "year_to_2day", "year", "day", 1L, 2L, cross_years,
    with(cal_gregorian, chronon_divmod(year(1L), day(2L), cross_years))
  )
)

crosscheck <- list(
  package = "mixtime",
  package_version = as.character(packageVersion("mixtime")),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  seed = 42L,
  inputs = list(
    days = I(cross_days),
    months = I(cross_months),
    years = I(cross_years)
  ),
  cases = cross_cases
)

cross_file <- file.path(script_dir, "crosscheck.json")
write_json(
  crosscheck,
  path = cross_file,
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE
)

cat(
  "Wrote", cross_file, "with", length(cross_cases), "cases x",
  length(cross_days), "days,", length(cross_months), "months,",
  length(cross_years), "years\n"
)
