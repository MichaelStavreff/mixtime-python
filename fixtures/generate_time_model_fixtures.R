#!/usr/bin/env Rscript
# Generate the time-model fixtures from the upstream R implementation.
#
# Run with the R library containing mixtime 0.4.0, e.g.:
#   R_LIBS=/tmp/opencode/Rlib Rscript fixtures/generate_time_model_fixtures.R
#
# The expected values in fixtures/time_model.json are produced by mixtime
# itself. Do not hand-edit that file.

suppressPackageStartupMessages(library(mixtime))
suppressPackageStartupMessages(library(jsonlite))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}
out_file <- file.path(script_dir, "time_model.json")

out <- list(
  package_version = as.character(packageVersion("mixtime")),
  constructors = list(),
  parsing = list(),
  cyclical = list(),
  durations = list(),
  operations = list(),
  comparisons = list(),
  mixed_comparisons = list(),
  mixed = list(),
  mixed_arithmetic = list(),
  numeric_arithmetic = list(),
  ordering = list(),
  predicates = list(),
  components = list(),
  compose = list(),
  circsum = list(),
  granules = list(),
  metadata = list(),
  timezones = list(),
  transitions = list()
)

chronon_info <- function(x) {
  ch <- time_chronon(x)
  txt <- format(ch)[1L]
  list(
    name = sub("s$", "", sub("^[0-9.]+[[:space:]]+", "", txt)),
    n = as.numeric(sub("[[:space:]].*$", "", txt))
  )
}

cycle_info <- function(x) {
  cy <- tryCatch(time_cycle(x), error = function(e) NULL)
  if (is.null(cy) || length(cy) == 0L) {
    return(NULL)
  }
  txt <- format(cy)[1L]
  if (is.na(txt) || identical(txt, "NA")) {
    return(NULL)
  }
  list(
    name = sub("s$", "", sub("^[0-9.]+[[:space:]]+", "", txt)),
    n = as.numeric(sub("[[:space:]].*$", "", txt))
  )
}

calendar_name <- function(x) {
  cal <- tryCatch(time_calendar(x), error = function(e) NULL)
  if (is.null(cal)) {
    return(NULL)
  }
  if (inherits(cal, "cal_gregorian")) "gregorian" else class(cal)[1L]
}

add_case <- function(section, name, x, kind, discrete = NULL) {
  if (is.null(discrete)) {
    discrete <- is.integer(S7::S7_data(x@x[[1]]))
  }
  ch <- chronon_info(x)
  cy <- cycle_info(x)
  entry <- list(
    name = name,
    kind = kind,
    chronon = ch$name,
    n = ch$n,
    cycle = if (is.null(cy)) NULL else cy$name,
    cycle_n = if (is.null(cy)) NULL else cy$n,
    calendar = calendar_name(x),
    discrete = discrete,
    values = as.numeric(x),
    formatted = as.character(format(x))
  )
  out[[section]][[length(out[[section]]) + 1L]] <<- entry
}

add_numeric_case <- function(section, name, value) {
  out[[section]][[length(out[[section]]) + 1L]] <<- list(
    name = name,
    values = as.numeric(value),
    formatted = as.character(format(value))
  )
}

add_comparison <- function(name, value) {
  out$comparisons[[length(out$comparisons) + 1L]] <<- list(
    name = name,
    result = as.logical(value)
  )
}

add_mixed_comparison <- function(name, e1, e2) {
  ops <- c("==", "!=", "<", "<=", ">", ">=")
  results <- lapply(ops, function(op) {
    as.logical(do.call(op, list(e1, e2)))
  })
  names(results) <- ops
  out$mixed_comparisons[[length(out$mixed_comparisons) + 1L]] <<- list(
    name = name,
    left = as.character(format(e1)),
    right = as.character(format(e2)),
    results = results
  )
}

add_mixed_arithmetic <- function(name, e1, e2) {
  result <- e1 - e2
  parts <- lapply(result@x, function(part) {
    list(
      kind = sub("^mixtime::mt_", "", class(part)[1L]),
      chronon = chronon_info(part),
      discrete = is.integer(S7::S7_data(part)),
      values = as.numeric(part),
      formatted = as.character(format(part))
    )
  })
  out$mixed_arithmetic[[length(out$mixed_arithmetic) + 1L]] <<- list(
    name = name,
    left = as.character(format(e1)),
    right = as.character(format(e2)),
    formatted = as.character(format(result)),
    parts = parts
  )
}

add_ordering <- function(name, x) {
  sorted <- sort(x)
  decreasing <- sort(x, decreasing = TRUE)
  mn <- min(x)
  mx <- max(x)
  out$ordering[[length(out$ordering) + 1L]] <<- list(
    name = name,
    formatted = as.character(format(x)),
    order = as.integer(order(x)),
    sorted = as.character(format(sorted)),
    decreasing = as.character(format(decreasing)),
    min = as.character(format(mn)),
    max = as.character(format(mx)),
    min_chronon = chronon_info(mn),
    max_chronon = chronon_info(mx),
    min_discrete = is.integer(S7::S7_data(mn@x[[1L]])),
    max_discrete = is.integer(S7::S7_data(mx@x[[1L]]))
  )
}

add_mixed <- function(name, x) {
  out$mixed[[length(out$mixed) + 1L]] <<- list(
    name = name,
    values = as.numeric(x),
    formatted = as.character(format(x))
  )
}

add_predicate <- function(name, value) {
  out$predicates[[length(out$predicates) + 1L]] <<- list(
    name = name,
    result = unname(as.logical(value))
  )
}

component_info <- function(x) {
  list(
    values = as.numeric(x@x[[1L]]),
    formatted = as.character(format(x)),
    chronon = chronon_info(x),
    cycle = cycle_info(x)
  )
}

add_components <- function(name, x, columns) {
  out$components[[length(out$components) + 1L]] <<- list(
    name = name,
    columns = lapply(columns, component_info)
  )
}

add_compose <- function(name, x) {
  out$compose[[length(out$compose) + 1L]] <<- list(
    name = name,
    values = as.numeric(x@x[[1L]]),
    formatted = as.character(format(x)),
    chronon = chronon_info(x),
    cycle = cycle_info(x)
  )
}

add_circsum <- function(name, x, size, step) {
  out$circsum[[length(out$circsum) + 1L]] <<- list(
    name = name,
    x = as.numeric(x),
    size = size,
    step = step,
    result = as.numeric(circsum(x, size, step))
  )
}

unit_info <- function(x) {
  list(name = sub("mixtime::tu_", "", class(x)[1L]), n = as.numeric(x@n))
}

add_granule <- function(name, x) {
  glb <- tryCatch(unit_info(chronon_glb(x)), error = function(e) NULL)
  lub <- tryCatch(unit_info(chronon_lub(x)), error = function(e) NULL)
  out$granules[[length(out$granules) + 1L]] <<- list(
    name = name,
    glb = glb,
    lub = lub
  )
}

add_metadata <- function(name, x) {
  out$metadata[[length(out$metadata) + 1L]] <<- list(
    name = name,
    chronon = as.character(format(time_chronon(x))),
    cycle = as.character(format(time_cycle(x))),
    calendar = sub("^cal_", "", class(time_calendar(x))[1L])
  )
}

add_timezone <- function(name, x) {
  offset <- tz_offset(x)
  out$timezones[[length(out$timezones) + 1L]] <<- list(
    name = name,
    tz_name = unname(tz_name(x)),
    abbreviation = unname(tz_abbreviation(x)),
    offset_values = as.numeric(offset),
    offset_chronon = tryCatch(chronon_info(offset), error = function(e) NULL),
    offset_formatted = as.character(format(offset))
  )
}

add_transitions <- function(name, start, end) {
  transitions <- tz_transitions(start, end)
  # as.numeric() on an empty mixtime returns the S7 object rather than a
  # numeric vector, so make the empty case explicit for jsonlite.
  as_num <- function(x) if (length(x) == 0L) numeric(0) else as.numeric(x)
  out$transitions[[length(out$transitions) + 1L]] <<- list(
    name = name,
    time = as_num(transitions$time),
    offset_before = as_num(transitions$offset_before),
    offset_after = as_num(transitions$offset_after)
  )
}

# -- constructors ----------------------------------------------------------
add_case("constructors", "year_strings", year(c("2024", "2025")), "linear")
add_case("constructors", "yearquarter_strings",
         yearquarter(c("2024-Q1", "2024-Q2", "2023-Q4")), "linear")
add_case("constructors", "yearmonth_strings",
         yearmonth(c("2024-01", "2024-02", "2024-03")), "linear")
add_case("constructors", "yearweek_strings",
         yearweek(c("2024-W01", "2024-W03", "2024-W09")), "linear")
add_case("constructors", "date_strings",
         date(c("2024-01-15", "2024-02-29", "1970-01-01")), "linear")
add_case("constructors", "datetime_strings",
         datetime(c("2024-01-15 12:34:56", "2024-01-15 00:00:00")), "linear")
add_case("constructors", "yearmonth_numeric_double", yearmonth(648), "linear")
add_case("constructors", "yearmonth_numeric_integer", yearmonth(648L), "linear")

# Continuous constructors use Date/POSIXct inputs because R's character
# parsers for coarser chronons intentionally reject full timestamps.
add_case("constructors", "yearmonth_continuous",
         yearmonth(as.Date("2024-01-15"), discrete = FALSE), "linear",
         discrete = FALSE)
add_case("constructors", "yearquarter_continuous",
         yearquarter(as.Date("2024-02-15"), discrete = FALSE), "linear",
         discrete = FALSE)
add_case("constructors", "year_continuous",
         year(as.Date("2024-07-01"), discrete = FALSE), "linear",
         discrete = FALSE)
add_case("constructors", "date_continuous",
         date(as.POSIXct("2024-01-15 12:00:00", tz = "UTC"),
              discrete = FALSE), "linear", discrete = FALSE)
add_case("constructors", "datetime_continuous",
         datetime(as.POSIXct("2024-01-15 12:00:00.250", tz = "UTC"),
                  discrete = FALSE), "linear", discrete = FALSE)

# -- parsing ---------------------------------------------------------------
add_case("parsing", "yearmonth_2024_jan",
         yearmonth("2024 Jan"), "linear")
add_case("parsing", "yearmonth_jan_2024",
         yearmonth("Jan 2024"), "linear")
add_case("parsing", "yearmonth_slash",
         yearmonth("2024/1"), "linear")
add_case("parsing", "yearmonth_dash",
         yearmonth("2024-1"), "linear")
add_case("parsing", "yearquarter_2024_q1",
         yearquarter("2024 Q1"), "linear")
add_case("parsing", "yearquarter_compact",
         yearquarter("2024Q1"), "linear")
add_case("parsing", "yearquarter_q1_2024",
         yearquarter("Q1 2024"), "linear")
add_case("parsing", "yearquarter_slash",
         yearquarter("2024/1"), "linear")
add_case("parsing", "yearquarter_month",
         yearquarter("2024-01"), "linear")
add_case("parsing", "date_day_month_name",
         date("15 Jan 2024"), "linear")
add_case("parsing", "date_month_name_day",
         date("Jan 15, 2024"), "linear")
add_case("parsing", "date_slash",
         date("2024/01/15"), "linear")
add_case("parsing", "date_short_digits",
         date("2024-1-5"), "linear")
add_case("parsing", "datetime_iso_t",
         datetime("2024-01-15T12:34:56"), "linear")

# -- cyclical helpers ------------------------------------------------------
d <- as.Date(c("2024-01-15", "2024-02-29", "2024-12-31"))
add_case("cyclical", "month_of_year", month_of_year(d), "cyclical")
add_case("cyclical", "day_of_year", day_of_year(d), "cyclical")
add_case("cyclical", "day_of_month", day_of_month(d), "cyclical")
add_case("cyclical", "day_of_week", day_of_week(d), "cyclical")
add_case("cyclical", "week_of_year", week_of_year(d), "cyclical")
add_case("cyclical", "time_of_day",
         time_of_day(as.POSIXct(c("2024-01-15 12:34:56",
                                  "2024-01-15 00:00:00"), tz = "UTC")),
         "cyclical")
add_case("cyclical", "month_of_year_continuous",
         month_of_year(as.Date("2024-01-15"), discrete = FALSE),
         "cyclical", discrete = FALSE)
add_case("cyclical", "day_of_year_continuous",
         day_of_year(as.Date("2024-01-15"), discrete = FALSE),
         "cyclical", discrete = FALSE)
add_case("cyclical", "day_of_month_continuous",
         day_of_month(as.Date("2024-01-15"), discrete = FALSE),
         "cyclical", discrete = FALSE)
add_case("cyclical", "day_of_week_continuous",
         day_of_week(as.Date("2024-01-15"), discrete = FALSE),
         "cyclical", discrete = FALSE)
add_case("cyclical", "day_of_week_continuous_datetime",
         day_of_week(as.POSIXct("2024-01-15 12:00:00", tz = "UTC"),
                     discrete = FALSE),
         "cyclical", discrete = FALSE)
add_case("cyclical", "week_of_year_continuous",
         week_of_year(as.Date("2024-01-15"), discrete = FALSE),
         "cyclical", discrete = FALSE)
add_case("cyclical", "time_of_day_continuous",
         time_of_day(as.POSIXct("2024-01-15 12:34:56.250", tz = "UTC"),
                     discrete = FALSE),
         "cyclical", discrete = FALSE)

# -- durations -------------------------------------------------------------
add_case("durations", "years", years(c(1, 2, 0.5)), "duration")
add_case("durations", "quarters", quarters(c(1, 2, 0.5)), "duration")
add_case("durations", "months", months(c(1, 2, 0.5)), "duration")
add_case("durations", "weeks", weeks(c(1, 2, 0.5)), "duration")
add_case("durations", "days", days(c(1, 2, 0.5)), "duration")
add_case("durations", "hours", hours(c(1, 2, 0.5)), "duration")
add_case("durations", "minutes", minutes(c(1, 2, 0.5)), "duration")
add_case("durations", "seconds", seconds(c(1, 2, 0.5)), "duration")
add_case("durations", "milliseconds", milliseconds(c(1, 2, 0.5)), "duration")
add_case("durations", "years_fractional", years(c(1 / 3, 1.5, 2)), "duration")
add_case("durations", "months_fractional",
         months(0.967741935483871), "duration")
add_case("durations", "days_fractional", days(c(1 / 3, -0.5)), "duration")
add_case("durations", "seconds_fractional", seconds(0.123456789), "duration")
add_case("durations", "milliseconds_fractional", milliseconds(1.2345),
         "duration")
add_case("durations", "years_discrete", years(c(1L, 2L, 0L, -1L)), "duration")
add_case("durations", "days_discrete", days(c(1L, 2L, 0L, -1L)), "duration")
add_case("durations", "months_3_discrete",
         duration(c(1L, 2L), cal_gregorian$month(3L)), "duration")
add_case("durations", "months_3_continuous",
         duration(c(1.5, 2.5), cal_gregorian$month(3L)), "duration")

# -- operations ------------------------------------------------------------
x <- yearmonth(c("2024-01", "2024-02", "2024-07", "2024-12"))
add_case("operations", "floor_year", time_floor(x, "year"), "linear")
add_case("operations", "ceiling_year", time_ceiling(x, "year"), "linear")
add_case("operations", "round_year", time_round(x, "year"), "linear")

s <- seq(yearmonth("2024-01"), yearmonth("2024-06"), by = 2)
add_case("operations", "seq_by_two", s, "linear")

add_case("operations", "add_months", x + months(1), "linear")
add_case("operations", "subtract_months", x - months(1), "linear")
add_case("operations", "difference_months", x - yearmonth("2023-12"), "duration")
add_case("operations", "difference_cross",
         yearmonth(as.Date("2024-01-31"), discrete = FALSE) -
           date(as.Date("2024-01-01"), discrete = FALSE),
         "duration")
add_case("operations", "difference_quarter_month",
         yearquarter(as.Date("2024-01-31"), discrete = FALSE) -
           yearmonth(as.Date("2024-01-01"), discrete = FALSE),
         "duration")
add_case("operations", "difference_month_week",
         yearmonth(as.Date("2024-01-31"), discrete = FALSE) -
           yearweek(as.Date("2024-01-01"), discrete = FALSE),
         "duration")
add_case("operations", "add_continuous_months",
         yearmonth(as.Date("2024-01-31"), discrete = FALSE) + months(1),
         "linear", discrete = FALSE)
add_case("operations", "duration_common_weeks_days", weeks(1) + days(7),
         "duration")
add_case("operations", "duration_common_months_years", months(1) + years(1),
         "duration")
add_case("operations", "duration_discrete_add", years(1L) + months(1L),
         "duration")
add_case("operations", "duration_discrete_sub", years(1L) - months(1L),
         "duration")
add_case("operations", "duration_discrete_scalar_add", years(1L) + 1,
         "duration")
add_case("operations", "duration_discrete_scalar_sub", years(1L) - 0.5,
         "duration")
add_case("operations", "duration_discrete_scalar_mul", years(1L) * 2,
         "duration")
add_case("operations", "duration_discrete_scalar_div", years(1L) / 2,
         "duration")
add_case("operations", "number_minus_duration", 1 - days(c(3, 4)), "duration")
add_case("operations", "time_plus_number", date(c("2024-01-01", "2024-01-02")) + 1,
         "linear")
add_case("operations", "number_plus_time", 1 + date(c("2024-01-01", "2024-01-02")),
         "linear")
add_case("operations", "time_minus_number", date(c("2024-01-01", "2024-01-02")) - 1,
         "linear")
add_case("operations", "number_minus_time", 1 - date(c("2024-01-01", "2024-01-02")),
         "linear")
add_numeric_case("numeric_arithmetic", "duration_ratio_days",
                 days(c(3, 4)) / days(2))
add_numeric_case("numeric_arithmetic", "duration_ratio_hours",
                 hours(2) / minutes(30))
add_numeric_case("numeric_arithmetic", "duration_ratio_months",
                 months(3) / months(1))
add_case("operations", "time_shift_months", time_shift(x, months(1)), "linear")
add_case("operations", "time_shift_days",
         time_shift(date(c("2024-01-15", "2024-02-29")), days(1)), "linear")
add_case("operations", "time_shift_date_months_clamp",
         time_shift(date("2024-01-31"), months(1)), "linear")
add_case("operations", "time_shift_date_years_leap",
         time_shift(date("2024-02-29"), years(1)), "linear")
add_case("operations", "time_shift_datetime_days_dst_forward",
         time_shift(datetime("2024-03-09 12:00:00",
                             tz = "America/New_York"), days(1)),
         "linear")
add_case("operations", "time_shift_datetime_hours_dst_forward",
         time_shift(datetime("2024-03-09 12:00:00",
                             tz = "America/New_York"), hours(24)),
         "linear")
add_case("operations", "time_shift_datetime_days_dst_backward",
         time_shift(datetime("2024-11-02 12:00:00",
                             tz = "America/New_York"), days(1)),
         "linear")
add_case("operations", "time_shift_datetime_days_vector",
         time_shift(datetime(c("2024-03-09 12:00:00",
                               "2024-11-02 12:00:00"),
                             tz = "America/New_York"), days(1)),
         "linear")
add_case("operations", "time_shift_datetime_months_clamp_tz",
         time_shift(datetime("2021-01-31 09:30:00",
                             tz = "America/New_York"), months(1)),
         "linear")
add_case("operations", "time_shift_datetime_years_leap_tz",
         time_shift(datetime("2024-02-29 09:30:00",
                             tz = "America/New_York"), years(1)),
         "linear")
add_case("operations", "time_shift_datetime_quarters_tz",
         time_shift(datetime("2024-01-31 09:30:00",
                             tz = "America/New_York"), quarters(1)),
         "linear")
add_case("operations", "time_shift_melbourne_gap",
         time_shift(datetime("2025-10-04 02:30:00",
                             tz = "Australia/Melbourne"), days(1)),
         "linear")
add_case("operations", "time_shift_melbourne_ambiguous",
         time_shift(datetime("2025-04-05 02:30:00",
                             tz = "Australia/Melbourne"), days(1)),
         "linear")
add_case("operations", "date_timezone_utc",
         date(as.POSIXct("2024-01-15 23:00:00", tz = "America/New_York"),
              tz = "UTC"),
         "linear")
add_case("operations", "time_shift_continuous_months",
         time_shift(yearmonth(as.Date("2024-01-31"), discrete = FALSE),
                    months(1)),
         "linear", discrete = FALSE)
add_case("operations", "continuous_string_month",
         yearmonth("2024-01", discrete = FALSE), "linear", discrete = FALSE)
add_case("operations", "continuous_string_quarter",
         yearquarter("2024-Q1", discrete = FALSE), "linear", discrete = FALSE)
add_case("operations", "continuous_string_week",
         yearweek("2024-W03", discrete = FALSE), "linear", discrete = FALSE)
add_case("operations", "continuous_string_year",
         year("2024", discrete = FALSE), "linear", discrete = FALSE)
add_case("operations", "continuous_string_date",
         date("2024-01-15", discrete = FALSE), "linear", discrete = FALSE)
add_case("operations", "continuous_string_datetime",
         datetime("2024-01-15 12:34:56", discrete = FALSE), "linear",
         discrete = FALSE)
add_case("operations", "seq_length_out_fractional",
         seq(yearmonth("2024-01"), yearmonth("2024-12"), length.out = 5),
         "linear", discrete = FALSE)
add_case("operations", "seq_length_out_integral",
         seq(date("2024-01-01"), date("2024-01-10"), length.out = 4),
         "linear", discrete = TRUE)
add_case("operations", "seq_fractional_step_month",
         seq(yearmonth("2024-01"), yearmonth("2024-02"), by = 0.5), "linear")
add_case("operations", "seq_fractional_step_date",
         seq(date("2024-01-01"), date("2024-01-02"), by = 0.5), "linear")
add_case("operations", "seq_fractional_step_date_06",
         seq(date("2024-01-01"), date("2024-01-02"), by = 0.6), "linear")
add_case("operations", "seq_calendar_months_clamp",
         suppressWarnings(seq(date("2024-01-31"), date("2024-05-31"),
                             by = "month")), "linear")
add_case("operations", "seq_calendar_months_endpoint_alignment",
         suppressWarnings(seq(date("2024-01-31"), date("2024-04-30"),
                             by = "month")), "linear")
add_case("operations", "seq_calendar_months_reverse",
         suppressWarnings(seq(date("2024-05-31"), date("2024-01-31"),
                             by = "-1 month")), "linear")
add_case("operations", "seq_calendar_two_months",
         suppressWarnings(seq(date("2024-01-31"), date("2025-07-31"),
                             by = months(2))), "linear")
add_case("operations", "seq_calendar_fractional_years",
         suppressWarnings(seq(date("2024-01-31"), date("2026-01-31"),
                             by = years(1.5))), "linear")
add_case("operations", "seq_calendar_quarters",
         suppressWarnings(seq(date("2024-01-31"), date("2025-01-31"),
                             by = "quarter")), "linear")
add_case("operations", "seq_calendar_years_leap",
         suppressWarnings(seq(date("2024-02-29"), date("2026-02-28"),
                             by = "year")), "linear")
add_case("operations", "seq_calendar_days",
         suppressWarnings(seq(date("2024-01-01"), date("2024-01-10"),
                             by = "3 days")), "linear")
add_case("operations", "seq_calendar_weeks",
         suppressWarnings(seq(date("2024-01-03"), date("2024-01-24"),
                             by = "week")), "linear")
add_case("operations", "seq_calendar_datetime_dst",
         suppressWarnings(seq(datetime("2024-03-09 02:30:00",
                                       tz = "America/New_York"),
                             datetime("2024-03-11 02:30:00",
                                      tz = "America/New_York"),
                             by = "day")), "linear")
add_case("operations", "seq_calendar_datetime_month_dst",
         suppressWarnings(seq(datetime("2024-02-10 02:30:00",
                                       tz = "America/New_York"),
                             datetime("2024-05-10 02:30:00",
                                      tz = "America/New_York"),
                             by = "month")), "linear")
add_case("operations", "seq_calendar_datetime_months_clamp",
         suppressWarnings(seq(datetime("2024-01-31 09:30:00",
                                       tz = "America/New_York"),
                             datetime("2024-05-31 09:30:00",
                                      tz = "America/New_York"),
                             by = "month")), "linear")

# -- comparisons -----------------------------------------------------------
add_comparison("month_vs_day_start",
               yearmonth("2024-01") == date("2024-01-01"))
add_comparison("month_vs_day_end",
               yearmonth("2024-01") == date("2024-01-31"))
add_comparison("month_vs_month",
               yearmonth("2024-01") == yearmonth("2024-01"))
add_comparison("continuous_month_vs_day",
               yearmonth(as.Date("2024-01-15"), discrete = FALSE) ==
                 date(as.Date("2024-01-15"), discrete = FALSE))
add_comparison("span_vs_instant_same_value",
               yearmonth("2024-01") ==
                 yearmonth(as.Date("2024-01-01"), discrete = FALSE))
add_comparison("duration_discrete_vs_continuous",
               years(1L) == years(1))
add_comparison("month_vs_day_gt",
               yearmonth("2024-01") > date("2024-01-01"))
add_comparison("month_vs_day_lt",
               yearmonth("2024-01") < date("2024-01-01"))
add_comparison("month_vs_day_ge",
               yearmonth("2024-01") >= date("2024-01-01"))

# Mixed-vector comparisons. Length-1 operands recycle, as in R.
mixed_linear <- c(
  yearmonth(c("2024-01", "2024-02")),
  year(c(2024L, 2025L))
)
add_mixed_comparison("mixed_linear_self", mixed_linear, mixed_linear)
add_mixed_comparison(
  "mixed_linear_scalar",
  mixed_linear,
  yearmonth("2024-01")
)
add_mixed_comparison(
  "year_vs_month_interval",
  year(1980L),
  yearmonth(as.integer((1980 - 1970) * 12 + 11))
)
add_mixed_comparison(
  "mixed_instant_span",
  c(yearmonth("2024-01"), date("2024-01-15"), year(2024L)),
  date("2024-02-01")
)
add_mixed_comparison(
  "cyclical_positions",
  month_of_year(as.Date(c("2024-12-15", "2025-01-15", "2024-06-15"))),
  month_of_year(as.Date("2024-03-15"))
)
add_mixed_comparison(
  "cyclical_mixed_chronon",
  c(month_of_year(as.Date(c("2024-12-15", "2025-01-15"))),
    day_of_year(as.Date("2024-06-01"))),
  day_of_year(as.Date("2024-03-01"))
)

# -- mixed arithmetic ------------------------------------------------------
month_pair <- c(yearmonth("2024-01"), yearmonth("2024-02"))
add_mixed_arithmetic("same_month", month_pair, month_pair)
add_mixed_arithmetic("mixed_self", mixed_linear, mixed_linear)
add_mixed_arithmetic(
  "mixed_shifted",
  mixed_linear,
  c(yearmonth(c("2024-02", "2024-03")), year(c(2025L, 2026L)))
)
add_mixed_arithmetic(
  "date_month",
  c(date("2024-01-31"), yearmonth("2024-01")),
  c(date("2024-02-29"), yearmonth("2024-02"))
)
add_mixed_arithmetic(
  "recycle_scalar",
  yearmonth("2024-01"),
  month_pair
)
add_mixed_arithmetic(
  "continuous_mixed",
  c(yearmonth(as.Date("2024-01-15"), discrete = FALSE),
    date(as.Date("2024-01-15"), discrete = FALSE)),
  c(yearmonth(as.Date("2024-02-15"), discrete = FALSE),
    date(as.Date("2024-02-15"), discrete = FALSE))
)
add_mixed_arithmetic(
  "missing_month",
  yearmonth(c("2024-01", NA)),
  yearmonth(c("2024-02", "2024-03"))
)
add_mixed_arithmetic(
  "missing_mixed",
  c(yearmonth(c("2024-01", NA)), year(2024L)),
  c(yearmonth(c("2024-02", "2024-03")), year(2025L))
)
add_mixed_arithmetic(
  "missing_both",
  yearmonth(c("2024-01", NA)),
  yearmonth(c(NA, "2024-03"))
)
add_mixed_arithmetic(
  "cyclical_shifted",
  month_of_year(c(1L, 3L)),
  month_of_year(c(2L, 1L))
)

# -- mixed granularity -----------------------------------------------------
add_mixed(
  "month_quarter",
  c(yearmonth(c("2024-01", "2024-02")),
    yearquarter(c("2024-Q2", "2024-Q3")))
)

# -- ordering --------------------------------------------------------------
add_ordering("linear_mixed", mixed_linear)
add_ordering(
  "linear_mixed_instants",
  c(yearmonth("2024-01"), date("2024-01-15"), year(2024L),
    yearmonth("2024-02"))
)
add_ordering(
  "linear_continuous_mixed",
  c(yearmonth(as.Date("2024-01-15"), discrete = FALSE),
    date(as.Date("2024-01-20"), discrete = FALSE),
    year(as.Date("2024-06-01"), discrete = FALSE))
)
add_ordering(
  "cyclical_months",
  c(month_of_year(as.Date(c("2024-12-15", "2025-01-15", "2024-06-15"))),
    month_of_year(as.Date("2024-03-15")))
)
add_ordering(
  "cyclical_mixed_chronon",
  c(month_of_year(as.Date(c("2024-12-15", "2025-01-15"))),
    day_of_year(as.Date("2024-06-01")))
)

# -- predicates ------------------------------------------------------------
pd <- as.Date(c("2024-01-15", "2024-02-29"))
add_predicate("linear_yearmonth", time_is_linear(yearmonth(pd)))
add_predicate("linear_month_of_year", time_is_linear(month_of_year(pd)))
add_predicate("cyclical_yearmonth", time_is_cyclical(yearmonth(pd)))
add_predicate("cyclical_month_of_year", time_is_cyclical(month_of_year(pd)))
add_predicate("duration_months", time_is_duration(months(c(1, 2))))
add_predicate("duration_yearmonth", time_is_duration(yearmonth(pd)))

add_predicate("determinate_year_month",
              time_is_determinate_at(year(2020L), cal_gregorian$month(1L)))
add_predicate("determinate_continuous_year_month",
              time_is_determinate_at(year(2020), cal_gregorian$month(1L)))
add_predicate("determinate_month_year",
              time_is_determinate_at(yearmonth("2020-02"), cal_gregorian$year(1L)))
add_predicate("determinate_month_week",
              time_is_determinate_at(yearmonth("2020-01"), cal_isoweek$week(1L)))
add_predicate("determinate_day_month",
              time_is_determinate_at(date("2020-01-01"), cal_gregorian$month(1L)))
add_predicate("determinate_day_week",
              time_is_determinate_at(date("2020-01-01"), cal_isoweek$week(1L)))
add_predicate("determinate_week_month",
              time_is_determinate_at(yearweek("2020-W01"), cal_gregorian$month(1L)))
add_predicate("determinate_quarter_month",
              time_is_determinate_at(yearquarter("2020-Q1"), cal_gregorian$month(1L)))
add_predicate("determinate_datetime_hour",
              time_is_determinate_at(datetime("2020-01-01 00:00:00"),
                                     cal_gregorian$hour(1L)))

add_predicate("complete_year_partial",
              time_is_complete_at(
                yearmonth(as.Date("2020-01-01")) + 0:9,
                cal_gregorian$year(1L)
              ))
add_predicate("complete_year_full",
              time_is_complete_at(
                yearmonth(as.Date("2020-01-01")) + 0:14,
                cal_gregorian$year(1L)
              ))
add_predicate("complete_quarter_full",
              time_is_complete_at(
                yearquarter(c("2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4")),
                cal_gregorian$year(1L)
              ))
add_predicate("complete_month_missing",
              time_is_complete_at(
                yearmonth(c("2020-01", "2020-03")),
                cal_gregorian$year(1L)
              ))
add_predicate("complete_days_month",
              time_is_complete_at(
                date(as.Date("2020-01-01")) + 0:30,
                cal_gregorian$month(1L)
              ))
add_predicate("complete_days_week",
              time_is_complete_at(
                date(as.Date("2020-01-01")) + 0:6,
                cal_isoweek$week(1L)
              ))

# -- components and composition ------------------------------------------
tc <- time_components(
  yearmonth(as.Date("2026-02-14") + c(0, 40, 400)),
  yr = lin(year),
  mth = cyc(month, year)
)
add_components("yearmonth", yearmonth(as.Date("2026-02-14") + c(0, 40, 400)),
               list(yr = tc$yr, mth = tc$mth))

tc <- time_components(
  date(as.Date("2025-12-15") + 0:3),
  dow = cyc(day, cal_isoweek$week),
  doy = cyc(day, year)
)
add_components("date_week", date(as.Date("2025-12-15") + 0:3),
               list(dow = tc$dow, doy = tc$doy))

tc <- time_components(
  datetime("2024-03-15 12:34:56"),
  yr = lin(year),
  mth = cyc(month, year),
  dom = cyc(day, month),
  hod = cyc(hour, day),
  min = cyc(minute, hour),
  sec = cyc(second, minute),
  mq = cyc(month, quarter),
  qy = cyc(quarter, year)
)
add_components("datetime", datetime("2024-03-15 12:34:56"),
               list(yr = tc$yr, mth = tc$mth, dom = tc$dom,
                    hod = tc$hod, min = tc$min, sec = tc$sec,
                    mq = tc$mq, qy = tc$qy))

tc <- time_components(yearmonth("2024-01"), day = cyc(day, month))
add_components("indeterminate", yearmonth("2024-01"), list(day = tc$day))

tc <- time_components(
  yearmonth(as.Date("2024-01-15"), discrete = FALSE),
  yr = lin(year),
  mth = cyc(month, year),
  dom = cyc(day, month)
)
add_components("continuous", yearmonth(as.Date("2024-01-15"), discrete = FALSE),
               list(yr = tc$yr, mth = tc$mth, dom = tc$dom))

add_compose(
  "year_month_day",
  time_compose(lin(year) ~ 1980, cyc(month, year) ~ 3, cyc(day, month) ~ 15)
)
add_compose("year", time_compose(lin(year) ~ 1980))
add_compose("month", time_compose(cyc(month, year) ~ 3))
add_compose(
  "day_month",
  time_compose(cyc(day, month) ~ 15, cyc(month, year) ~ 3)
)
add_compose(
  "multi_unit",
  time_compose(lin(month(3L)) ~ 3, cyc(month(1L), month(3L)) ~ 3)
)
parts <- time_components(
  as.Date("2024-03-15"),
  yr = lin(year),
  mth = cyc(month, year),
  dom = cyc(day, month)
)
add_compose("tagged", time_compose(parts$yr, parts$mth, parts$dom))

# -- utility and chronon-bound cases -------------------------------------
add_circsum("basic", c(1, 2, 3, 4), 2, 2)
add_circsum("window3", c(1, 2, 3, 4, 5), 3, 3)
add_circsum("negative_step", c(1, 2, 3, 4, 5), 1, -1)
add_circsum("negative_size", c(1, 2, 3, 4, 5), -2, 1)
add_circsum("step_one", c(1, 2, 3, 4), 2, 1)
add_circsum("single_step_two", c(1, 2, 3, 4), 1, 2)
add_circsum("oversized", c(1, 2, 3, 4), 6, 1)
add_circsum("empty", numeric(0), 2, 1)
add_circsum("zero_size", c(1, 2, 3, 4), 0, 1)
add_circsum("zero_step", c(1, 2, 3, 4), 2, 0)

add_granule(
  "yearmonth_date",
  c(yearmonth(as.Date("2024-01-01")), date(as.Date("2024-01-01")))
)
add_granule(
  "week_month",
  c(yearweek(as.Date("2024-01-01")), yearmonth(as.Date("2024-01-01")))
)
add_granule(
  "week_hour",
  c(yearweek(as.Date("2024-01-01")), linear_time(1L, hour(1L)))
)
add_granule(
  "second_minute",
  c(linear_time(1L, second(1L)), linear_time(1L, minute(1L)))
)
add_granule("year_quarter", c(year(1L), yearquarter(1L)))
add_granule(
  "month_2_3",
  c(linear_time(1L, month(2L)), linear_time(1L, month(3L)))
)
add_granule(
  "year_2_month",
  c(linear_time(1L, year(2L)), linear_time(1L, month(1L)))
)
add_granule(
  "week_day",
  c(yearweek(as.Date("2024-01-01")), date(as.Date("2024-01-01")))
)
add_granule(
  "year_week",
  c(year(1L), yearweek(as.Date("2024-01-01")))
)

# -- metadata accessors ----------------------------------------------------
add_metadata("yearmonth", yearmonth("2024-01"))
add_metadata("yearquarter", yearquarter("2024-Q1"))
add_metadata("yearweek", yearweek("2024-W03"))
add_metadata("date", date("2024-01-15"))
add_metadata("datetime", datetime("2024-01-15 12:00:00", tz = "UTC"))
add_metadata("month_of_year", month_of_year(as.Date("2024-01-15")))
add_metadata("day_of_week", day_of_week(as.Date("2024-01-15")))
add_metadata("time_of_day",
             time_of_day(as.POSIXct("2024-01-15 12:34:56", tz = "UTC")))
add_metadata("duration_months", months(2))
add_metadata(
  "mixed",
  c(yearmonth("2024-01"), date("2024-01-15"), months(2))
)

# -- timezone helpers ------------------------------------------------------
add_timezone("naive_datetime", datetime("2024-01-15 12:00:00"))
add_timezone("utc_datetime", datetime("2024-01-15 12:00:00", tz = "UTC"))
add_timezone(
  "ny_winter_datetime",
  datetime("2024-01-15 12:00:00", tz = "America/New_York")
)
add_timezone(
  "ny_summer_datetime",
  datetime("2024-07-15 12:00:00", tz = "America/New_York")
)
add_timezone(
  "sydney_datetime",
  datetime("2024-01-15 12:00:00", tz = "Australia/Sydney")
)
add_timezone("naive_date", date("2024-01-15"))
add_timezone(
  "ny_date",
  date("2024-01-15", tz = "America/New_York")
)
add_timezone(
  "ny_yearmonth",
  yearmonth("2024-01", tz = "America/New_York")
)
add_timezone(
  "ny_yearquarter",
  yearquarter("2024-Q1", tz = "America/New_York")
)
add_timezone("ny_year", year("2024", tz = "America/New_York"))
add_timezone(
  "ny_week",
  yearweek("2024-W03", tz = "America/New_York")
)
add_timezone(
  "ny_hour",
  linear_time(1L, cal_gregorian$hour(1L, tz = "America/New_York"))
)
add_timezone(
  "mixed",
  c(
    datetime("2024-01-15 12:00:00", tz = "America/New_York"),
    date("2024-01-15", tz = "America/New_York"),
    yearmonth("2024-01", tz = "America/New_York")
  )
)

add_transitions(
  "ny_2024",
  as.POSIXct("2024-01-01", tz = "America/New_York"),
  as.POSIXct("2024-12-31", tz = "America/New_York")
)
add_transitions(
  "london_2024",
  as.POSIXct("2024-01-01", tz = "Europe/London"),
  as.POSIXct("2024-12-31", tz = "Europe/London")
)
add_transitions(
  "utc_2024",
  as.POSIXct("2024-01-01", tz = "UTC"),
  as.POSIXct("2024-12-31", tz = "UTC")
)

json <- toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = NA,
               na = "null", null = "null")
writeLines(json, out_file)
cat("wrote", out_file, "\n")
