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
  mixed = list(),
  predicates = list(),
  components = list(),
  compose = list(),
  circsum = list(),
  granules = list()
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

add_comparison <- function(name, value) {
  out$comparisons[[length(out$comparisons) + 1L]] <<- list(
    name = name,
    result = as.logical(value)
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
add_case("operations", "time_shift_months", time_shift(x, months(1)), "linear")
add_case("operations", "time_shift_days",
         time_shift(date(c("2024-01-15", "2024-02-29")), days(1)), "linear")
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

# -- mixed granularity -----------------------------------------------------
add_mixed(
  "month_quarter",
  c(yearmonth(c("2024-01", "2024-02")),
    yearquarter(c("2024-Q2", "2024-Q3")))
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

json <- toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = NA,
               na = "null", null = "null")
writeLines(json, out_file)
cat("wrote", out_file, "\n")
