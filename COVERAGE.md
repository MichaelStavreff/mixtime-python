# Coverage map: `R/cal_gregorian.R` → port tests

Source: `upstream/mixtime/R/cal_gregorian.R` at mixtime **0.4.0**.
Tests: `cargo test` (14 tests) and `uv run pytest` (19 tests).

"Covered" means a test asserts the branch's observable result. The ported
surface is the four `chronon_divmod` splits; everything else in the file is
listed separately as excluded. No coverage percentage is claimed.

## Ported surface

| R function / branch | Covering test(s) | Covered |
|---|---|---|
| `is_leap_year`: divisible by 4, not 100 | `tests::leap_years` (2024), `prop_day_to_month` / `prop_month_to_day` edge 2024-02-29, `r_crosscheck_fixture_matches` | Yes |
| `is_leap_year`: divisible by 400 | `tests::leap_years` (2000), `prop_day_to_month` edge 2000-02-29, `prop_month_to_day` edge month 361 (2000-02) | Yes |
| `is_leap_year`: not leap (not /4, or /100 not /400) | `tests::leap_years` (1900, 2023, 1970), `prop_month_to_day` edge months 637 (2023-02), -839 (1900-02) | Yes |
| `is_leap_year`: `floor()` of fractional year | `prop_year_to_day` edge years 54.5 (2024), 30.5 (2000), 29.5 (1999), -1.5 (1968), -0.5 (1969) | Yes |
| `is_leap_year`: negative year | — | **No** (year domain is -500..500, i.e. 1470..2470; no negative years) |
| `month_start_days`: positive month index | `tests::day_to_month_matches_r`, `tests::day_to_quarter_matches_r`, `r_crosscheck_fixture_matches`, `prop_day_to_month_matches_oracle` | Yes |
| `month_start_days`: negative month index | `tests::day_to_month_matches_r` (x=-1), `r_crosscheck_fixture_matches` (negative days), `prop_day_to_month_matches_oracle` | Yes |
| `month_start_days`: `month > 2` leap adjustment | `tests::day_to_month_matches_r` (19723, 19782, 20700), `prop_day_to_month` edges 2024-02-29/03-01 | Yes |
| `month_start_days`: January/February (`month <= 2`) | `tests::day_to_month_matches_r` (0, 1, 31.25, 59), `prop_day_to_month_matches_oracle` | Yes |
| `month_start_days`: year < 0 | — | **No** (same domain limit) |
| `chronon_divmod(day, month)`: `x_scale = from@n` | `twoday_to_month` in `r_crosscheck_fixture_matches`, `prop_day_to_month` `from_n=2` | Yes |
| `chronon_divmod(day, month)`: `z >= 0` era branch | all day fixtures and `prop_day_to_month` / `prop_day_to_year` | Yes |
| `chronon_divmod(day, month)`: `z < 0` era branch | — | **No** (needs day < -719468; original domain is -100000..100000) |
| `chronon_divmod(day, month)`: `mp >= 10` month adjustment | `tests::day_to_month_matches_r` (365, 366), `r_crosscheck_fixture_matches`, `prop_day_to_month` | Yes |
| `chronon_divmod(day, month)`: `month <= 2` year adjustment | `tests::day_to_month_matches_r` (0, 59), `prop_day_to_month` | Yes |
| `chronon_divmod(day, month)`: `n > 0` target branch | `tests::day_to_month_matches_r` (n=1), `tests::day_to_quarter_matches_r` (n=3), `prop_day_to_month` (n=1,3) | Yes |
| `chronon_divmod(day, month)`: `n < 0` target branch | — | **No** |
| `chronon_divmod(day, month)`: fractional `x_frac` | `tests::day_to_month_matches_r` (31.25), `r_crosscheck_fixture_matches` (`runif` values), `prop_day_to_month` | Yes |
| `chronon_divmod(month, day)`: `from@n` scaling | `tests::bimonth_to_day_matches_r`, `bimonth_to_day` crosscheck, `prop_month_to_day` (`from_n=2,3`) | Yes |
| `chronon_divmod(month, day)`: negative `x_int` / `floor()` | `tests::month_to_day_matches_r` (-1), `r_crosscheck_fixture_matches`, `prop_month_to_day` | Yes |
| `chronon_divmod(month, day)`: `month > 2` leap adjustment | `tests::month_to_day_matches_r` (600, 660.5), `prop_month_to_day` | Yes |
| `chronon_divmod(month, day)`: `month == 2` leap day | `prop_month_to_day` edge months 361 (2000-02), 649 (2024-02), 637 (2023-02), -839 (1900-02) | Yes |
| `chronon_divmod(month, day)`: `to@n` scaling | `month_to_2day` in `r_crosscheck_fixture_matches`, `prop_month_to_day` (`to_n=2`) | Yes |
| `chronon_divmod(month, day)`: fractional month | `tests::month_to_day_matches_r` (660.5), `r_crosscheck_fixture_matches`, `prop_month_to_day` | Yes |
| `chronon_divmod(month, day)`: all 12 `monthdays` entries | random crosscheck/property; fixtures 600 and 660.5 | Yes |
| `chronon_divmod(day, year)`: `to@n != 1` error | `tests_property::day_to_year_rejects_multi_year_targets` | Yes (Rust panic; R raises `stop()`) |
| `chronon_divmod(day, year)`: `from@n` scaling | `twoday_to_year` in `r_crosscheck_fixture_matches`, `prop_day_to_year` (`from_n=2`) | Yes |
| `chronon_divmod(day, year)`: `z >= 0` era branch | all day fixtures and `prop_day_to_year` | Yes |
| `chronon_divmod(day, year)`: `z < 0` era branch | — | **No** (same day-domain limit) |
| `chronon_divmod(day, year)`: `mp >= 10`, `month <= 2` | `tests::day_to_year_matches_r` (365/366, 0/59), `prop_day_to_year` | Yes |
| `chronon_divmod(day, year)`: leap `ly` + fractional `x_frac` | `tests::day_to_year_matches_r` (19723, 19782), `prop_day_to_year` | Yes |
| `chronon_divmod(year, day)`: `from@n` scaling | `prop_year_to_day` (`from_n=2`) | Yes |
| `chronon_divmod(year, day)`: `to@n` scaling | `year_to_2day` in `r_crosscheck_fixture_matches`, `prop_year_to_day` (`to_n=2`) | Yes |
| `chronon_divmod(year, day)`: `floor_int` on positive/negative values | `tests::year_to_day_matches_r` (-1, 56.5), `prop_year_to_day` | Yes |
| `chronon_divmod(year, day)`: leap-year fractional adjustment | `prop_year_to_day` edges 54.5 (2024), 55.5 (2025), 30.5 (2000), 29.5 (1999), -1.5 (1968), -0.5 (1969) | Yes |
| `chronon_divmod(year, day)`: negative year (`x < -1970`) | — | **No** (year domain is -500..500) |
| `chronon_epoch(year) == 1970L` | hardcoded in Rust; every fixture depends on the 1970 epoch | Yes (implicit) |
| `cal_gregorian` unit constructor | hardcoded unit set in Rust; no object is constructed | N/A (not a computation) |

## Excluded surface (not ported, no tests)

| R function / method | Why no test | Covered |
|---|---|---|
| `time_unit_full` for year/quarter/month | formatting; out of scope | No |
| `time_unit_abbr` for year/quarter/month | formatting; out of scope | No |
| `chronon_format_linear` for year/quarter/month/day | formatting; out of scope | No |
| `chronon_format_cyclical` for month/year | formatting; out of scope | No |
| `chronon_parse_linear` for month/quarter/day | parsing; out of scope | No |
| `chronon_parse_cyclical` for month/year | parsing; out of scope | No |
| `chronon_cardinality_fixed` quarter→year (4), month→year (12), month→quarter (3) | cardinality algebra; out of scope | No |
| `chronon_cardinality` day→year (`at = NULL` error, `y@n != 1` error, leap result) | cardinality algebra; out of scope | No |
| `chronon_cardinality` day→month (`at = NULL` error, multi-month window) | cardinality algebra; out of scope | No |
| `bc_ad_format` (`i <= 0` BC branch) | labels; out of scope | No |
| `bc_ad_parse` (BC decode branch) | parsing; out of scope | No |
| `linear_labels_format` / `linear_labels_parse` for year | labels; out of scope | No |
| `cyclical_labels` for quarter/month | labels; out of scope | No |

The generic `chronon_divmod` dispatch lives in `R/00_chronon_divmod.R`, not
`cal_gregorian.R`, and is also not ported; the Rust API exposes the four
resolved methods directly.

## Uncovered branches, explicitly

1. `z < 0` era branch in day→month and day→year (requires day < -719468).
2. `n < 0` target-granule branch in day→month.
3. Negative `from_n` / `to_n` granule sizes in all four functions.
4. Negative years in `is_leap_year` / `month_start_days`.
5. Target granule sizes outside the tested 1/2/3 combinations (the code has no
   branch on these except day→year's `to@n != 1` error).
6. R error classes and messages are not reproduced; the Rust port panics/asserts
   where R calls `stop()`.
