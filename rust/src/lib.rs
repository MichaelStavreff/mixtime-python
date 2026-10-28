//! Rust port of mixtime's Gregorian `chronon_divmod` methods.
//!
//! Ported from `R/cal_gregorian.R` in mixtime 0.4.0. The R implementation
//! works on doubles throughout, so this port does too; floor division and
//! floor modulo are reproduced explicitly (`fdiv`/`fmod` in R/00_utils.R).
//!
//! Only vector operations are ported. The kernels take an `ndarray` view so
//! the PyO3 layer can hand them NumPy's buffer without copying it.

use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// R's `%/%`: floor(x / y).
fn fdiv(x: f64, y: f64) -> f64 {
    (x / y).floor()
}

/// R's `%%`: x - floor(x / y) * y.
fn fmod(x: f64, y: f64) -> f64 {
    x - (x / y).floor() * y
}

fn is_leap_year(year: f64) -> bool {
    let year = year.floor();
    let divisible = |n: f64| year == n * (year / n).floor();
    (divisible(4.0) && !divisible(100.0)) || divisible(400.0)
}

const MONTH_DAYS: [f64; 12] = [
    31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0,
];

/// Days since the Gregorian epoch (1970-01-01) to the first day of the month
/// `month_index` months after the epoch (0-indexed, may be negative).
fn month_start_days(month_index: f64) -> f64 {
    let year = fdiv(month_index, 12.0) + 1970.0;
    let month = month_index - 12.0 * (year - 1970.0) + 1.0;
    let ly = if is_leap_year(year) { 1.0 } else { 0.0 };

    365.0 * (year - 1970.0) + fdiv(year - 1968.0, 4.0) - fdiv(year - 1900.0, 100.0)
        + fdiv(year - 1600.0, 400.0)
        + fdiv(367.0 * month - 362.0, 12.0)
        + if month > 2.0 { -2.0 + ly } else { 0.0 }
        - ly
}

/// Split proleptic-Gregorian days since epoch into (year, month).
fn civil_from_days(x_int: f64) -> (f64, f64) {
    // Shift to days since 0000-03-01 (algorithm anchor).
    let z = x_int + 719468.0;

    let era = if z >= 0.0 {
        fdiv(z, 146097.0)
    } else {
        fdiv(z - 146096.0, 146097.0)
    };
    let doe = z - era * 146097.0;
    let yoe = fdiv(
        doe - fdiv(doe, 1460.0) + fdiv(doe, 36524.0) - fdiv(doe, 146096.0),
        365.0,
    );
    let doy = doe - (365.0 * yoe + fdiv(yoe, 4.0) - fdiv(yoe, 100.0) + fdiv(yoe, 400.0));
    let mp = fdiv(5.0 * doy + 2.0, 153.0);
    let month = mp + 3.0 - 12.0 * if mp >= 10.0 { 1.0 } else { 0.0 };
    let year = yoe + era * 400.0 + if month <= 2.0 { 1.0 } else { 0.0 };

    (year, month)
}

/// Fill `divs` and `mods` with `chronon_divmod(day(from_n), month(to_n), x)`.
fn day_to_month_into(
    x: ArrayView1<'_, f64>,
    from_n: f64,
    to_n: f64,
    divs: &mut [f64],
    mods: &mut [f64],
) {
    assert_eq!(divs.len(), x.len(), "output length mismatch");
    assert_eq!(mods.len(), x.len(), "output length mismatch");

    for (i, &value) in x.iter().enumerate() {
        let scaled = from_n * value;
        let x_int = scaled.floor();
        let x_frac = scaled - x_int;

        let (year, month) = civil_from_days(x_int);
        let res = (year - 1970.0) * 12.0 + month - 1.0;

        // Block index whose window [div*n, div*n + |n|) contains `res`.
        let n = to_n;
        let div = if n > 0.0 {
            fdiv(res, n)
        } else {
            -fdiv(res, -n)
        };

        divs[i] = div;
        mods[i] = (x_int - month_start_days(div * n) + x_frac) / from_n;
    }
}

/// `chronon_divmod(day(from_n), month(to_n), x)`.
pub fn day_to_month(x: ArrayView1<'_, f64>, from_n: f64, to_n: f64) -> (Vec<f64>, Vec<f64>) {
    let mut divs = vec![0.0; x.len()];
    let mut mods = vec![0.0; x.len()];
    day_to_month_into(x, from_n, to_n, &mut divs, &mut mods);
    (divs, mods)
}

/// Fill `divs` and `mods` with `chronon_divmod(day(from_n), year(to_n), x)`.
fn day_to_year_into(
    x: ArrayView1<'_, f64>,
    from_n: f64,
    to_n: f64,
    divs: &mut [f64],
    mods: &mut [f64],
) {
    assert!(
        (to_n - 1.0).abs() < f64::EPSILON,
        "Converting to non-year chronons from days not yet supported"
    );
    assert_eq!(divs.len(), x.len(), "output length mismatch");
    assert_eq!(mods.len(), x.len(), "output length mismatch");

    for (i, &value) in x.iter().enumerate() {
        let scaled = from_n * value;
        let x_int = scaled.floor();
        let x_frac = scaled - x_int;

        let z = x_int + 719468.0;
        let era = if z >= 0.0 {
            fdiv(z, 146097.0)
        } else {
            fdiv(z - 146096.0, 146097.0)
        };
        let doe = z - era * 146097.0;
        let yoe = fdiv(
            doe - fdiv(doe, 1460.0) + fdiv(doe, 36524.0) - fdiv(doe, 146096.0),
            365.0,
        );
        let doy = doe - (365.0 * yoe + fdiv(yoe, 4.0) - fdiv(yoe, 100.0) + fdiv(yoe, 400.0));
        let mp = fdiv(5.0 * doy + 2.0, 153.0);
        let month = mp + 3.0 - 12.0 * if mp >= 10.0 { 1.0 } else { 0.0 };
        let year = yoe + era * 400.0 + if month <= 2.0 { 1.0 } else { 0.0 };

        let ly = if is_leap_year(year) { 1.0 } else { 0.0 };
        let yday = fmod(doy + 59.0 + ly, 365.0 + ly);

        divs[i] = year - 1970.0;
        mods[i] = (yday + x_frac) / from_n;
    }
}

/// `chronon_divmod(day(from_n), year(to_n), x)`.
pub fn day_to_year(x: ArrayView1<'_, f64>, from_n: f64, to_n: f64) -> (Vec<f64>, Vec<f64>) {
    let mut divs = vec![0.0; x.len()];
    let mut mods = vec![0.0; x.len()];
    day_to_year_into(x, from_n, to_n, &mut divs, &mut mods);
    (divs, mods)
}

/// Fill `divs` with `chronon_divmod(month(from_n), day(to_n), x)`; `mods` is
/// zero-filled.
fn month_to_day_into(
    x: ArrayView1<'_, f64>,
    from_n: f64,
    to_n: f64,
    divs: &mut [f64],
    mods: &mut [f64],
) {
    assert_eq!(divs.len(), x.len(), "output length mismatch");
    assert_eq!(mods.len(), x.len(), "output length mismatch");
    mods.fill(0.0);

    for (i, &value) in x.iter().enumerate() {
        let scaled = from_n * value;
        let x_int = scaled.floor();
        let x_frac = scaled - x_int;

        let year = fdiv(x_int, 12.0) + 1970.0;
        let ly = if is_leap_year(year) { 1.0 } else { 0.0 };
        let month = x_int - 12.0 * (year - 1970.0) + 1.0;

        let mut result = 365.0 * (year - 1970.0) + fdiv(year - 1968.0, 4.0)
            - fdiv(year - 1900.0, 100.0)
            + fdiv(year - 1600.0, 400.0)
            + fdiv(367.0 * month - 362.0, 12.0)
            + if month > 2.0 { -2.0 + ly } else { 0.0 }
            - ly;

        let month_idx = month as usize - 1;
        let days_in_month = MONTH_DAYS[month_idx] + if month == 2.0 { ly } else { 0.0 };
        result += x_frac * days_in_month;

        divs[i] = result / to_n;
    }
}

/// `chronon_divmod(month(from_n), day(to_n), x)`.
pub fn month_to_day(x: ArrayView1<'_, f64>, from_n: f64, to_n: f64) -> (Vec<f64>, Vec<f64>) {
    let mut divs = vec![0.0; x.len()];
    let mut mods = vec![0.0; x.len()];
    month_to_day_into(x, from_n, to_n, &mut divs, &mut mods);
    (divs, mods)
}

/// Fill `divs` with `chronon_divmod(year(from_n), day(to_n), x)`; `mods` is
/// zero-filled.
fn year_to_day_into(
    x: ArrayView1<'_, f64>,
    from_n: f64,
    to_n: f64,
    divs: &mut [f64],
    mods: &mut [f64],
) {
    assert_eq!(divs.len(), x.len(), "output length mismatch");
    assert_eq!(mods.len(), x.len(), "output length mismatch");
    mods.fill(0.0);

    for (i, &value) in x.iter().enumerate() {
        let scaled = from_n * value;

        let days = 365.0 * scaled + ((scaled + 1.0) / 4.0).floor()
            - ((scaled + 69.0) / 100.0).floor()
            + ((scaled + 369.0) / 400.0).floor()
            + if is_leap_year(scaled + 1970.0) {
                scaled - scaled.floor()
            } else {
                0.0
            };

        divs[i] = days / to_n;
    }
}

/// `chronon_divmod(year(from_n), day(to_n), x)`.
pub fn year_to_day(x: ArrayView1<'_, f64>, from_n: f64, to_n: f64) -> (Vec<f64>, Vec<f64>) {
    let mut divs = vec![0.0; x.len()];
    let mut mods = vec![0.0; x.len()];
    year_to_day_into(x, from_n, to_n, &mut divs, &mut mods);
    (divs, mods)
}

type OutputPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Allocate the two output arrays and fill them in place, so the Python side
/// receives NumPy buffers without an intermediate `Vec` or list conversion.
fn output_pair<'py>(py: Python<'py>, n: usize) -> OutputPair<'py> {
    (
        PyArray1::<f64>::zeros(py, n, false),
        PyArray1::<f64>::zeros(py, n, false),
    )
}

#[pyfunction]
#[pyo3(name = "day_to_month")]
#[pyo3(signature = (x, from_n=1.0, to_n=1.0))]
fn py_day_to_month<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    from_n: f64,
    to_n: f64,
) -> PyResult<OutputPair<'py>> {
    let (divs, mods) = output_pair(py, x.len()?);
    {
        let mut divs_rw = divs.readwrite();
        let mut mods_rw = mods.readwrite();
        day_to_month_into(
            x.as_array(),
            from_n,
            to_n,
            divs_rw.as_slice_mut()?,
            mods_rw.as_slice_mut()?,
        );
    }
    Ok((divs, mods))
}

#[pyfunction]
#[pyo3(name = "day_to_year")]
#[pyo3(signature = (x, from_n=1.0, to_n=1.0))]
fn py_day_to_year<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    from_n: f64,
    to_n: f64,
) -> PyResult<OutputPair<'py>> {
    let (divs, mods) = output_pair(py, x.len()?);
    {
        let mut divs_rw = divs.readwrite();
        let mut mods_rw = mods.readwrite();
        day_to_year_into(
            x.as_array(),
            from_n,
            to_n,
            divs_rw.as_slice_mut()?,
            mods_rw.as_slice_mut()?,
        );
    }
    Ok((divs, mods))
}

#[pyfunction]
#[pyo3(name = "month_to_day")]
#[pyo3(signature = (x, from_n=1.0, to_n=1.0))]
fn py_month_to_day<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    from_n: f64,
    to_n: f64,
) -> PyResult<OutputPair<'py>> {
    let (divs, mods) = output_pair(py, x.len()?);
    {
        let mut divs_rw = divs.readwrite();
        let mut mods_rw = mods.readwrite();
        month_to_day_into(
            x.as_array(),
            from_n,
            to_n,
            divs_rw.as_slice_mut()?,
            mods_rw.as_slice_mut()?,
        );
    }
    Ok((divs, mods))
}

#[pyfunction]
#[pyo3(name = "year_to_day")]
#[pyo3(signature = (x, from_n=1.0, to_n=1.0))]
fn py_year_to_day<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    from_n: f64,
    to_n: f64,
) -> PyResult<OutputPair<'py>> {
    let (divs, mods) = output_pair(py, x.len()?);
    {
        let mut divs_rw = divs.readwrite();
        let mut mods_rw = mods.readwrite();
        year_to_day_into(
            x.as_array(),
            from_n,
            to_n,
            divs_rw.as_slice_mut()?,
            mods_rw.as_slice_mut()?,
        );
    }
    Ok((divs, mods))
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_day_to_month, m)?)?;
    m.add_function(wrap_pyfunction!(py_day_to_year, m)?)?;
    m.add_function(wrap_pyfunction!(py_month_to_day, m)?)?;
    m.add_function(wrap_pyfunction!(py_year_to_day, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(x: &[f64]) -> ArrayView1<'_, f64> {
        ArrayView1::from(x)
    }

    #[test]
    fn leap_years() {
        assert!(is_leap_year(2000.0));
        assert!(is_leap_year(2024.0));
        assert!(!is_leap_year(1900.0));
        assert!(!is_leap_year(2023.0));
        assert!(!is_leap_year(1970.0));
    }

    #[test]
    fn day_to_month_matches_r() {
        let x = [
            -1.0, 0.0, 1.0, 59.0, 60.0, 90.0, 365.0, 366.0, 19723.0, 19782.0, 20700.0, 31.25,
        ];
        let (div, modu) = day_to_month(view(&x), 1.0, 1.0);
        assert_eq!(
            div,
            vec![-1.0, 0.0, 0.0, 2.0, 2.0, 3.0, 12.0, 12.0, 648.0, 649.0, 680.0, 1.0]
        );
        assert_eq!(
            modu,
            vec![30.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 28.0, 3.0, 0.25]
        );
    }

    #[test]
    fn day_to_quarter_matches_r() {
        let x = [
            -1.0, 0.0, 1.0, 59.0, 60.0, 90.0, 365.0, 366.0, 19723.0, 19782.0, 20700.0, 31.25,
        ];
        let (div, modu) = day_to_month(view(&x), 1.0, 3.0);
        assert_eq!(
            div,
            vec![-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 216.0, 216.0, 226.0, 0.0]
        );
        assert_eq!(
            modu,
            vec![91.0, 0.0, 1.0, 59.0, 60.0, 0.0, 0.0, 1.0, 0.0, 59.0, 65.0, 31.25]
        );
    }

    #[test]
    fn day_to_year_matches_r() {
        let x = [
            -1.0, 0.0, 1.0, 59.0, 60.0, 90.0, 365.0, 366.0, 19723.0, 19782.0, 20700.0, 31.25,
        ];
        let (div, modu) = day_to_year(view(&x), 1.0, 1.0);
        assert_eq!(
            div,
            vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 54.0, 54.0, 56.0, 0.0]
        );
        assert_eq!(
            modu,
            vec![364.0, 0.0, 1.0, 59.0, 60.0, 90.0, 0.0, 1.0, 0.0, 59.0, 246.0, 31.25]
        );
    }

    #[test]
    fn month_to_day_matches_r() {
        let x = [-1.0, 0.0, 1.0, 12.0, 13.0, 600.0, 660.5];
        let (div, modu) = month_to_day(view(&x), 1.0, 1.0);
        assert_eq!(div, vec![-31.0, 0.0, 31.0, 365.0, 396.0, 18262.0, 20104.5]);
        assert_eq!(modu, vec![0.0; 7]);
    }

    #[test]
    fn bimonth_to_day_matches_r() {
        let x = [-1.0, 0.0, 1.0, 12.0, 13.0, 600.0, 660.5];
        let (div, _) = month_to_day(view(&x), 2.0, 1.0);
        assert_eq!(div, vec![-61.0, 0.0, 59.0, 730.0, 790.0, 36525.0, 40208.0]);
    }

    #[test]
    fn year_to_day_matches_r() {
        let x = [-1.0, 0.0, 1.0, 54.0, 55.0, 56.5];
        let (div, modu) = year_to_day(view(&x), 1.0, 1.0);
        assert_eq!(div, vec![-365.0, 0.0, 365.0, 19723.0, 20089.0, 20636.5]);
        assert_eq!(modu, vec![0.0; 6]);
    }
}

#[cfg(test)]
mod tests_property;
