use numpy::PyArray1;
use pyo3::prelude::*;

use nufrost_core::{
    hants_pixel, nufrost_pixel, NufrostConfig, HantsConfig, Zhu2015Config,
    zhu2015::fit_predict_pixel,
};

// ── Config merge helpers ─────────────────────────────────────────────────
// Fixture configs contain only a subset of fields.  We start from library
// defaults (matching Python test defaults) and overlay caller-provided JSON.
// This allows callers to pass partial configs safely.

fn default_nufrost_config() -> NufrostConfig {
    serde_json::from_str(
        r#"{
            "modes": 4096, "eps": 1e-12, "num_peaks": 10, "power_cum": 0.7,
            "ignore_dc_hz": 1e-10, "frequency_selection": "spectral",
            "preferred_periods_days": "365.25,182.625,91.3125,30.4375",
            "preferred_top_k": 4, "spectral_top_k": 4, "spectral_merge_tol": 0.15,
            "refine_peaks": true, "include_trend": true, "ridge_lam": 0.005,
            "freq_weight": 2.0, "huber_iters": 3, "huber_delta": 0.05,
            "min_obs": 12, "outlier_sigma": 2.0, "lambda_step": 1e30,
            "lambda_high": 0.005, "low_freq_period_days": 0.0,
            "step_dt_weighting": false, "max_outer_iter": 5, "outer_tol": 1e-3,
            "joint_outlier": false, "joint_outlier_sigma": 2.5,
            "admm_rho": 1.0, "admm_max_iter": 80, "admm_tol": 1e-4
        }"#,
    ).expect("hardcoded default config must be valid")
}

fn merge_nufrost_config(json_str: &str) -> Result<NufrostConfig, String> {
    let overrides: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("Invalid JSON: {}", e))?;
    let mut cfg = default_nufrost_config();
    if let serde_json::Value::Object(map) = &overrides {
        macro_rules! set_num { ($f:ident, $k:expr, $t:ty, $d:expr) => {
            if let Some(v) = map.get($k) { cfg.$f = v.as_f64().unwrap_or($d as f64) as $t; }
        }}
        macro_rules! set_bool { ($f:ident, $k:expr) => {
            if let Some(v) = map.get($k) { cfg.$f = v.as_bool().unwrap_or(cfg.$f); }
        }}
        macro_rules! set_str { ($f:ident, $k:expr) => {
            if let Some(v) = map.get($k) { if let Some(s) = v.as_str() { cfg.$f = s.to_string(); } }
        }}
        set_num!(modes, "modes", u32, 4096);
        set_num!(eps, "eps", f64, 1e-12);
        set_num!(num_peaks, "num_peaks", u32, 10);
        set_num!(power_cum, "power_cum", f64, 0.7);
        set_num!(ignore_dc_hz, "ignore_dc_hz", f64, 1e-10);
        set_num!(ridge_lam, "ridge_lam", f64, 0.005);
        set_num!(freq_weight, "freq_weight", f64, 2.0);
        set_num!(huber_iters, "huber_iters", u32, 3);
        set_num!(huber_delta, "huber_delta", f64, 0.05);
        set_num!(min_obs, "min_obs", u32, 12);
        set_num!(preferred_top_k, "preferred_top_k", u32, 4);
        set_num!(spectral_top_k, "spectral_top_k", u32, 4);
        set_num!(spectral_merge_tol, "spectral_merge_tol", f64, 0.15);
        set_num!(outlier_sigma, "outlier_sigma", f64, 2.0);
        set_num!(lambda_step, "lambda_step", f64, 1e30);
        set_num!(lambda_high, "lambda_high", f64, 0.005);
        set_num!(low_freq_period_days, "low_freq_period_days", f64, 0.0);
        set_num!(max_outer_iter, "max_outer_iter", u32, 5);
        set_num!(outer_tol, "outer_tol", f64, 1e-3);
        set_num!(admm_rho, "admm_rho", f64, 1.0);
        set_num!(admm_max_iter, "admm_max_iter", u32, 80);
        set_num!(admm_tol, "admm_tol", f64, 1e-4);
        set_bool!(refine_peaks, "refine_peaks");
        set_bool!(include_trend, "include_trend");
        set_bool!(step_dt_weighting, "step_dt_weighting");
        set_bool!(joint_outlier, "joint_outlier");
        set_str!(frequency_selection, "frequency_selection");
        set_str!(preferred_periods_days, "preferred_periods_days");
    }
    Ok(cfg)
}

// ── NUFROST wrapper ─────────────────────────────────────────────────────

#[pyfunction]
fn nufrost_pixel_rust(
    t_sec: Vec<f64>,
    y: Vec<f64>,
    target_t: f64,
    config_json: &str,
) -> PyResult<f64> {
    let config = merge_nufrost_config(config_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
    })?;

    if t_sec.len() != y.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("t_sec and y must have the same length (got {} vs {})", t_sec.len(), y.len()),
        ));
    }

    let (pred, _n_freqs) = nufrost_pixel(&t_sec, &y, target_t, &config);
    Ok(pred)
}

// ═══════════════════════════════════════════════════════════════════════════
//  HANTS Python wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// Run HANTS reconstruction on a single pixel.
///
/// Args:
///     t_days: 1-D array of timestamps in days since first observation.
///     y: 1-D array of observations (may contain NaN).
///     target_t: Target timestamp in days since first observation.
///     config_json: JSON string with HANTS configuration.
///
/// Returns:
///     Predicted value at target time, or NaN if not enough valid data.
#[pyfunction]
fn hants_pixel_rust(
    t_days: Vec<f64>,
    y: Vec<f64>,
    target_t: f64,
    config_json: &str,
) -> PyResult<f64> {
    let config: HantsConfig = serde_json::from_str(config_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid HANTS config JSON: {}", e))
    })?;

    if t_days.len() != y.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("t_days and y must have the same length (got {} vs {})", t_days.len(), y.len()),
        ));
    }

    let pred = hants_pixel(
        &t_days,
        &y,
        target_t,
        config.nof,
        &config.sf,
        config.valid_min,
        config.valid_max,
        config.fet,
        config.dod,
        config.period,
    );
    Ok(pred)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Zhu2015 Python wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// Run Zhu2015 (LASSO-based) reconstruction on a single pixel.
///
/// Args:
///     t_days: 1-D array of timestamps in days since first observation.
///     y: 1-D array of observations (may contain NaN).
///     target_t: Target timestamp in days since first observation.
///     config_json: JSON string with Zhu2015 configuration.
///
/// Returns:
///     Tuple of (prediction, qa_code).
///     prediction: f64 — predicted value or NaN.
///     qa_code: u32 — model order used (0 = median fallback, 1-3 = harmonic order).
#[pyfunction]
fn zhu2015_pixel_rust(
    t_days: Vec<f64>,
    y: Vec<f64>,
    target_t: f64,
    config_json: &str,
) -> PyResult<(f64, u32)> {
    let config: Zhu2015Config = serde_json::from_str(config_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid Zhu2015 config JSON: {}", e))
    })?;

    if t_days.len() != y.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("t_days and y must have the same length (got {} vs {})", t_days.len(), y.len()),
        ));
    }

    let result = fit_predict_pixel(&t_days, &y, target_t, config.lasso_alpha);
    Ok((result.prediction, result.qa))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Batch raster reconstruction (N-D variant with rayon parallelism)
// ═══════════════════════════════════════════════════════════════════════════

/// Run NUFROST on a full 2-D raster cube (time × pixels).
///
/// This uses rayon for parallel pixel processing internally.
///
/// Args:
///     py: Python GIL token (passed automatically).
///     t_sec: 1-D array of timestamps in seconds since epoch (length = n_time).
///     cube: 2-D array of shape (n_time, n_pixels), column-major pixel layout.
///     target_t: Target timestamp in seconds since epoch.
///     config_json: JSON string with NUFROST configuration.
///
/// Returns:
///     1-D numpy array of predictions (length = n_pixels).
#[pyfunction]
fn nufrost_raster_rust<'py>(
    py: Python<'py>,
    t_sec: Vec<f64>,
    cube: numpy::PyReadonlyArray2<'_, f64>,
    target_t: f64,
    config_json: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let config = merge_nufrost_config(config_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
    })?;

    let cube_view = cube.as_array();
    let n_time = cube_view.nrows();
    let n_pixels = cube_view.ncols();

    if n_time != t_sec.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("cube rows ({}) != t_sec length ({})", n_time, t_sec.len()),
        ));
    }

    let mut predictions = vec![f64::NAN; n_pixels];
    use rayon::prelude::*;
    predictions.par_iter_mut().enumerate().for_each(|(p, pred)| {
        let y: Vec<f64> = cube_view.column(p).to_vec();
        let (val, _n_freqs) = nufrost_pixel(&t_sec, &y, target_t, &config);
        *pred = val;
    });

    Ok(PyArray1::from_vec(py, predictions))
}

/// Run HANTS on a full 2-D raster cube (time × pixels).
///
/// See `nufrost_raster_rust` for argument description.
#[pyfunction]
fn hants_raster_rust<'py>(
    py: Python<'py>,
    t_days: Vec<f64>,
    cube: numpy::PyReadonlyArray2<'_, f64>,
    target_t: f64,
    config_json: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let config: HantsConfig = serde_json::from_str(config_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid HANTS config JSON: {}", e))
    })?;

    let cube_view = cube.as_array();
    let n_time = cube_view.nrows();
    let n_pixels = cube_view.ncols();

    if n_time != t_days.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("cube rows ({}) != t_days length ({})", n_time, t_days.len()),
        ));
    }

    let mut predictions = vec![f64::NAN; n_pixels];

    use rayon::prelude::*;

    predictions
        .par_iter_mut()
        .enumerate()
        .for_each(|(p, pred)| {
        let y: Vec<f64> = cube_view.column(p).to_vec();
        let val = hants_pixel(
            &t_days,
            &y,
            target_t,
            config.nof,
            &config.sf,
            config.valid_min,
            config.valid_max,
            config.fet,
            config.dod,
            config.period,
        );
        *pred = val;
    });

    Ok(PyArray1::from_vec(py, predictions))
}

/// Run Zhu2015 on a full 2-D raster cube (time × pixels).
///
/// Returns: tuple of (predictions, qa_band) as numpy arrays.
#[pyfunction]
fn zhu2015_raster_rust<'py>(
    py: Python<'py>,
    t_days: Vec<f64>,
    cube: numpy::PyReadonlyArray2<'_, f64>,
    target_t: f64,
    config_json: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u32>>)> {
    let config: Zhu2015Config = serde_json::from_str(config_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid Zhu2015 config JSON: {}", e))
    })?;

    let cube_view = cube.as_array();
    let n_time = cube_view.nrows();
    let n_pixels = cube_view.ncols();

    if n_time != t_days.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("cube rows ({}) != t_days length ({})", n_time, t_days.len()),
        ));
    }

    let mut predictions = vec![f64::NAN; n_pixels];
    let mut qa_band = vec![0u32; n_pixels];

    use rayon::prelude::*;

    predictions
        .par_iter_mut()
        .zip(qa_band.par_iter_mut())
        .enumerate()
        .for_each(|(p, (pred, qa))| {
        let y: Vec<f64> = cube_view.column(p).to_vec();
        let result = fit_predict_pixel(&t_days, &y, target_t, config.lasso_alpha);
        *pred = result.prediction;
        *qa = result.qa;
    });

    Ok((
        PyArray1::from_vec(py, predictions),
        PyArray1::from_vec(py, qa_band),
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Python module definition
// ═══════════════════════════════════════════════════════════════════════════

/// NUFROST Python module — Rust-backed reconstruction algorithms.
#[pymodule]
fn nufrost_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nufrost_pixel_rust, m)?)?;
    m.add_function(wrap_pyfunction!(hants_pixel_rust, m)?)?;
    m.add_function(wrap_pyfunction!(zhu2015_pixel_rust, m)?)?;
    m.add_function(wrap_pyfunction!(nufrost_raster_rust, m)?)?;
    m.add_function(wrap_pyfunction!(hants_raster_rust, m)?)?;
    m.add_function(wrap_pyfunction!(zhu2015_raster_rust, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_no_py() {
        assert_eq!(2_usize + 2, 4);
    }
}
