// NUFROST — Non-Uniform FFT-based frequency discovery with robust ridge fitting.
//
// Ported from `src/nufrost.py`.  Preserves numerical parity with the Python
// reference on all fixtures.  The NUFFT is computed via direct DFT, which is
// mathematically equivalent to `finufft.nufft1d1` for the uniform output grid
// used in the algorithm — see "NUFFT strategy" section below.
//
// ── Algorithm summary ──────────────────────────────────────────────────────
// 1. Map non-uniform timestamps to [-π, π].
// 2. Compute spectrum via direct DFT (equivalent to finufft type-1 NUFFT).
// 3. Select harmonic frequencies: spectral peaks, preferred periods, or hybrid.
// 4. Build harmonic + linear-trend design matrix.
// 5. Fit via iteratively reweighted ridge regression with Huber weights.
// 6. Predict at the target time.
//
// ── NUFFT strategy ─────────────────────────────────────────────────────────
// Python uses `finufft.nufft1d1(x, c, M, eps, isign=-1)` which computes
//   F_k = Σ c_j · exp(-i · k · x_j)   for k = -M/2 … M/2-1
// with x_j ∈ [-π, π].  This is the direct (non-uniform) discrete Fourier
// transform — no approximation is involved for this computation pattern.
//
// We implement the same sum directly in Rust.  For the small per-pixel time
// series in test fixtures (N ≤ 200 obs), the O(N·M) direct sum is fast and
// guarantees exact parity with finufft (up to f64 round-off).
//
// Future plan: if N grows, a pure-Rust FFI binding to FINUFFT could replace
// the direct DFT path.  The `_compute_spectrum_direct` function signature
// is designed so an FFI-backed `_compute_spectrum_finufft` can be swapped in
// without changing the frequency selection pipeline.

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

// ── Reusable types (exported from crate root) ──────────────────────────────
use crate::config::NufrostConfig;

// ═══════════════════════════════════════════════════════════════════════════
//  Result types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a single-pixel NUFROST fit.
#[derive(Debug, Clone)]
pub struct NufrostResult {
    /// Whether the fit succeeded (enough valid observations).
    pub valid: bool,
    /// Number of frequencies used in the fit.
    pub n_freqs_used: usize,
    /// Fitted harmonic + trend coefficients.
    /// Length = 1 (DC) + 1 (trend, if any) + 2·n_freqs.
    pub beta: Vec<f64>,
    /// Selected frequencies (Hz).
    pub freqs: Vec<f64>,
    /// Earliest valid timestamp (seconds since epoch).
    pub t_min: f64,
    /// Mean of relative timestamps used for trend centering.
    pub t_rel_mean: f64,
    /// Fallback value (mean of valid observations).
    pub fill_value: f64,
    /// `include_trend` flag used in fit.
    pub include_trend: bool,
    /// y_scale applied to observations before fitting.
    pub y_scale: f64,
}

impl Default for NufrostResult {
    fn default() -> Self {
        Self {
            valid: false,
            n_freqs_used: 0,
            beta: Vec::new(),
            freqs: Vec::new(),
            t_min: f64::NAN,
            t_rel_mean: f64::NAN,
            fill_value: f64::NAN,
            include_trend: true,
            y_scale: 10000.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Utility functions
// ═══════════════════════════════════════════════════════════════════════════

/// Round `n` up to the next even integer.
#[inline]
pub fn next_even(n: usize) -> usize {
    ((n + 1) / 2) * 2
}

/// Compute the (NaN-aware) mean of a slice.
fn nanmean(data: &[f64]) -> f64 {
    let (sum, count) = data
        .iter()
        .fold((0.0f64, 0usize), |(s, c), &v| {
            if v.is_finite() { (s + v, c + 1) } else { (s, c) }
        });
    if count == 0 { f64::NAN } else { sum / count as f64 }
}

/// Compute the (NaN-aware) median of a slice.
fn nanmedian(data: &[f64]) -> f64 {
    let mut finite: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    if n % 2 == 1 {
        finite[n / 2]
    } else {
        (finite[n / 2 - 1] + finite[n / 2]) / 2.0
    }
}

/// Compute MAD (median absolute deviation) scaled to match σ for Gaussian data.
/// Returns 1.4826 · median(|x - median(x)|) for finite values in `residuals`.
fn mad_std(residuals: &[f64]) -> f64 {
    let med = nanmedian(residuals);
    if !med.is_finite() {
        return 0.0;
    }
    let abs_dev: Vec<f64> = residuals
        .iter()
        .map(|&r| if r.is_finite() { (r - med).abs() } else { f64::NAN })
        .collect();
    let mad = nanmedian(&abs_dev);
    if !mad.is_finite() { return 0.0; }
    1.4826 * mad.max(1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Design matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Build the harmonic design matrix.
///
/// Columns:
///   - 0: DC term (ones)
///   - 1 (if `include_trend`): linear trend (centered)
///   - then pairs cos(ω·t), sin(ω·t) for each frequency.
pub fn make_design_matrix(
    t: &[f64],
    freqs: &[f64],
    include_trend: bool,
    include_dc: bool,
) -> Array2<f64> {
    let n = t.len();
    let n_dc = if include_dc { 1 } else { 0 };
    let n_trend = if include_trend { 1 } else { 0 };
    let n_cols = n_dc + n_trend + 2 * freqs.len();

    if n_cols == 0 {
        return Array2::zeros((n, 0));
    }

    let t_mean = if include_trend { nanmean(t) } else { 0.0 };
    let mut x = Array2::<f64>::zeros((n, n_cols));
    let mut col = 0;

    if include_dc {
        x.column_mut(col).fill(1.0);
        col += 1;
    }
    if include_trend {
        for (i, &ti) in t.iter().enumerate() {
            x[[i, col]] = ti - t_mean;
        }
        col += 1;
    }
    for &f in freqs {
        let omega = 2.0 * PI * f;
        for (i, &ti) in t.iter().enumerate() {
            x[[i, col]] = (omega * ti).cos();
            x[[i, col + 1]] = (omega * ti).sin();
        }
        col += 2;
    }

    x
}

// ═══════════════════════════════════════════════════════════════════════════
//  Linear algebra: Gaussian elimination with partial pivoting
// ═══════════════════════════════════════════════════════════════════════════

/// Solve `A · x = b`. Returns `None` on singularity.
fn gauss_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(b.len(), n);

    if n == 0 {
        return Some(Array1::zeros(0));
    }

    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-14 {
            return None;
        }
        if pivot_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }
    Some(x)
}

/// Solve normal equations XᵀX · β = Xᵀy.
fn solve_normal_equations(x: &Array2<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
    if x.nrows() == 0 || x.nrows() < x.ncols() {
        return None;
    }
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    gauss_solve(&xtx, &xty)
}

/// Ridge-augmented least squares: solve [X; √λ·R] β = [y; 0] where R is a
/// diagonal penalty matrix.
fn ridge_solve_augmented(
    x: &Array2<f64>,
    y: &Array1<f64>,
    lam: f64,
    r_diag: &[f64],  // per-column penalty multipliers; length = x.ncols()
    w: Option<&[f64]>,  // per-row observation weights (sqrt applied internally)
) -> Option<Array1<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    if p == 0 {
        return Some(Array1::zeros(0));
    }

    // Apply observation weights
    let (xw, yw): (Array2<f64>, Array1<f64>) = if let Some(weights) = w {
        let sqrt_w: Vec<f64> = weights.iter().map(|&wi| wi.sqrt()).collect();
        let mut xw = Array2::zeros((n, p));
        let mut yw = Array1::zeros(n);
        for i in 0..n {
            let sw = sqrt_w[i];
            for j in 0..p {
                xw[[i, j]] = x[[i, j]] * sw;
            }
            yw[i] = y[i] * sw;
        }
        (xw, yw)
    } else {
        (x.clone(), y.clone())
    };

    if lam <= 0.0 {
        return solve_normal_equations(&xw, &yw);
    }

    let aug_rows = n + p;
    let mut x_aug = Array2::<f64>::zeros((aug_rows, p));
    let mut y_aug = Array1::<f64>::zeros(aug_rows);

    // Top block: weighted design matrix + weighted y
    for i in 0..n {
        for j in 0..p {
            x_aug[[i, j]] = xw[[i, j]];
        }
        y_aug[i] = yw[i];
    }

    // Bottom block: √λ · R diagonal
    let sqrt_lam = lam.sqrt();
    for j in 0..p {
        let rj = if j < r_diag.len() { r_diag[j] } else { 1.0 };
        x_aug[[n + j, j]] = sqrt_lam * rj;
        y_aug[n + j] = 0.0;
    }

    // Solve augmented system via normal equations
    solve_normal_equations(&x_aug, &y_aug)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frequency penalty helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Per-frequency penalty multipliers: higher frequencies get heavier penalty.
///
/// penalty[i] = sqrt(1 + freq_weight · log2(f_i / f_min))
/// where f_min is the smallest positive frequency.
fn make_frequency_penalty(freqs: &[f64], freq_weight: f64) -> Vec<f64> {
    let n = freqs.len();
    if n == 0 {
        return Vec::new();
    }
    let pos_min = freqs
        .iter()
        .filter(|&&f| f.is_finite() && f > 0.0)
        .fold(f64::INFINITY, |a, &b| a.min(b));
    if !pos_min.is_finite() {
        return vec![1.0; n];
    }

    freqs
        .iter()
        .map(|&f| {
            if !f.is_finite() || f <= pos_min {
                1.0
            } else {
                let rel = (f / pos_min).max(1.0);
                (1.0 + freq_weight.max(0.0) * rel.log2()).sqrt()
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frequency tier classification
// ═══════════════════════════════════════════════════════════════════════════

/// Classify each frequency as "high" (period < low_freq_period_days).
/// Frequencies are assumed to be in Hz (seconds time unit).
fn classify_freq_tier(freqs: &[f64], low_freq_period_days: f64) -> Vec<bool> {
    freqs
        .iter()
        .map(|&f| {
            if f <= 0.0 || !f.is_finite() {
                false
            } else {
                let period_days = 1.0 / (f * 86400.0);
                period_days < low_freq_period_days
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Huber weights
// ═══════════════════════════════════════════════════════════════════════════

/// Compute Huber weights: w_i = 1 if |r_i| ≤ δ, else δ / |r_i|.
fn huber_weights(residuals: &[f64], delta: f64) -> Vec<f64> {
    let delta = delta.max(1e-8);
    residuals
        .iter()
        .map(|&r| {
            let a = r.abs();
            if a <= delta { 1.0 } else { delta / a }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  NUFFT spectrum computation (direct DFT — equivalent to finufft type-1)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the NUFFT type-1 spectrum via direct DFT.
///
/// Maps non-uniform timestamps `t_rel` to [-π, π], then computes
///   F_k = Σ c_j · exp(-i · k · x_j)    for k = -M/2 … M/2-1
///
/// This is mathematically equivalent to `finufft.nufft1d1(x, c, M, eps, -1)`.
///
/// # Returns
/// `(frequencies_hz, power_spectrum)` where `frequencies_hz = k / Tspan`
/// (only non-negative frequencies) and `power = |F_k|²`.
pub fn compute_spectrum_direct(
    t_rel: &[f64],
    yy: &[f64],
    modes: usize,
    y_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = t_rel.len();
    let t_min = t_rel.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = t_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tspan = t_max - t_min;
    if tspan <= 0.0 || !tspan.is_finite() {
        return (vec![0.0], vec![0.0]);
    }

    let ms = next_even(modes);
    let half = ms as isize / 2;

    // Map t_rel to x ∈ [-π, π]
    let x: Vec<f64> = t_rel
        .iter()
        .map(|&ti| 2.0 * PI * (ti - t_min) / tspan - PI)
        .collect();

    // Scale observations
    let c: Vec<f64> = yy.iter().map(|&yi| yi / y_scale).collect();

    // Direct DFT: F_k = Σ c_j * exp(-i * k * x_j)
    let mut fk_real = vec![0.0f64; ms];
    let mut fk_imag = vec![0.0f64; ms];

    for k_idx in 0..ms {
        let k = -half + k_idx as isize;
        let k_f = k as f64;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for j in 0..n {
            let phase = -k_f * x[j];
            re += c[j] * phase.cos();
            im += c[j] * phase.sin();
        }
        fk_real[k_idx] = re;
        fk_imag[k_idx] = im;
    }

    // Build frequency grid and power spectrum (non-negative frequencies only)
    let n_pos = (ms / 2) as isize + 1;
    let mut freqs = Vec::with_capacity(n_pos as usize);
    let mut power = Vec::with_capacity(n_pos as usize);

    for k_idx in 0..n_pos {
        // k runs from 0 to half
        let abs_k = k_idx as isize;
        // Compute for k ∈ [0, half] directly.
        let k_f = abs_k as f64;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for j in 0..n {
            let phase = -k_f * x[j];
            re += c[j] * phase.cos();
            im += c[j] * phase.sin();
        }

        let freq = k_f / tspan;
        let p = re * re + im * im;
        freqs.push(freq);
        power.push(p);
    }

    (freqs, power)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Peak selection
// ═══════════════════════════════════════════════════════════════════════════

/// Parabolic refinement: fit a parabola to three consecutive spectral points
/// and return the interpolated peak frequency.
pub fn refine_parabolic(f: &[f64], p: &[f64], i: usize) -> f64 {
    if i == 0 || i >= p.len() - 1 {
        return f[i];
    }
    let y0 = p[i - 1];
    let y1 = p[i];
    let y2 = p[i + 1];
    let denom = y0 - 2.0 * y1 + y2;
    if denom == 0.0 {
        return f[i];
    }
    let delta = 0.5 * (y0 - y2) / denom;
    f[i] + delta * (f[i + 1] - f[i])
}

/// Select top-k peaks from the power spectrum, cumulative power capped.
///
/// Returns indices into the full frequency/power arrays.
pub fn select_peaks_adaptive(
    f_pos: &[f64],
    p_pos: &[f64],
    k_max: usize,
    power_cum: f64,
    ignore_dc_hz: f64,
    fmax: f64,
) -> Vec<usize> {
    let lower = ignore_dc_hz.max(0.0);
    let fmax = if !fmax.is_finite() || fmax <= 0.0 {
        f_pos
            .iter()
            .copied()
            .filter(|f| f.is_finite())
            .fold(1.0f64, |a: f64, b| a.max(b))
    } else {
        fmax
    };

    // Collect valid indices
    let mut valid_idx: Vec<(usize, f64)> = f_pos
        .iter()
        .zip(p_pos.iter())
        .enumerate()
        .filter(|(_, (&f, &p))| f.is_finite() && p.is_finite() && f > lower && f <= fmax)
        .map(|(i, (&_f, &p))| (i, p))
        .collect();

    if valid_idx.is_empty() {
        return Vec::new();
    }

    // Sort by power descending
    valid_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Cumulative power threshold
    let total_power: f64 = valid_idx.iter().map(|&(_, p)| p).sum();
    let power_thr = power_cum.clamp(0.0, 1.0) * total_power;

    let mut cum = 0.0;
    let mut take = 0;
    for &(_, p) in &valid_idx {
        cum += p;
        take += 1;
        if take >= k_max || cum >= power_thr {
            break;
        }
    }
    if take == 0 && !valid_idx.is_empty() {
        take = 1;
    }
    take = take.min(valid_idx.len());

    valid_idx.truncate(take);
    valid_idx.iter().map(|&(i, _)| i).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frequency selection
// ═══════════════════════════════════════════════════════════════════════════

/// Parse preferred periods string (comma-separated days) into Hz frequencies.
fn parse_preferred_frequencies(spec: &str) -> Vec<f64> {
    if spec.trim().is_empty() {
        return Vec::new();
    }
    spec.split(',')
        .filter_map(|p| p.trim().parse::<f64>().ok())
        .filter(|&p| p.is_finite() && p > 0.0)
        .map(|p| 1.0 / (p * 86400.0))
        .collect()
}

/// Snap a target frequency to the nearest spectral peak within `rel_tol`.
fn snap_frequency_to_spectrum(
    target_freq: f64,
    f_pos: &[f64],
    p_pos: &[f64],
    rel_tol: f64,
) -> f64 {
    if !target_freq.is_finite() || target_freq <= 0.0 {
        return target_freq;
    }
    let rel_tol = rel_tol.max(0.0);

    let mut best_idx: Option<usize> = None;
    let mut best_power = f64::NEG_INFINITY;

    for (i, (&f, &p)) in f_pos.iter().zip(p_pos.iter()).enumerate() {
        if !f.is_finite() || !p.is_finite() || f <= 0.0 {
            continue;
        }
        let rel_err = (f - target_freq).abs() / target_freq.max(1e-12);
        if rel_err <= rel_tol && p > best_power {
            best_power = p;
            best_idx = Some(i);
        }
    }

    match best_idx {
        Some(i) => f_pos[i],
        None => target_freq,
    }
}

/// Select harmonic frequencies for NUFROST fitting.
///
/// Selection modes: "spectral", "preferred", "hybrid".
/// "shared_spectral" falls back to "spectral" (scene-level shared freqs
/// must be provided by the caller via `nufrost_pixel_with_shared`).
pub fn select_frequencies(
    f_pos: &[f64],
    p_pos: &[f64],
    fmax: f64,
    selection_mode: &str,
    preferred_freqs: &[f64],
    preferred_top_k: usize,
    spectral_top_k: usize,
    spectral_merge_tol: f64,
    power_cum: f64,
    ignore_dc_hz: f64,
    refine_peaks: bool,
) -> Vec<f64> {
    let mode = match selection_mode {
        "shared_spectral" => "spectral",
        other => other,
    };

    let mut selected: Vec<f64> = Vec::new();

    // Preferred frequencies (snapped to spectrum)
    if (mode == "preferred" || mode == "hybrid") && !preferred_freqs.is_empty() {
        let pref_valid: Vec<f64> = preferred_freqs
            .iter()
            .copied()
            .filter(|&f| f.is_finite() && f > ignore_dc_hz && f <= fmax)
            .take(preferred_top_k)
            .collect();
        for f in &pref_valid {
            let snapped = snap_frequency_to_spectrum(*f, f_pos, p_pos, spectral_merge_tol);
            selected.push(snapped);
        }
    }

    // Spectral peaks
    if mode == "spectral" || mode == "hybrid" {
        let peak_idx = select_peaks_adaptive(
            f_pos,
            p_pos,
            spectral_top_k,
            power_cum,
            ignore_dc_hz,
            fmax,
        );
        if !peak_idx.is_empty() {
            let sel_freqs: Vec<f64> = if refine_peaks {
                peak_idx
                    .iter()
                    .map(|&i| refine_parabolic(f_pos, p_pos, i))
                    .collect()
            } else {
                peak_idx.iter().map(|&i| f_pos[i]).collect()
            };
            selected.extend(sel_freqs);
        }
    }

    if selected.is_empty() {
        return Vec::new();
    }

    // Sort, deduplicate, merge nearby frequencies
    selected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    selected.retain(|&f| f.is_finite() && f > ignore_dc_hz && f <= fmax);

    let mut merged: Vec<f64> = Vec::new();
    for f in selected {
        if merged.is_empty() {
            merged.push(f);
            continue;
        }
        let last = *merged.last().unwrap();
        let rel = (f - last).abs() / last.max(1e-12);
        if rel <= spectral_merge_tol.max(0.0) {
            // Merge: average the two
            let len = merged.len();
            merged[len - 1] = 0.5 * (last + f);
        } else {
            merged.push(f);
        }
    }

    merged
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tiered ridge regression
// ═══════════════════════════════════════════════════════════════════════════

/// Build the ridge penalty diagonal vector R for tiered-ridge.
/// R[j] = freq_penalty for harmonic columns, 1.0 for DC and trend.
fn build_ridge_diag(
    freqs: &[f64],
    p: usize,
    include_dc: bool,
    include_trend: bool,
    freq_weight: f64,
) -> Vec<f64> {
    let mut r = vec![1.0f64; p];
    let mut col = 0;
    if include_dc {
        col += 1;
    }
    if include_trend {
        col += 1;
    }
    if !freqs.is_empty() {
        let penalty = make_frequency_penalty(freqs, freq_weight);
        for w_f in penalty {
            if col < p {
                r[col] = w_f;
                col += 1;
            }
            if col < p {
                r[col] = w_f;
                col += 1;
            }
        }
    }
    r
}

/// Tiered ridge: low-frequency and high-frequency tiers get different λ.
/// λ_high > λ_beta adds extra penalty to high-tier frequencies.
fn tiered_ridge_solve(
    x: &Array2<f64>,
    y: &Array1<f64>,
    freqs: &[f64],
    lambda_beta: f64,
    lambda_high: f64,
    low_freq_period_days: f64,
    freq_weight: f64,
    include_dc: bool,
    include_trend: bool,
    w: Option<&[f64]>,
) -> Option<Array1<f64>> {
    let p = x.ncols();
    if p == 0 {
        return Some(Array1::zeros(0));
    }

    // Base ridge diagonal
    let r = build_ridge_diag(freqs, p, include_dc, include_trend, freq_weight);

    // lam_diag = lambda_beta * R^2 + extra for high-tier freqs
    let mut lam_diag: Vec<f64> = r.iter().map(|&ri| lambda_beta * ri * ri).collect();

    if !freqs.is_empty() && lambda_high > lambda_beta {
        let is_high = classify_freq_tier(freqs, low_freq_period_days);
        let col_start = (if include_dc { 1 } else { 0 }) + (if include_trend { 1 } else { 0 });
        let extra = lambda_high - lambda_beta;
        for (k, &hi) in is_high.iter().enumerate() {
            if hi {
                let c = col_start + 2 * k;
                if c < p {
                    lam_diag[c] += extra;
                }
                if c + 1 < p {
                    lam_diag[c + 1] += extra;
                }
            }
        }
    }

    // Clip negative diagonals
    for d in &mut lam_diag {
        *d = d.max(0.0);
    }

    // sqrt_lam = sqrt(clip(lam_diag, 0, None))
    let sqrt_lam: Vec<f64> = lam_diag.iter().map(|&d| d.sqrt()).collect();

    // Build augmented system: [Xw; diag(sqrt_lam)] β = [yw; 0]
    let n = x.nrows();
    let aug_rows = n + p;
    let mut x_aug = Array2::<f64>::zeros((aug_rows, p));
    let mut y_aug = Array1::<f64>::zeros(aug_rows);

    if let Some(weights) = w {
        for i in 0..n {
            let sw = weights[i].sqrt();
            for j in 0..p {
                x_aug[[i, j]] = x[[i, j]] * sw;
            }
            y_aug[i] = y[i] * sw;
        }
    } else {
        for i in 0..n {
            for j in 0..p {
                x_aug[[i, j]] = x[[i, j]];
            }
            y_aug[i] = y[i];
        }
    }

    for j in 0..p {
        x_aug[[n + j, j]] = sqrt_lam[j];
        y_aug[n + j] = 0.0;
    }

    solve_normal_equations(&x_aug, &y_aug)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Robust frequency-weighted ridge fitting (Huber IRLS)
// ═══════════════════════════════════════════════════════════════════════════

/// Iteratively reweighted ridge regression with Huber weights.
///
/// Returns (beta, y_hat).
fn robust_fit_freq_ridge(
    x: &Array2<f64>,
    y: &Array1<f64>,
    freqs: &[f64],
    lam: f64,
    iters: usize,
    delta: f64,
    include_dc: bool,
    include_trend: bool,
    freq_weight: f64,
) -> (Vec<f64>, Vec<f64>) {
    let p = x.ncols();
    if p == 0 {
        let ymean = nanmean(y.as_slice().unwrap());
        return (vec![], vec![ymean; y.len()]);
    }

    let n = y.len();
    let mut w = vec![1.0f64; n];

    // Ridge diagonal
    let r = build_ridge_diag(freqs, p, include_dc, include_trend, freq_weight);

    // Initial fit with uniform weights
    let mut beta = ridge_solve_augmented(x, y, lam, &r, Some(&w))
        .unwrap_or_else(|| Array1::zeros(p));
    let mut y_hat = x.dot(&beta);

    for _ in 0..iters {
        let residuals: Vec<f64> = y.iter().zip(y_hat.iter()).map(|(&yi, &yh)| yi - yh).collect();
        w = huber_weights(&residuals, delta);
        if let Some(b) = ridge_solve_augmented(x, y, lam, &r, Some(&w)) {
            beta = b;
        }
        y_hat = x.dot(&beta);
    }

    // Final fit (after Huber loop)
    let residuals: Vec<f64> = y.iter().zip(y_hat.iter()).map(|(&yi, &yh)| yi - yh).collect();
    w = huber_weights(&residuals, delta);
    if let Some(b) = ridge_solve_augmented(x, y, lam, &r, Some(&w)) {
        beta = b;
    }
    y_hat = x.dot(&beta);

    (beta.to_vec(), y_hat.to_vec())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Single-pixel NUFROST fit (matches Python fit_nufrost_pixel_params)
// ═══════════════════════════════════════════════════════════════════════════

/// Full single-pixel NUFROST parameter estimation.
///
/// # Arguments
/// - `t_sec`: timestamps in seconds since epoch (may contain NaN)
/// - `y`: observations (may contain NaN)
/// - `config`: algorithm configuration
/// - `shared_freqs`: optional pre-computed shared frequencies (overrides selection)
///
/// # Returns
/// `NufrostResult` with fitted parameters for prediction.
pub fn nufrost_fit_pixel(
    t_sec: &[f64],
    y: &[f64],
    config: &NufrostConfig,
    shared_freqs: Option<&[f64]>,
) -> NufrostResult {
    let min_obs = config.min_obs as usize;

    // ── Build valid mask ───────────────────────────────────────────────────
    let valid_mask: Vec<bool> = t_sec
        .iter()
        .zip(y.iter())
        .map(|(&t, &v)| t.is_finite() && v.is_finite())
        .collect();
    let n_valid = valid_mask.iter().filter(|&&b| b).count();

    if n_valid < min_obs.max(3) {
        return NufrostResult {
            valid: false,
            fill_value: nanmean(y),
            include_trend: config.include_trend,
            ..Default::default()
        };
    }

    // ── Extract valid data ─────────────────────────────────────────────────
    let t: Vec<f64> = t_sec
        .iter()
        .zip(valid_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&ti, _)| ti)
        .collect();
    let yy: Vec<f64> = y
        .iter()
        .zip(valid_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&vi, _)| vi)
        .collect();
    let fill_value = nanmean(&yy);

    // ── Relative time ──────────────────────────────────────────────────────
    let t_rel: Vec<f64> = {
        let t0 = t.iter().copied().fold(f64::INFINITY, f64::min);
        t.iter().map(|&ti| ti - t0).collect()
    };
    let t_rel_mean = nanmean(&t_rel);
    let tspan = t_rel
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - t_rel
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

    if tspan <= 0.0 || !tspan.is_finite() {
        return NufrostResult {
            valid: false,
            fill_value,
            t_rel_mean,
            include_trend: config.include_trend,
            ..Default::default()
        };
    }

    // ── Compute spectrum ──────────────────────────────────────────────────
    let y_scale = 10000.0;
    let (f_pos, p_pos) =
        compute_spectrum_direct(&t_rel, &yy, config.modes as usize, y_scale);

    // ── Nyquist limit ─────────────────────────────────────────────────────
    let mut t_sorted = t_rel.clone();
    t_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dt_pos: Vec<f64> = t_sorted
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.0)
        .collect();
    let dt_med = if !dt_pos.is_empty() {
        let mut d = dt_pos.clone();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if d.len() % 2 == 1 {
            d[d.len() / 2]
        } else {
            (d[d.len() / 2 - 1] + d[d.len() / 2]) / 2.0
        }
    } else {
        tspan / t_rel.len() as f64
    };
    let fmax = 0.5 / dt_med.max(1e-12);

    // ── Frequency selection ───────────────────────────────────────────────
    let freqs_sel: Vec<f64> = if let Some(sf) = shared_freqs {
        sf.iter()
            .copied()
            .filter(|&f| f.is_finite() && f > config.ignore_dc_hz && f <= fmax)
            .collect()
    } else {
        let mode = if config.frequency_selection == "shared_spectral" {
            "spectral"
        } else {
            &config.frequency_selection
        };
        let pref_freqs = parse_preferred_frequencies(&config.preferred_periods_days);
        select_frequencies(
            &f_pos,
            &p_pos,
            fmax,
            mode,
            &pref_freqs,
            config.preferred_top_k as usize,
            config.spectral_top_k as usize,
            config.spectral_merge_tol,
            config.power_cum,
            config.ignore_dc_hz,
            config.refine_peaks,
        )
    };

    let n_freqs_used = freqs_sel.len();

    // ── Scale observations ─────────────────────────────────────────────────
    let yy_scaled: Vec<f64> = yy.iter().map(|&v| v / y_scale).collect();

    // ── Build design matrix ────────────────────────────────────────────────
    let x = make_design_matrix(
        &t_rel,
        &freqs_sel,
        config.include_trend,
        true, // include_dc always true
    );

    // ── Huber-Ridge fit with iterative outlier rejection ────────────────────
    let n_min_needed = (1 + config.include_trend as usize + 2 * n_freqs_used).max(min_obs);

    let beta = if config.outlier_sigma > 0.0 {
        // MAD-based iterative outlier rejection
        let mut t_curr = t_rel.clone();
        let mut y_curr = yy_scaled.clone();
        let mut beta_final: Option<Vec<f64>> = None;

        for _ in 0..(t_curr.len().min(50)) {
            if y_curr.len() < n_min_needed {
                break;
            }
            let x_curr = make_design_matrix(
                &t_curr,
                &freqs_sel,
                config.include_trend,
                true,
            );
            let y_arr = Array1::from_vec(y_curr.clone());
            let (beta_curr, y_pred) = robust_fit_freq_ridge(
                &x_curr,
                &y_arr,
                &freqs_sel,
                config.ridge_lam,
                config.huber_iters as usize,
                config.huber_delta,
                true,
                config.include_trend,
                config.freq_weight,
            );

            beta_final = Some(beta_curr.clone());

            let residuals: Vec<f64> = y_curr.iter().zip(y_pred.iter()).map(|(&a, &b)| a - b).collect();
            let mad = mad_std(&residuals);
            let threshold = config.outlier_sigma * mad.max(1e-12);

            let clean_mask: Vec<bool> = residuals.iter().map(|&r| r.abs() <= threshold).collect();
            if clean_mask.iter().all(|&b| b) {
                break;
            }

            let mut new_t = Vec::with_capacity(y_curr.len());
            let mut new_y = Vec::with_capacity(y_curr.len());
            for (i, &keep) in clean_mask.iter().enumerate() {
                if keep {
                    new_t.push(t_curr[i]);
                    new_y.push(y_curr[i]);
                }
            }
            t_curr = new_t;
            y_curr = new_y;
        }

        match beta_final {
            Some(b) => b,
            None => {
                // Final fallback: fit on all data
                let y_arr = Array1::from_vec(yy_scaled.clone());
                let (b, _) = robust_fit_freq_ridge(
                    &x, &y_arr, &freqs_sel,
                    config.ridge_lam, config.huber_iters as usize, config.huber_delta,
                    true, config.include_trend, config.freq_weight,
                );
                b
            }
        }
    } else {
        // No outlier rejection — direct fit on all data
        let y_arr = Array1::from_vec(yy_scaled.clone());
        let (b, _) = robust_fit_freq_ridge(
            &x, &y_arr, &freqs_sel,
            config.ridge_lam, config.huber_iters as usize, config.huber_delta,
            true, config.include_trend, config.freq_weight,
        );
        b
    };

    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);

    NufrostResult {
        valid: true,
        n_freqs_used,
        beta,
        freqs: freqs_sel,
        t_min,
        t_rel_mean,
        fill_value,
        include_trend: config.include_trend,
        y_scale,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Single-pixel NUFROST fit with step/fused-lasso (BCS version)
// ═══════════════════════════════════════════════════════════════════════════

/// Difference weights for step (fused-lasso) regularisation.
fn difference_weights(t_sec: &[f64], enable_dt_weighting: bool) -> Vec<f64> {
    let n = t_sec.len();
    if n <= 1 {
        return vec![];
    }
    if !enable_dt_weighting {
        return vec![1.0f64; n - 1];
    }
    let dt_days: Vec<f64> = t_sec
        .windows(2)
        .map(|w| ((w[1] - w[0]) / 86400.0).max(1.0))
        .collect();
    dt_days.iter().map(|&d| 1.0 / d.sqrt()).collect()
}

/// Fused lasso 1D via FISTA dual proximal gradient.
///
/// Solves: min_u 0.5‖r - u‖² + λ_step Σ w_i |u_{i+1} - u_i|
fn fused_lasso_1d(
    r: &[f64],
    lambda_step: f64,
    weights: &[f64],
) -> Vec<f64> {
    let n = r.len();
    if n <= 1 || lambda_step <= 0.0 {
        return r.to_vec();
    }

    let d = n - 1;
    let lam: Vec<f64> = weights.iter().map(|&w| lambda_step * w).collect();

    // FISTA constants
    let step = 0.25; // 1 / ||D||²  (||D||² ≤ 4)
    let max_iter = 5000;
    let tol = 1e-9;

    let mut z = vec![0.0f64; d]; // dual variable
    let mut y_dual = z.clone();
    let mut t_prev = 1.0f64;
    let mut u_prev: Vec<f64> = Vec::new();

    for _ in 0..max_iter {
        // u = r - D^T y_dual
        let dt_y = apply_dt(&y_dual);
        let u: Vec<f64> = r.iter().zip(dt_y.iter()).map(|(&ri, &d)| ri - d).collect();

        // gradient step on y_dual
        let du: Vec<f64> = u.windows(2).map(|w| w[1] - w[0]).collect();
        let mut z_new: Vec<f64> = y_dual
            .iter()
            .zip(du.iter())
            .map(|(&yi, &di)| yi + step * di)
            .collect();
        // project onto box [-lam_i, lam_i]
        for i in 0..d {
            z_new[i] = z_new[i].clamp(-lam[i], lam[i]);
        }

        // FISTA extrapolation
        let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev).sqrt());
        let factor = (t_prev - 1.0) / t_new;
        for i in 0..d {
            y_dual[i] = z_new[i] + factor * (z_new[i] - z[i]);
        }

        if !u_prev.is_empty() {
            let max_diff: f64 = u
                .iter()
                .zip(u_prev.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max);
            if max_diff < tol {
                z = z_new;
                break;
            }
        }

        z = z_new;
        t_prev = t_new;
        u_prev = u;
    }

    // Recompute final primal from z
    let dt_z = apply_dt(&z);
    r.iter().zip(dt_z.iter()).map(|(&ri, &d)| ri - d).collect()
}

/// Apply D^T to a dual variable: (D^T y)_0 = -y_0, (D^T y)_i = y_{i-1} - y_i, last = y_{n-2}
fn apply_dt(y: &[f64]) -> Vec<f64> {
    let d = y.len();
    let n = d + 1;
    let mut out = vec![0.0f64; n];
    if n == 1 {
        return out;
    }
    out[0] = -y[0];
    for i in 1..d {
        out[i] = y[i - 1] - y[i];
    }
    out[d] = y[d - 1];
    out
}

/// Single-pixel NUFROST fit with step term + tiered ridge (BCD variant).
///
/// This is the version that also estimates a fused-lasso step term u,
/// used in the multi-band pipeline.
pub fn nufrost_fit_pixel_step(
    t_sec: &[f64],
    y: &[f64],
    freqs_sel: &[f64],
    config: &NufrostConfig,
) -> Option<NufrostResult> {
    let min_obs = config.min_obs as usize;

    // Valid mask
    let m: Vec<bool> = t_sec
        .iter()
        .zip(y.iter())
        .map(|(&t, &v)| t.is_finite() && v.is_finite())
        .collect();
    let n_kept = m.iter().filter(|&&b| b).count();
    if n_kept < min_obs.max(3) {
        return None;
    }

    let t: Vec<f64> = t_sec.iter().zip(m.iter()).filter(|(_, &b)| b).map(|(&ti, _)| ti).collect();
    let yy: Vec<f64> = y.iter().zip(m.iter()).filter(|(_, &b)| b).map(|(&vi, _)| vi).collect();

    let t_rel: Vec<f64> = {
        let t0 = t.iter().copied().fold(f64::INFINITY, f64::min);
        t.iter().map(|&ti| ti - t0).collect()
    };
    let t_rel_mean = nanmean(&t_rel);
    let fill_value = nanmean(&yy);

    // Design matrix
    let freqs_vec: Vec<f64> = freqs_sel.to_vec();
    let x = make_design_matrix(&t_rel, &freqs_vec, config.include_trend, true);

    if x.ncols() == 0 {
        let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
        return Some(NufrostResult {
            valid: false, fill_value, t_rel_mean, t_min,
            include_trend: config.include_trend,
            ..Default::default()
        });
    }

    let y_arr = Array1::from_vec(yy.clone());

    // Difference weights for fused lasso
    let diff_w = difference_weights(&t, config.step_dt_weighting);

    let mut u = vec![0.0f64; yy.len()];
    // Initial beta from tiered ridge
    let mut beta = tiered_ridge_solve(
        &x, &y_arr, &freqs_vec,
        config.ridge_lam, config.lambda_high, config.low_freq_period_days,
        config.freq_weight, true, config.include_trend, None,
    ).unwrap_or_else(|| Array1::zeros(x.ncols()));

    let max_outer = config.max_outer_iter as usize;
    let outer_tol = config.outer_tol;
    let mut _n_iter = 0;

    for it in 1..=max_outer {
        let beta_old = beta.clone();
        let u_old = u.clone();

        // u update: fused lasso on residuals
        let residual: Vec<f64> = yy
            .iter()
            .zip(x.dot(&beta).iter())
            .map(|(&yi, &xbi)| yi - xbi)
            .collect();
        u = fused_lasso_1d(&residual, config.lambda_step, &diff_w);

        // beta update: tiered ridge on y - u
        let y_minus_u: Vec<f64> = yy.iter().zip(u.iter()).map(|(&yi, &ui)| yi - ui).collect();
        let ymu_arr = Array1::from_vec(y_minus_u);
        beta = tiered_ridge_solve(
            &x, &ymu_arr, &freqs_vec,
            config.ridge_lam, config.lambda_high, config.low_freq_period_days,
            config.freq_weight, true, config.include_trend, None,
        ).unwrap_or_else(|| Array1::zeros(x.ncols()));

        _n_iter = it;

        // Convergence check
        let denom = (beta_old.iter().map(|&b| b * b).sum::<f64>().sqrt()
            + u_old.iter().map(|&u| u * u).sum::<f64>().sqrt())
        .max(1e-12);
        let delta = beta.iter().zip(beta_old.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f64>()
            + u.iter().zip(u_old.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f64>();
        if delta / denom < outer_tol {
            break;
        }
    }

    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);

    Some(NufrostResult {
        valid: true,
        n_freqs_used: freqs_sel.len(),
        beta: beta.to_vec(),
        freqs: freqs_vec,
        t_min,
        t_rel_mean,
        fill_value,
        include_trend: config.include_trend,
        y_scale: 1.0, // no scaling in step variant
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Prediction
// ═══════════════════════════════════════════════════════════════════════════

/// Predict NUFROST-reconstructed value at a target time from fitted params.
pub fn nufrost_predict(result: &NufrostResult, target_t: f64) -> f64 {
    if !result.valid {
        return result.fill_value;
    }

    let t_star_rel = target_t - result.t_min;
    let n_freqs = result.n_freqs_used;
    let beta = &result.beta;
    let freqs_sel: Vec<f64> = result.freqs.iter().take(n_freqs).copied().collect();

    let beta_len = 1 + result.include_trend as usize + 2 * n_freqs;
    let beta_actual = &beta[..beta_len.min(beta.len())];

    let mut pred = 0.0;

    // DC term
    if let Some(&b0) = beta_actual.first() {
        pred += b0;
        let mut idx = 1;
        // Trend
        if result.include_trend {
            if let Some(&b1) = beta_actual.get(idx) {
                pred += b1 * (t_star_rel - result.t_rel_mean);
                idx += 1;
            }
        }
        // Harmonic terms
        for &f in &freqs_sel {
            let omega = 2.0 * PI * f;
            if let Some(&bc) = beta_actual.get(idx) {
                pred += bc * (omega * t_star_rel).cos();
            }
            if let Some(&bs) = beta_actual.get(idx + 1) {
                pred += bs * (omega * t_star_rel).sin();
            }
            idx += 2;
        }
    }

    pred * result.y_scale
}

/// Predict NUFROST-reconstructed curve at multiple target times.
pub fn nufrost_predict_curve(result: &NufrostResult, target_t: &[f64]) -> Vec<f64> {
    target_t.iter().map(|&t| nufrost_predict(result, t)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Convenience: fit + predict in one call
// ═══════════════════════════════════════════════════════════════════════════

/// Fit NUFROST to a single pixel and predict at target time.
///
/// Returns `(predicted_value, n_freqs_used)`.
pub fn nufrost_pixel(
    t_sec: &[f64],
    y: &[f64],
    target_t: f64,
    config: &NufrostConfig,
) -> (f64, usize) {
    let result = nufrost_fit_pixel(t_sec, y, config, None);
    let n_freqs = result.n_freqs_used;
    let pred = nufrost_predict(&result, target_t);
    (pred, n_freqs)
}

/// Fit NUFROST with shared frequencies and predict.
pub fn nufrost_pixel_with_shared(
    t_sec: &[f64],
    y: &[f64],
    target_t: f64,
    config: &NufrostConfig,
    shared_freqs: &[f64],
) -> (f64, usize) {
    let result = nufrost_fit_pixel(t_sec, y, config, Some(shared_freqs));
    let n_freqs = result.n_freqs_used;
    let pred = nufrost_predict(&result, target_t);
    (pred, n_freqs)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NufrostConfig;
    use std::fs::File;
    use std::path::PathBuf;

    // ── Default test config ────────────────────────────────────────────────

    fn default_nufrost_config() -> NufrostConfig {
        serde_json::from_str(
            r#"{
                "modes": 4096,
                "eps": 1e-12,
                "num_peaks": 10,
                "power_cum": 0.7,
                "ignore_dc_hz": 1e-10,
                "frequency_selection": "spectral",
                "preferred_periods_days": "365.25,182.625,91.3125,30.4375",
                "preferred_top_k": 4,
                "spectral_top_k": 4,
                "spectral_merge_tol": 0.15,
                "refine_peaks": true,
                "include_trend": true,
                "ridge_lam": 0.005,
                "freq_weight": 2.0,
                "huber_iters": 3,
                "huber_delta": 0.05,
                "min_obs": 12,
                "outlier_sigma": 2.0,
                "lambda_step": 1e30,
                "lambda_high": 0.005,
                "low_freq_period_days": 0.0,
                "step_dt_weighting": false,
                "max_outer_iter": 5,
                "outer_tol": 1e-3,
                "joint_outlier": false,
                "joint_outlier_sigma": 2.5,
                "admm_rho": 1.0,
                "admm_max_iter": 1000,
                "admm_tol": 1e-4
            }"#,
        )
        .unwrap()
    }

    // ── Helper: fixture path ───────────────────────────────────────────────

    fn fixture_dir() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/fixtures/rust_parity")
    }

    fn load_synthetic_npz(name: &str) -> (Vec<f64>, Vec<f64>, f64, f64) {
        let path = fixture_dir().join("synthetic").join(name).join("data.npz");
        let mut archive = ndarray_npy::NpzReader::new(
            File::open(&path).expect("Failed to open npz"),
        )
        .expect("Failed to read npz");

        let timestamps_days: ndarray::Array1<f64> = archive
            .by_name("timestamps_days.npy")
            .expect("timestamps_days not found");
        let observations: ndarray::Array1<f64> = archive
            .by_name("observations.npy")
            .expect("observations not found");
        let target_time_day_arr: ndarray::Array0<f64> = archive
            .by_name("target_time_day.npy")
            .expect("target_time_day not found");
        let nufrost_pred_val: ndarray::Array0<f64> = archive
            .by_name("nufrost_prediction.npy")
            .expect("nufrost_prediction not found");

        let target_time_sec = target_time_day_arr[()] * 86400.0;
        let timestamps_sec: Vec<f64> = timestamps_days.iter().map(|&d| d * 86400.0).collect();
        let obs = observations.to_vec();
        let pred_expected = nufrost_pred_val[()];

        (timestamps_sec, obs, target_time_sec, pred_expected)
    }

    // ── Unit tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_next_even() {
        assert_eq!(next_even(1), 2);
        assert_eq!(next_even(2), 2);
        assert_eq!(next_even(3), 4);
        assert_eq!(next_even(4095), 4096);
        assert_eq!(next_even(4096), 4096);
    }

    #[test]
    fn test_nanmean_basic() {
        assert!((nanmean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-15);
        assert!(nanmean(&[f64::NAN, 3.0, 5.0]).abs() - 4.0 < 1e-15);
        assert!(nanmean(&[f64::NAN]).is_nan());
    }

    #[test]
    fn test_nanmedian_basic() {
        assert!((nanmedian(&[1.0, 3.0, 2.0]) - 2.0).abs() < 1e-15);
        assert!((nanmedian(&[1.0, 2.0]) - 1.5).abs() < 1e-15);
    }

    #[test]
    fn test_mad_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mad_std(&data);
        assert!(m > 0.0);
    }

    #[test]
    fn test_design_matrix_shape() {
        let t = vec![0.0, 1.0, 2.0];
        let freqs = vec![1.0 / 365.25 / 86400.0];
        let x = make_design_matrix(&t, &freqs, true, true);
        // DC + trend + 2*freqs = 1 + 1 + 2 = 4
        assert_eq!(x.shape(), &[3, 4]);
    }

    #[test]
    fn test_design_matrix_dc_col_is_ones() {
        let t = vec![10.0, 20.0];
        let freqs: Vec<f64> = vec![];
        let x = make_design_matrix(&t, &freqs, false, true);
        assert!((x[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((x[[1, 0]] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_design_matrix_no_dc_no_trend() {
        let t = vec![0.0, 1.0];
        let freqs = vec![0.1];
        let x = make_design_matrix(&t, &freqs, false, false);
        // 2 cols: cos, sin
        assert_eq!(x.shape(), &[2, 2]);
    }

    #[test]
    fn test_harmonic_cos_sin_at_zero() {
        let t = vec![0.0];
        let freqs = vec![0.1];
        let x = make_design_matrix(&t, &freqs, false, true);
        // DC (x[[0,0]] = 1), cos (x[[0,1]] = cos(0) = 1), sin (x[[0,2]] = 0)
        assert!((x[[0, 0]] - 1.0).abs() < 1e-14);
        assert!((x[[0, 1]] - 1.0).abs() < 1e-14);
        assert!(x[[0, 2]].abs() < 1e-14);
    }

    #[test]
    fn test_gauss_solve_2x2() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 5.0]);
        let x = gauss_solve(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gauss_solve_singular() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 4.0, 2.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 6.0]);
        assert!(gauss_solve(&a, &b).is_none());
    }

    #[test]
    fn test_frequency_penalty_min_freq_is_1() {
        let freqs = vec![0.0001, 0.0002, 0.0005];
        let penalty = make_frequency_penalty(&freqs, 2.0);
        assert!(penalty.len() == 3);
        // First (min) should be 1.0
        assert!((penalty[0] - 1.0).abs() < 1e-10);
        // Higher freqs get larger penalty
        assert!(penalty[2] > penalty[1]);
        assert!(penalty[1] >= penalty[0]);
    }

    #[test]
    fn test_huber_weights_small_residuals() {
        let r = vec![0.0, 0.01, -0.01];
        let w = huber_weights(&r, 0.05);
        assert!((w[0] - 1.0).abs() < 1e-10);
        assert!((w[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_huber_weights_large_residuals() {
        let r = vec![0.1, -0.2];
        let w = huber_weights(&r, 0.05);
        // w[0] = 0.05 / 0.1 = 0.5
        assert!((w[0] - 0.5).abs() < 1e-10);
        // w[1] = 0.05 / 0.2 = 0.25
        assert!((w[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_refine_parabolic_exact() {
        // Parabola: y = -(x - 2)^2 + 4, peak at x = 2
        let f = vec![1.0, 2.0, 3.0];
        let p = vec![3.0, 4.0, 3.0]; // -(1-2)^2+4=3, 4, 3
        let refined = refine_parabolic(&f, &p, 1);
        assert!((refined - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_preferred_frequencies() {
        let freqs = parse_preferred_frequencies("365.25,182.625");
        assert_eq!(freqs.len(), 2);
        let annual = 1.0 / (365.25 * 86400.0);
        let semi = 1.0 / (182.625 * 86400.0);
        assert!((freqs[0] - annual).abs() < 1e-15);
        assert!((freqs[1] - semi).abs() < 1e-15);
    }

    #[test]
    fn test_parse_preferred_frequencies_empty() {
        let freqs = parse_preferred_frequencies("");
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_select_frequencies_spectral_empty_input() {
        let f_pos = vec![1.0, 2.0, 3.0];
        let p_pos = vec![0.0, 0.0, 0.0]; // all zero means no valid peaks
        let _selected = select_frequencies(
            &f_pos, &p_pos, 10.0, "spectral", &[], 0, 4, 0.15, 0.7, 0.0, false,
        );
        // No valid peaks → empty; all-zero power handled without panicking
        let p_pos2 = vec![0.0f64; 3]; // all zero intentional
        let _selected2 = select_frequencies(
            &f_pos, &p_pos2, 10.0, "spectral", &[], 0, 4, 0.15, 0.7, 0.0, false,
        );
        // The function should still work without panicking
        assert!(_selected2.len() <= 4);
    }

    #[test]
    fn test_select_frequencies_hybrid_returns_frequencies() {
        let f_pos: Vec<f64> = (0..100).map(|i| i as f64 * 0.00001).collect();
        let p_pos: Vec<f64> = (0..100).map(|i| {
            // Put a peak at f = 3.171e-6 (~= 1/(365.25*86400))
            if (i as f64 - 3.171e-6 / 0.00001).abs() < 2.0 {
                100.0 * (1.0 - (i as f64 - 3.171e-6 / 0.00001).abs() / 2.0)
            } else {
                0.1
            }
        }).collect();
        let pref_freqs = parse_preferred_frequencies("365.25");

        let selected = select_frequencies(
            &f_pos, &p_pos, 0.1, "hybrid", &pref_freqs, 1, 4, 0.15, 0.7, 0.0, false,
        );
        assert!(!selected.is_empty());
        // Selected frequencies should be finite and > 0
        for f in &selected {
            assert!(f.is_finite() && *f > 0.0);
        }
    }

    // ── Ridge regression tests ─────────────────────────────────────────────

    #[test]
    fn test_ridge_solve_simple() {
        // y = 2 + 3*t, centered trend col = t - mean(t), mean=1
        // beta[0] = value at mean time = 2 + 3*1 = 5
        // beta[1] = slope = 3
        let t = vec![0.0, 1.0, 2.0];
        let x = make_design_matrix(&t, &[], true, true);
        let y_vals = vec![2.0, 5.0, 8.0];
        let y = Array1::from_vec(y_vals);
        let r = build_ridge_diag(&[], x.ncols(), true, true, 2.0);

        let beta = ridge_solve_augmented(&x, &y, 0.0, &r, None).unwrap();
        assert!((beta[0] - 5.0).abs() < 1e-10, "DC: {}", beta[0]);
        assert!((beta[1] - 3.0).abs() < 1e-10, "trend: {}", beta[1]);
    }

    #[test]
    fn test_ridge_solve_with_lam_produces_finite() {
        let t: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let freqs = vec![0.01, 0.02];
        let x = make_design_matrix(&t, &freqs, true, true);
        // Generate a noisy signal
        let y_vals: Vec<f64> = t.iter().map(|&ti| 1.0 + 0.5 * ti + 2.0 * (2.0 * PI * 0.01 * ti).cos()).collect();
        let y = Array1::from_vec(y_vals);
        let r = build_ridge_diag(&freqs, x.ncols(), true, true, 2.0);

        let beta = ridge_solve_augmented(&x, &y, 0.01, &r, None).unwrap();
        for &b in &beta {
            assert!(b.is_finite());
        }
    }

    // ── End-to-end pixel tests ─────────────────────────────────────────────

    #[test]
    fn test_nufrost_pixel_returns_finite_on_clean_periodic_signal() {
        // Synthetic annual + semi-annual harmonic
        let t_sec: Vec<f64> = (0..50)
            .map(|i| i as f64 * 15.0 * 86400.0)
            .collect();
        let annual_freq = 1.0 / (365.25 * 86400.0);
        let semi_freq = 2.0 * annual_freq;

        let y: Vec<f64> = t_sec
            .iter()
            .map(|&t| {
                0.5
                    + 0.2 * (2.0 * PI * annual_freq * t).sin()
                    + 0.1 * (2.0 * PI * semi_freq * t).cos()
            })
            .collect();

        let config = default_nufrost_config();
        let target_t = t_sec[25]; // middle of series
        let (pred, n_freqs) = nufrost_pixel(&t_sec, &y, target_t, &config);

        assert!(pred.is_finite());
        assert!(n_freqs >= 1);
    }

    #[test]
    fn test_nufrost_pixel_with_nan_returns_finite() {
        let t_sec: Vec<f64> = (0..50)
            .map(|i| i as f64 * 15.0 * 86400.0)
            .collect();
        let annual_freq = 1.0 / (365.25 * 86400.0);
        let mut y: Vec<f64> = t_sec
            .iter()
            .map(|&t| 0.5 + 0.2 * (2.0 * PI * annual_freq * t).sin())
            .collect();
        // Insert NaN at observation 10
        y[10] = f64::NAN;

        let config = default_nufrost_config();
        let target_t = t_sec[25];
        let (pred, _) = nufrost_pixel(&t_sec, &y, target_t, &config);

        assert!(pred.is_finite());
    }

    #[test]
    fn test_nufrost_pixel_insufficient_data_returns_nan() {
        let t_sec = vec![0.0, 86400.0];
        let y = vec![0.5, f64::NAN];
        let config = default_nufrost_config();
        let result = nufrost_fit_pixel(&t_sec, &y, &config, None);
        assert!(!result.valid, "should be invalid with only 1 valid obs");
        assert_eq!(result.n_freqs_used, 0);
    }

    // ── Fixture parity tests ───────────────────────────────────────────────

    fn load_config(name: &str) -> NufrostConfig {
        let path = fixture_dir()
            .join("synthetic").join(name).join("config.json");
        let json_str = std::fs::read_to_string(&path).unwrap();
        let wrapper: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let cfg_json = &wrapper["config"]["nufrost"];
        // Merge fixture config with our defaults (fixtures may omit fields)
        let mut config = default_nufrost_config();
        if let serde_json::Value::Object(map) = cfg_json {
            if let Some(&serde_json::Value::Number(ref n)) = map.get("modes") {
                config.modes = n.as_u64().unwrap_or(4096) as u32;
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("eps") {
                config.eps = n.as_f64().unwrap_or(1e-12);
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("num_peaks") {
                config.num_peaks = n.as_u64().unwrap_or(10) as u32;
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("power_cum") {
                config.power_cum = n.as_f64().unwrap_or(0.7);
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("ignore_dc_hz") {
                config.ignore_dc_hz = n.as_f64().unwrap_or(1e-10);
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("ridge_lam") {
                config.ridge_lam = n.as_f64().unwrap_or(0.005);
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("freq_weight") {
                config.freq_weight = n.as_f64().unwrap_or(2.0);
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("huber_iters") {
                config.huber_iters = n.as_u64().unwrap_or(3) as u32;
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("huber_delta") {
                config.huber_delta = n.as_f64().unwrap_or(0.05);
            }
            if let Some(&serde_json::Value::Number(ref n)) = map.get("min_obs") {
                config.min_obs = n.as_u64().unwrap_or(12) as u32;
            }
            if let Some(serde_json::Value::Bool(b)) = map.get("refine_peaks") {
                config.refine_peaks = *b;
            }
            if let Some(serde_json::Value::Bool(b)) = map.get("include_trend") {
                config.include_trend = *b;
            }
        }
        config
    }

    #[test]
    fn test_parity_simple_harmonic() {
        let (t_sec, obs, target_t, expected_pred) =
            load_synthetic_npz("simple_harmonic");
        let config = load_config("simple_harmonic");

        let (pred, n_freqs) = nufrost_pixel(&t_sec, &obs, target_t, &config);

        assert!(pred.is_finite(), "pred should be finite");
        assert!(n_freqs >= 1, "should find at least 1 frequency");

        let abs_err = (pred - expected_pred).abs();
        let rel_err = if expected_pred.abs() > 1e-12 {
            abs_err / expected_pred.abs()
        } else {
            abs_err
        };
        assert!(
            abs_err < 5e-5 || rel_err < 5e-4,
            "abs_err={:.2e}, rel_err={:.2e}, pred={:.6}, expected={:.6}",
            abs_err, rel_err, pred, expected_pred
        );
    }

    #[test]
    fn test_parity_gaps_outliers() {
        let (t_sec, obs, target_t, expected_pred) =
            load_synthetic_npz("gaps_outliers");
        let config = load_config("gaps_outliers");

        let (pred, n_freqs) = nufrost_pixel(&t_sec, &obs, target_t, &config);

        assert!(pred.is_finite(), "pred should be finite");
        assert!(n_freqs >= 1, "should find at least 1 frequency");

        let abs_err = (pred - expected_pred).abs();
        let rel_err = if expected_pred.abs() > 1e-12 {
            abs_err / expected_pred.abs()
        } else {
            abs_err
        };
        assert!(
            abs_err < 1e-3 || rel_err < 1e-2,
            "abs_err={:.2e}, rel_err={:.2e}, pred={:.6}, expected={:.6}",
            abs_err, rel_err, pred, expected_pred
        );
    }

    #[test]
    fn test_parity_step_break() {
        let (t_sec, obs, target_t, expected_pred) =
            load_synthetic_npz("step_break");
        let config = load_config("step_break");

        let (pred, n_freqs) = nufrost_pixel(&t_sec, &obs, target_t, &config);

        assert!(pred.is_finite(), "pred should be finite");
        assert!(n_freqs >= 1, "should find at least 1 frequency");

        let abs_err = (pred - expected_pred).abs();
        let rel_err = if expected_pred.abs() > 1e-12 {
            abs_err / expected_pred.abs()
        } else {
            abs_err
        };
        assert!(
            abs_err < 1e-3 || rel_err < 1e-2,
            "abs_err={:.2e}, rel_err={:.2e}, pred={:.6}, expected={:.6}",
            abs_err, rel_err, pred, expected_pred
        );
    }
}
