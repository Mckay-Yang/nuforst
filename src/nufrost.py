import os
import math
from typing import Optional, Tuple, Sequence, Union, cast
from datetime import datetime
import numpy as np
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import finufft  # type: ignore
except ModuleNotFoundError:
    raise ModuleNotFoundError("finufft is required. Install with: pip install finufft")

try:
    from joblib import Parallel, delayed, cpu_count  # type: ignore
    JOBLIB_AVAILABLE = True
except ModuleNotFoundError:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore
    cpu_count = None  # type: ignore
    JOBLIB_AVAILABLE = False

try:
    from tqdm import tqdm  # type: ignore
    TQDM_AVAILABLE = True
except ModuleNotFoundError:
    tqdm = None  # type: ignore
    TQDM_AVAILABLE = False

import rasterio

from config import Args, build_args
from .data_loader import RSCube

def _to_seconds_since_start(ts_utc: np.ndarray) -> np.ndarray:
    t0 = np.min(ts_utc)
    return np.ascontiguousarray(ts_utc - t0, dtype=np.float64)

def _parse_timestamp_str(ts: str) -> Optional[datetime]:
    if PANDAS_AVAILABLE:
        try:
            # 优先使用 pandas 解析，并统一转为 UTC，这与 bak/revive_test.py 逻辑一致
            dt = pd.to_datetime(ts, utc=True)
            return dt.to_pydatetime()
        except Exception:
            pass

    ts = str(ts).strip()
    fmts = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%dT%H%M%S",
        "%Y%m%d",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None

def _datetime64_to_seconds(ts: np.datetime64) -> float:
    epoch = np.datetime64("1970-01-01T00:00:00")
    return float((ts - epoch) / np.timedelta64(1, "s"))

def timestamps_to_seconds(timestamps: np.ndarray, unit: str = "seconds") -> np.ndarray:
    out = []
    for ts in timestamps:
        if isinstance(ts, datetime):
            out.append(ts.timestamp())
            continue
        if isinstance(ts, np.datetime64):
            out.append(_datetime64_to_seconds(ts))
            continue
        dt = _parse_timestamp_str(str(ts))
        if dt is None:
            out.append(np.nan)
            continue
        out.append(dt.timestamp())
    arr = np.array(out, dtype=np.float64)
    if unit == "days":
        arr = arr / 86400.0
    return arr

def parse_timestamp_str(ts: str) -> Optional[datetime]:
    return _parse_timestamp_str(ts)

def next_even(n: int) -> int:
    return int(np.ceil(n/2.0))*2

def refine_parabolic(f: np.ndarray, P: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(P)-1:
        return float(f[i])
    y0,y1,y2 = P[i-1],P[i],P[i+1]
    denom = (y0 - 2*y1 + y2)
    if denom == 0:
        return float(f[i])
    delta = 0.5*(y0 - y2)/denom
    return float(f[i] + delta*(f[i+1] - f[i]))

def select_peaks_adaptive(f_pos: np.ndarray, P_pos: np.ndarray, k_max: int, power_cum: float, ignore_dc_hz: float, fmax: float) -> np.ndarray:
    lower = max(ignore_dc_hz, 0.0)
    if not np.isfinite(fmax) or fmax <= 0:
        fmax = np.nanmax(f_pos[np.isfinite(f_pos)]) or 1.0
    valid = np.isfinite(f_pos) & np.isfinite(P_pos) & (f_pos > lower) & (f_pos <= fmax)
    if not np.any(valid):
        return np.array([], dtype=int)
    idx = np.where(valid)[0]
    order = np.argsort(-P_pos[idx])
    idx_sorted = idx[order]
    cum = np.cumsum(P_pos[idx_sorted])
    thr = np.clip(power_cum, 0.0, 1.0) * cum[-1]
    take = np.searchsorted(cum, thr) + 1
    take = min(take, k_max, len(idx_sorted))
    return idx_sorted[:take]

def design_matrix(t: np.ndarray, freqs: Union[Sequence[float], np.ndarray], include_trend: bool = True, include_dc: bool = True) -> np.ndarray:
    cols = []
    if include_dc:
        cols.append(np.ones_like(t))
    if include_trend:
        cols.append(t - t.mean())
    for f in freqs:
        w = 2*np.pi*f
        cols.append(np.cos(w*t))
        cols.append(np.sin(w*t))
    return np.vstack(cols).T if cols else np.empty((len(t),0))

def _classify_freq_tier(freqs: np.ndarray, low_freq_period_days: float,
                        time_unit_seconds: bool = True) -> np.ndarray:
    """Return boolean mask: True where frequency belongs to high-freq tier
    (period < low_freq_period_days). Zero frequencies are treated as low.

    Args:
        freqs: 1D array of frequencies in Hz (when time_unit_seconds=True)
            or in cycles/day (when False).
        low_freq_period_days: tier threshold in days.
        time_unit_seconds: True if `freqs` are Hz.
    """
    f = np.asarray(freqs, dtype=np.float64)
    if f.size == 0:
        return np.zeros(0, dtype=np.bool_)
    pos = f > 0.0
    period_days = np.full_like(f, np.inf)
    if time_unit_seconds:
        period_days[pos] = 1.0 / (f[pos] * 86400.0)
    else:
        period_days[pos] = 1.0 / f[pos]
    return period_days < float(low_freq_period_days)

def _tiered_ridge_solve(X: np.ndarray, y: np.ndarray,
                        freqs: np.ndarray,
                        lambda_beta: float, lambda_high: float,
                        low_freq_period_days: float,
                        freq_weight: float,
                        include_dc: bool, include_trend: bool,
                        time_unit_seconds: bool = True) -> np.ndarray:
    """Closed-form ridge with a two-tier penalty on frequency coefficients.

    Solves the normal equation
        (XᵀX + Λ) β = Xᵀ y
    with the diagonal penalty
        Λ_kk = λ_β · (W_freq)_kk² + λ_high · 1[k is a high-tier coef].

    `W_freq` here has diagonal entries:
      - 1.0  for the DC column,
      - 1.0  for the trend column (when present),
      - per-frequency penalty `_make_frequency_penalty(...)` for cos/sin pairs.

    Note this DC/trend-row choice (1.0) differs from the legacy
    `ridge_with_freq_weights`, which uses 0.0 there. Setting them to 1.0
    makes `λ_β W_freqᵀ W_freq` strictly positive definite, which §14.1
    of the design spec relies on. The numerical effect on DC/trend
    coefficients is a microscopic shrinkage proportional to `λ_β`.

    When `lambda_high == lambda_beta` and the DC/trend ridges are turned off
    (legacy mode), this reduces to a standard ridge. We do NOT replicate
    the legacy behavior here — that test was removed.

    When `lambda_high <= lambda_beta`, the high-tier additive term is
    omitted (i.e., high-tier coefficients receive only the W_freq-weighted
    ridge, same as low-tier). Negative extra penalties are not supported;
    if you want less shrinkage on the high tier, lower `lambda_beta` and
    rebuild `W_freq` instead.
    """
    if X.size == 0:
        return np.zeros(0, dtype=np.float64)
    p = X.shape[1]

    # Build the legacy diagonal penalty (DC and trend get weight 1; freq
    # columns get the freq_weight modulation).
    R = np.zeros(p, dtype=np.float64)
    col = 0
    if include_dc:
        R[col] = 1.0
        col += 1
    if include_trend:
        R[col] = 1.0
        col += 1
    freq_arr = np.asarray(freqs, dtype=np.float64)
    if freq_arr.size > 0:
        penalty = _make_frequency_penalty(freq_arr, freq_weight)
        for w_f in penalty:
            if col < p:
                R[col] = w_f
                col += 1
            if col < p:
                R[col] = w_f
                col += 1

    # Tier-dependent diagonal: λ_β · R^2 + λ_high · 1[high-tier]
    lam_diag = lambda_beta * (R ** 2)
    if freq_arr.size > 0 and lambda_high > lambda_beta:
        is_high = _classify_freq_tier(freq_arr, low_freq_period_days,
                                      time_unit_seconds=time_unit_seconds)
        # extra penalty applies to cos and sin columns of high-tier freqs
        col_h = (1 if include_dc else 0) + (1 if include_trend else 0)
        for k, hi in enumerate(is_high):
            if hi:
                lam_diag[col_h + 2 * k] += (lambda_high - lambda_beta)
                lam_diag[col_h + 2 * k + 1] += (lambda_high - lambda_beta)

    # Clip defends against negative lambda_beta from caller (algorithmic
    # path otherwise produces non-negative diagonals only).
    sqrt_lam = np.sqrt(np.clip(lam_diag, 0.0, None))
    X_aug = np.vstack([X, np.diag(sqrt_lam)])
    y_aug = np.concatenate([y, np.zeros(p, dtype=np.float64)])
    beta = _safe_lstsq(X_aug, y_aug)
    return np.asarray(beta, dtype=np.float64)

def _difference_weights(t_sec: np.ndarray, enable_dt_weighting: bool) -> np.ndarray:
    """Return per-step weights w_i for the difference operator.

    w_i = 1 / sqrt(max(Δt_i_in_days, 1.0)) when enabled, else 1.0.
    Output length is `len(t_sec) - 1`.
    """
    n = len(t_sec)
    if n <= 1:
        return np.zeros(0, dtype=np.float64)
    if not enable_dt_weighting:
        return np.ones(n - 1, dtype=np.float64)
    dt_days = np.diff(np.asarray(t_sec, dtype=np.float64)) / 86400.0
    dt_clamped = np.maximum(dt_days, 1.0)
    return 1.0 / np.sqrt(dt_clamped)

def _fused_lasso_1d(r: np.ndarray, lambda_step: float,
                    weights: np.ndarray,
                    max_iter: int = 5000, tol: float = 1e-9) -> np.ndarray:
    """Reference fused lasso via accelerated (FISTA) proximal gradient on the dual.

    Solves   min_u 0.5 ||r - u||^2 + lambda_step * sum_i w_i |u_{i+1} - u_i|

    by iterating on the dual variable z in R^{n-1}:

        u = r - D^T z
        z <- project_{|z_i| <= lam_i}(z + step * D u)

    where D is the first-difference operator and lam_i = lambda_step * w_i.
    Step size 1 / ||D D^T||_2 = 1/4 guarantees convergence (since the
    largest eigenvalue of D D^T on R^{n-1} is < 4). FISTA momentum gives
    O(1/k^2) convergence on the dual objective, important when lambda is
    very large because the slowest mode otherwise drags out for thousands
    of plain-gradient steps.
    """
    r = np.ascontiguousarray(r, dtype=np.float64)
    n = r.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return r.copy()
    if lambda_step <= 0.0:
        return r.copy()

    w = np.ascontiguousarray(weights, dtype=np.float64)
    if w.size != n - 1:
        raise ValueError(f"weights length {w.size} != n-1 ({n - 1})")
    lam = lambda_step * w

    # Disable sentinel: when the smallest per-step penalty already dwarfs
    # the data magnitude by ~1e6×, the fused-lasso solution is provably the
    # constant signal equal to mean(r). Short-circuit to avoid spending
    # FISTA iterations on a problem whose answer is closed-form.
    r_abs_max = float(np.max(np.abs(r))) if n > 0 else 0.0
    lam_min = float(lam.min()) if lam.size > 0 else 0.0
    if lam_min > 0.0 and lam_min >= 1e6 * (r_abs_max + 1.0):
        return np.full(n, float(r.mean()), dtype=np.float64)

    step = 0.25  # < 1 / ||D D^T||_2

    z = np.zeros(n - 1, dtype=np.float64)
    y = z.copy()             # FISTA extrapolated point
    t_prev = 1.0
    u_prev = None

    for _ in range(max_iter):
        # u from extrapolated dual y: u = r - D^T y
        DT_y = np.zeros(n, dtype=np.float64)
        DT_y[:-1] -= y
        DT_y[1:] += y
        u = r - DT_y
        # gradient step on y
        Du = np.diff(u)
        z_new = y + step * Du
        np.clip(z_new, -lam, lam, out=z_new)
        # FISTA extrapolation
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_prev * t_prev))
        y = z_new + ((t_prev - 1.0) / t_new) * (z_new - z)

        if u_prev is not None and np.max(np.abs(u - u_prev)) < tol:
            z = z_new
            break
        z = z_new
        t_prev = t_new
        u_prev = u

    # Recompute final primal from latest z (consistency).
    DT_z = np.zeros(n, dtype=np.float64)
    DT_z[:-1] -= z
    DT_z[1:] += z
    return r - DT_z

def fit_nufrost_pixel_step_singleband(
    t_sec: np.ndarray,
    y: np.ndarray,
    freqs_sel: np.ndarray,
    *,
    lambda_beta: float,
    lambda_high: float,
    lambda_step: float,
    low_freq_period_days: float,
    step_dt_weighting: bool,
    max_outer_iter: int,
    outer_tol: float,
    freq_weight: float,
    include_trend: bool,
    min_obs: int = 12,
) -> dict:
    """Single-band per-pixel fit with step term + tiered ridge.

    Solves the §3 BCD problem at native data scale (no y_scale normalization).
    Callers operating on Sentinel-2 DN values should rescale before calling
    if a different penalty regime is desired.

    Returns a dict with keys:
        valid, beta, u, freqs, t_min, t_rel_mean, n_iter, fill_value,
        include_trend
    `u` is returned at the same scale as `y`, padded to length len(y) with
    zeros at masked-out indices.
    """
    m = np.isfinite(y) & np.isfinite(t_sec)
    n_kept = int(m.sum())
    out_u = np.zeros_like(y, dtype=np.float64)
    freqs_arr = np.asarray(freqs_sel, dtype=np.float64)
    if n_kept < max(3, min_obs):
        return {
            "valid": False,
            "beta": np.zeros(0, dtype=np.float64),
            "u": out_u,
            "freqs": freqs_arr,
            "t_min": float("nan"),
            "t_rel_mean": float("nan"),
            "n_iter": 0,
            "fill_value": float(np.nanmean(y)) if np.isfinite(y).any() else float("nan"),
            "include_trend": include_trend,
        }

    t = np.asarray(t_sec[m], dtype=np.float64)
    yy = np.asarray(y[m], dtype=np.float64)
    t_rel = t - t.min()
    t_rel_mean = float(t_rel.mean())

    X = design_matrix(t_rel, freqs_sel, include_trend=include_trend, include_dc=True)
    if X.shape[1] == 0:
        return {
            "valid": False,
            "beta": np.zeros(0, dtype=np.float64),
            "u": out_u,
            "freqs": freqs_arr,
            "t_min": float(t.min()),
            "t_rel_mean": t_rel_mean,
            "n_iter": 0,
            "fill_value": float(np.nanmean(yy)),
            "include_trend": include_trend,
        }

    weights = _difference_weights(t, enable_dt_weighting=step_dt_weighting)
    u = np.zeros_like(yy, dtype=np.float64)
    beta = _tiered_ridge_solve(
        X, yy - u, freqs_sel,
        lambda_beta=lambda_beta, lambda_high=lambda_high,
        low_freq_period_days=low_freq_period_days,
        freq_weight=freq_weight,
        include_dc=True, include_trend=include_trend,
    )

    n_iter = 0
    for n_iter in range(1, max_outer_iter + 1):
        beta_old = beta
        u_old = u
        residual = yy - X @ beta
        u = _fused_lasso_1d(residual, lambda_step=lambda_step, weights=weights)
        beta = _tiered_ridge_solve(
            X, yy - u, freqs_sel,
            lambda_beta=lambda_beta, lambda_high=lambda_high,
            low_freq_period_days=low_freq_period_days,
            freq_weight=freq_weight,
            include_dc=True, include_trend=include_trend,
        )
        denom = max(float(np.linalg.norm(beta_old)) + float(np.linalg.norm(u_old)), 1e-12)
        delta = float(np.linalg.norm(beta - beta_old) + np.linalg.norm(u - u_old))
        if delta / denom < outer_tol:
            break

    full_u = np.zeros_like(y, dtype=np.float64)
    full_u[m] = u
    return {
        "valid": True,
        "beta": beta,
        "u": full_u,
        "freqs": freqs_arr,
        "t_min": float(t.min()),
        "t_rel_mean": t_rel_mean,
        "n_iter": n_iter,
        "fill_value": float(np.nanmean(yy)),
        "include_trend": include_trend,
    }


def _joint_outlier_mask(residuals: np.ndarray,
                         sigmas: np.ndarray,
                         sigma: float) -> np.ndarray:
    """Single-pass joint outlier mask across bands.

    Args:
        residuals: shape (n, B) per-band residuals from a quick ridge fit.
        sigmas: shape (B,) per-band MAD-derived robust scale estimates.
            Bands with sigma <= 0 are excluded from the joint score.
        sigma: rejection threshold applied to MAD(score).

    Returns:
        mask: shape (n,) bool. True where the joint anomaly score is within
        sigma * MAD(score) of the median; False where the timestep is
        rejected as a correlated outlier across bands.
    """
    R = np.asarray(residuals, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"residuals must be 2-D (n, B); got shape {R.shape}")
    n, B = R.shape
    if n == 0:
        return np.zeros(0, dtype=np.bool_)

    s = np.asarray(sigmas, dtype=np.float64)
    if s.size != B:
        raise ValueError(f"sigmas length {s.size} != B={B}")

    # Standardize residuals; ignore degenerate bands (sigma <= 0).
    valid_band = s > 0
    if not np.any(valid_band):
        return np.ones(n, dtype=np.bool_)
    Rz = np.zeros_like(R)
    Rz[:, valid_band] = R[:, valid_band] / s[valid_band]

    # Single effective band: fall back to a marginal sigma threshold on the
    # standardized residual. The MAD-of-score recipe used for B>=2 is biased
    # for a half-normal score distribution and would over-reject here.
    n_valid = int(np.sum(valid_band))
    if n_valid == 1:
        z = Rz[:, valid_band].ravel()
        return np.abs(z) <= sigma

    # Joint anomaly score per timestep.
    score = np.linalg.norm(Rz[:, valid_band], axis=1)

    # Threshold via MAD on the score.
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med))) * 1.4826
    if mad <= 0.0:
        # Fallback: degenerate score distribution; keep everything finite.
        return np.isfinite(score)
    threshold = med + sigma * mad
    return score <= threshold


def _solve_tridiag_thomas(diag: np.ndarray, off: np.ndarray,
                           rhs: np.ndarray) -> np.ndarray:
    """In-place Thomas solver for symmetric tridiagonal `Ax = rhs`.

    Args:
        diag: shape (n,) main diagonal of A.
        off: shape (n-1,) sub/super-diagonal of A.
        rhs: shape (n,) or (n, B) right-hand side; solved column-wise.

    Returns:
        x: same shape as rhs. New array; inputs unchanged.
    """
    n = diag.size
    if rhs.ndim == 1:
        rhs2 = rhs.reshape(n, 1)
    else:
        rhs2 = rhs
    cprime = np.empty(n - 1, dtype=np.float64)
    dprime = np.empty_like(rhs2, dtype=np.float64)
    cprime[0] = off[0] / diag[0]
    dprime[0] = rhs2[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - off[i - 1] * cprime[i - 1]
        if i < n - 1:
            cprime[i] = off[i] / denom
        dprime[i] = (rhs2[i] - off[i - 1] * dprime[i - 1]) / denom
    x = np.empty_like(dprime)
    x[-1] = dprime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dprime[i] - cprime[i] * x[i + 1]
    if rhs.ndim == 1:
        return x.ravel()
    return x


def _group_fused_lasso_admm(R: np.ndarray, lambda_step: float,
                             weights: np.ndarray,
                             rho: float = 1.0,
                             max_iter: int = 80,
                             tol: float = 1e-4) -> np.ndarray:
    """Solve the multi-band group fused lasso via ADMM.

    Solves
        min_U  0.5 * ||R - U||_F^2
             + lambda_step * sum_i weights_i * ||(D U)_i||_2

    where D in R^{(n-1) x n} is the first-difference operator and
    ||.||_2 acts on the row of D U at time index i, treating the
    band axis as the group.

    For a 2-D R of shape (n, B). For 1-D input it returns 1-D.

    Convergence: ADMM on convex objective with proximable g = group L1.
    Boyd (2011) Theorem 1 gives objective and primal-residual convergence;
    iterate convergence holds because (1/2)||R - U||_F^2 is strictly convex
    in U (spec section 14.5.2).
    """
    R_in = np.asarray(R, dtype=np.float64)
    squeeze = R_in.ndim == 1
    if squeeze:
        R_in = R_in.reshape(-1, 1)
    n, B = R_in.shape
    if n == 0:
        out = np.zeros((0, B), dtype=np.float64)
        return out.ravel() if squeeze else out
    if n == 1:
        out = R_in.copy()
        return out.ravel() if squeeze else out
    w = np.asarray(weights, dtype=np.float64)
    if w.size != n - 1:
        raise ValueError(f"weights length {w.size} != n-1 ({n - 1})")
    if lambda_step <= 0.0:
        out = R_in.copy()
        return out.ravel() if squeeze else out

    lam = lambda_step * w  # shape (n-1,)

    # Disable sentinel: when the smallest per-step penalty already exceeds
    # max_i ||(D R)_i||_2 by a comfortable margin, V is zeroed at every
    # iteration and U converges to the per-band column mean. Short-circuit.
    if lam.size > 0:
        DR = np.diff(R_in, axis=0)
        dr_max = float(np.max(np.linalg.norm(DR, axis=1))) if DR.size > 0 else 0.0
        lam_min = float(lam.min())
        if lam_min > 0.0 and lam_min >= 10.0 * (dr_max + 1e-12):
            means = R_in.mean(axis=0, keepdims=True)
            out = np.broadcast_to(means, (n, B)).copy()
            return out.ravel() if squeeze else out

    # Build I + rho * D^T D as a tridiagonal SPD matrix.
    diag = np.full(n, 1.0 + 2.0 * rho, dtype=np.float64)
    diag[0] = 1.0 + rho
    diag[-1] = 1.0 + rho
    off = np.full(n - 1, -rho, dtype=np.float64)

    V = np.zeros((n - 1, B), dtype=np.float64)
    Lam = np.zeros((n - 1, B), dtype=np.float64)
    U_prev = R_in.copy()

    U = R_in.copy()
    for _ in range(max_iter):
        # U-update: solve (I + rho D^T D) U = R + rho D^T (V - Lam)
        diff = V - Lam                               # (n-1, B)
        DT_term = np.zeros((n, B), dtype=np.float64)
        DT_term[:-1] -= diff
        DT_term[1:] += diff
        rhs = R_in + rho * DT_term
        U = _solve_tridiag_thomas(diag, off, rhs)

        # V-update: group soft-threshold per row
        DU = np.diff(U, axis=0)                      # (n-1, B)
        Z = DU + Lam
        norms = np.linalg.norm(Z, axis=1)            # (n-1,)
        thresh = lam / rho                           # (n-1,)
        scale = np.where(norms > 0, np.maximum(0.0, 1.0 - thresh / np.maximum(norms, 1e-30)), 0.0)
        V = scale[:, None] * Z

        # Dual update
        Lam = Lam + DU - V

        # Convergence: primal residual on each block
        primal = np.max(np.abs(DU - V))
        change = np.max(np.abs(U - U_prev))
        U_prev = U
        if primal < tol and change < tol:
            break

    out = U
    return out.ravel() if squeeze else out


def huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    w = np.ones_like(r)
    m = a > delta
    w[m] = (delta / a[m])
    return w

def _safe_lstsq(X: np.ndarray, y: np.ndarray, rcond: Optional[float] = None) -> np.ndarray:
    """更稳健的 lstsq：失败则小抖动回退；再失败提升 ridge。"""
    try:
        return np.linalg.lstsq(X, y, rcond=rcond)[0]
    except Exception:
        # 加一个小的对角抖动
        try:
            eps = 1e-8
            p = X.shape[1]
            return np.linalg.lstsq(
                np.vstack([X, np.sqrt(eps)*np.eye(p)]),
                np.concatenate([y, np.zeros(p)]),
                rcond=None
            )[0]
        except Exception:
            # 再失败就进一步增大 ridge
            eps2 = 1e-4
            p = X.shape[1]
            return np.linalg.lstsq(
                np.vstack([X, np.sqrt(eps2)*np.eye(p)]),
                np.concatenate([y, np.zeros(p)]),
                rcond=None
            )[0]

def _make_frequency_penalty(freqs: np.ndarray, freq_weight: float = 2.0) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=np.float64)
    penalty = np.ones(len(freqs), dtype=np.float64)
    positive = freqs[np.isfinite(freqs) & (freqs > 0)]
    if positive.size == 0:
        return penalty
    base = float(np.min(positive))
    for idx, freq in enumerate(freqs):
        if not np.isfinite(freq) or freq <= base:
            continue
        rel = max(float(freq) / base, 1.0)
        penalty[idx] = math.sqrt(1.0 + max(0.0, freq_weight) * math.log2(rel))
    return penalty

def ridge_with_freq_weights(X: np.ndarray, y: np.ndarray, freqs: Optional[np.ndarray], lam: float, include_dc: bool = True, include_trend: bool = True, freq_weight: float = 2.0, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return np.zeros(0, dtype=float), np.full_like(y, np.nanmean(y))
    if w is None:
        w = np.ones_like(y)
    W = np.sqrt(w)
    Xw = X * W[:, None]
    yw = y * W

    if lam <= 0:
        beta = _safe_lstsq(Xw, yw)
        y_hat = X @ beta
        return beta, y_hat

    p = X.shape[1]
    R = np.zeros(p, dtype=np.float64)
    col = 0
    if include_dc:
        col += 1
    if include_trend:
        col += 1
    if freqs is not None and len(freqs) > 0:
        penalty = _make_frequency_penalty(np.asarray(freqs, dtype=np.float64), freq_weight)
        for w_f in penalty:
            if col < p:
                R[col] = w_f
                col += 1
            if col < p:
                R[col] = w_f
                col += 1

    if np.any(R > 0):
        X_aug = np.vstack([Xw, np.diag(np.sqrt(lam)*R)])
    else:
        X_aug = np.vstack([Xw, np.sqrt(lam)*np.eye(p)])
    y_aug = np.concatenate([yw, np.zeros(p, dtype=np.float64)])

    beta = _safe_lstsq(X_aug, y_aug)
    y_hat = X @ beta
    return beta, y_hat

def robust_fit_freq_ridge(X: np.ndarray, y: np.ndarray, freqs: np.ndarray, lam: float, iters: int, delta: float, include_dc: bool, include_trend: bool, freq_weight: float) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[1] == 0:
        return np.zeros(0), np.full_like(y, np.nanmean(y))
    w = np.ones_like(y)
    _, y_hat = ridge_with_freq_weights(X, y, freqs, lam,
                                       include_dc, include_trend, freq_weight, w=w)
    for _ in range(max(0, iters)):
        r = y - y_hat
        w = huber_weights(r, max(1e-8, delta))
        _, y_hat = ridge_with_freq_weights(X, y, freqs, lam,
                                           include_dc, include_trend, freq_weight, w=w)
    beta, y_hat = ridge_with_freq_weights(X, y, freqs, lam,
                                          include_dc, include_trend, freq_weight, w=w)
    return beta, y_hat

def predict_single_pixel(t_sec: np.ndarray, y: np.ndarray, target_t: float,
                         nufft_modes: int, eps: float,
                         num_peaks: int, power_cum: float, ignore_dc_hz: float,
                         refine_peaks: bool, include_trend: bool,
                         ridge_lam: float, freq_weight: float, huber_iters: int, huber_delta: float,
                         min_obs: int) -> Tuple[float, int]:
    m = np.isfinite(y) & np.isfinite(t_sec)
    if m.sum() < max(3, min_obs):
        return np.nan, 0
    t = np.asarray(t_sec[m], dtype=np.float64)
    yy = np.asarray(y[m], dtype=np.float64)

    t_rel = _to_seconds_since_start(t)
    t_rel_mean = float(t_rel.mean())
    Tspan = float(t_rel.max() - t_rel.min())
    if not np.isfinite(Tspan) or Tspan <= 0:
        return np.nan, 0

    x = 2*np.pi*(t_rel - t_rel.min())/Tspan - np.pi
    x = np.ascontiguousarray(x, dtype=np.float64)
    c = np.ascontiguousarray(yy.astype(np.complex128))
    ms = next_even(nufft_modes)
    Fk = finufft.nufft1d1(x, c, ms, eps=eps, isign=-1)
    k = np.arange(-ms//2, ms//2, dtype=np.int64)
    freqs = k.astype(np.float64)/Tspan

    pos = freqs >= 0
    f_pos = freqs[pos]
    P_pos = (np.abs(Fk[pos])**2)

    dt = np.diff(np.sort(t_rel))
    dt_pos = dt[dt > 0]
    dt_med = float(np.median(dt_pos)) if dt_pos.size else Tspan/len(t_rel)
    fmax = 0.5 / max(dt_med, 1e-12)

    pos_idx = select_peaks_adaptive(f_pos, P_pos, k_max=num_peaks,
                                    power_cum=power_cum,
                                    ignore_dc_hz=ignore_dc_hz, fmax=fmax)
    if len(pos_idx) == 0:
        freqs_sel = np.array([], dtype=np.float64)
    else:
        if refine_peaks:
            freqs_sel = np.array([refine_parabolic(f_pos, P_pos, i) for i in pos_idx], dtype=np.float64)
        else:
            freqs_sel = np.array(f_pos[pos_idx], dtype=np.float64)

    X = design_matrix(t_rel, freqs_sel, include_trend=include_trend, include_dc=True)
    beta, _ = robust_fit_freq_ridge(X, yy, freqs_sel,
                                    lam=ridge_lam, iters=huber_iters, delta=huber_delta,
                                    include_dc=True, include_trend=include_trend, freq_weight=freq_weight)

    t_star_rel = float(target_t - t.min())
    cols = [1.0]
    if include_trend:
        cols.append(t_star_rel - t_rel_mean)
    for f in freqs_sel:
        w = 2*np.pi*f
        cols.append(math.cos(w*t_star_rel))
        cols.append(math.sin(w*t_star_rel))
    X_star = np.array(cols, dtype=np.float64).reshape(1, -1) if len(cols) else np.zeros((1,0))

    if X.shape[1] > 0:
        y_star = float((X_star @ beta).item())
    else:
        y_star = float(np.nanmean(yy))
    return y_star, len(freqs_sel)

def reconstruct_nufrost(
    image: str,
    target_time: str,
    output_path: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    一键重建接口。

    参数:
        image: 输入的多波段 TIF 路径
        target_time: 目标重建时间 (如 '2023-06-15')
        output_path: 输出 TIF 路径 (可选)
        **kwargs: 其他算法参数 (如 n_jobs, ridge, etc.)
    """
    # 1. 构建参数
    overrides = {**kwargs, "image": image, "target_time": target_time}
    if output_path:
        overrides["output_path"] = output_path

    args = build_args("nufrost", overrides=overrides)

    # 2. 加载数据
    loader = RSCube(args.image, cache_dir=args.cache_dir, force_refresh=args.force_refresh)
    data = loader.load()
    cube = data["cube"]
    timestamps = data["timestamps"]

    # 3. 执行重建
    recon = nufrost_core(cube, timestamps, args.target_time, args=args)

    # 4. 保存结果
    if args.output_path:
        out_p = Path(args.output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        transform = None
        if "transform" in data:
            transform = rasterio.Affine(*data["transform"])

        with rasterio.open(
            out_p, "w",
            driver="GTiff",
            height=recon.shape[0],
            width=recon.shape[1],
            count=1,
            dtype=recon.dtype,
            crs=data.get("crs_wkt"),
            transform=transform,
        ) as dst:
            dst.write(recon, 1)
        print(f"[Success] Saved to: {out_p}")

    return recon

def nufrost_core(cube: np.ndarray, timestamps: np.ndarray, target_time: str, args: Optional[Args] = None, shared_freqs=None, **kwargs) -> np.ndarray:
    """核心重建函数，支持直接传入参数或 Args 对象"""
    # 合并参数
    if args is None:
        from config import Args
        args = Args(**kwargs)

    t_sec = timestamps_to_seconds(timestamps, unit=args.time_unit)
    target_dt = parse_timestamp_str(target_time)
    if target_dt is None:
        raise ValueError(f"Unrecognized target_time: {target_time}")

    target_t = target_dt.timestamp()
    if args.time_unit == "days":
        target_t = target_t / 86400.0

    bands, H, W = cube.shape
    out = np.full((H, W), np.nan, dtype=np.float32)

    def _predict_row(i: int) -> Tuple[int, np.ndarray]:
        row = np.full(W, np.nan, dtype=np.float32)
        for j in range(W):
            y = cube[:, i, j]
            pred, _ = predict_single_pixel(
                t_sec, y, target_t,
                nufft_modes=args.modes, eps=args.eps,
                num_peaks=args.num_peaks, power_cum=args.power_cum, ignore_dc_hz=args.ignore_dc_hz,
                frequency_selection=args.frequency_selection,
                preferred_periods_days=args.preferred_periods_days,
                preferred_top_k=args.preferred_top_k,
                spectral_top_k=args.spectral_top_k,
                spectral_merge_tol=args.spectral_merge_tol,
                refine_peaks=args.refine_peaks, include_trend=args.include_trend,
                ridge_lam=args.ridge, freq_weight=args.freq_weight, huber_iters=args.huber_iters, huber_delta=args.huber_delta,
                    min_obs=args.min_obs,
                    shared_freqs=shared_freqs,
                    outlier_sigma=args.outlier_sigma,
                )
            row[j] = pred
        return i, row

    # 确定并行任务数
    n_jobs = args.n_jobs
    if n_jobs <= 0:
        n_jobs = max(1, int(os.cpu_count() or 1)) if cpu_count is None else max(1, int(cpu_count()))
    n_jobs = min(n_jobs, H)

    print(f"[System] Starting NuFrost reconstruction with {n_jobs} jobs...")

    # 串行执行
    if not JOBLIB_AVAILABLE or n_jobs == 1:
        iterator = range(H)
        if args.show_progress and TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=H, desc="Processing Rows")

        for i in iterator:
            _, row = _predict_row(i)
            out[i, :] = row
        return out

    # 并行执行
    assert Parallel is not None and delayed is not None
    try:
        if args.show_progress and TQDM_AVAILABLE:
            # 尝试使用 generator 模式配合 tqdm 显示进度
            try:
                results_gen = Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator")(
                    delayed(_predict_row)(i) for i in range(H)
                )
                for i, row in tqdm(results_gen, total=H, desc="Processing Rows"):
                    out[i, :] = row
                return out
            except TypeError:
                # 如果 joblib 版本太老不支持 return_as="generator"，回退到普通模式
                pass

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_predict_row)(i) for i in range(H)
        )
        for i, row in results:
            out[i, :] = row
    except Exception as e:
        print(f"[Error] Parallel execution failed: {e}. Falling back to serial.")
        for i in range(H):
            _, row = _predict_row(i)
            out[i, :] = row

    return out

from typing import Any, Dict as DictType
import math as _math


def _parse_preferred_periods_days(periods_spec):
    if isinstance(periods_spec, str):
        parts = [p.strip() for p in periods_spec.split(",") if p.strip()]
        vals = [float(p) for p in parts]
    else:
        vals = [float(v) for v in periods_spec]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return np.array(vals, dtype=np.float64)


def _preferred_periods_to_freqs(periods_days, time_unit="seconds"):
    periods_days_arr = _parse_preferred_periods_days(periods_days)
    if periods_days_arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    if time_unit == "days":
        return 1.0 / periods_days_arr
    return 1.0 / (periods_days_arr * 86400.0)


def _snap_frequency_to_spectrum(target_freq, f_pos, P_pos, rel_tol):
    if not np.isfinite(target_freq) or target_freq <= 0:
        return float(target_freq)
    valid = np.isfinite(f_pos) & np.isfinite(P_pos) & (f_pos > 0)
    if not np.any(valid):
        return float(target_freq)
    cand_f = f_pos[valid]
    cand_p = P_pos[valid]
    rel_err = np.abs(cand_f - target_freq) / max(target_freq, 1e-12)
    nearby = np.where(rel_err <= max(rel_tol, 0.0))[0]
    if nearby.size == 0:
        return float(target_freq)
    best_local = nearby[np.argmax(cand_p[nearby])]
    return float(cand_f[best_local])


def select_frequencies(
    f_pos, P_pos, fmax, selection_mode, preferred_freqs,
    preferred_top_k, spectral_top_k, spectral_merge_tol,
    power_cum, ignore_dc_hz, refine_peaks,
):
    selected = []
    if selection_mode in ("preferred", "hybrid") and preferred_freqs.size > 0:
        pref_valid = preferred_freqs[np.isfinite(preferred_freqs) & (preferred_freqs > ignore_dc_hz)]
        pref_valid = pref_valid[pref_valid <= fmax]
        for f in pref_valid[:max(0, preferred_top_k)]:
            selected.append(_snap_frequency_to_spectrum(f, f_pos, P_pos, spectral_merge_tol))
    if selection_mode in ("spectral", "hybrid"):
        peak_idx = select_peaks_adaptive(f_pos, P_pos, k_max=max(0, spectral_top_k),
                                          power_cum=power_cum, ignore_dc_hz=ignore_dc_hz, fmax=fmax)
        if len(peak_idx) > 0:
            if refine_peaks:
                sel_freqs = [refine_parabolic(f_pos, P_pos, i) for i in peak_idx]
            else:
                sel_freqs = [float(f_pos[i]) for i in peak_idx]
            selected.extend(sel_freqs)
    if not selected:
        return np.zeros(0, dtype=np.float64)
    selected = sorted(float(f) for f in selected if np.isfinite(f) and f > ignore_dc_hz and f <= fmax)
    merged = []
    for f in selected:
        if not merged:
            merged.append(f)
            continue
        rel = abs(f - merged[-1]) / max(merged[-1], 1e-12)
        if rel <= max(spectral_merge_tol, 0.0):
            merged[-1] = 0.5 * (merged[-1] + f)
        else:
            merged.append(f)
    return np.array(merged, dtype=np.float64)


def fit_nufrost_pixel_params(
    t_sec, y, target_t=None,
    nufft_modes=4096, eps=1e-12, num_peaks=10, power_cum=0.7, ignore_dc_hz=1e-10,
    frequency_selection="spectral", preferred_periods_days="365.25,182.625,91.3125,30.4375",
    preferred_top_k=4, spectral_top_k=4, spectral_merge_tol=0.15,
    refine_peaks=True, include_trend=True, ridge_lam=0.01, freq_weight=2.0,
    huber_iters=3, huber_delta=1.5, min_obs=12, max_freqs=None, shared_freqs=None,
    outlier_sigma=2.0,
):
    m = np.isfinite(y) & np.isfinite(t_sec)
    if m.sum() < max(3, min_obs):
        return {"valid": False, "include_trend": include_trend, "n_freqs_used": 0,
                "t_min": np.nan, "t_rel_mean": np.nan, "fill_value": np.nan,
                "freqs": np.array([]), "beta": np.array([])}
    t = np.asarray(t_sec[m], dtype=np.float64)
    yy = np.asarray(y[m], dtype=np.float64)
    fill_value = float(np.nanmean(yy))
    t_rel = _to_seconds_since_start(t)
    t_rel_mean = float(t_rel.mean())
    Tspan = float(t_rel.max() - t_rel.min())
    if not np.isfinite(Tspan) or Tspan <= 0:
        return {"valid": False, "include_trend": include_trend, "n_freqs_used": 0,
                "t_min": np.nan, "t_rel_mean": t_rel_mean, "fill_value": fill_value,
                "freqs": np.array([]), "beta": np.array([])}
    x = 2 * np.pi * (t_rel - t_rel.min()) / Tspan - np.pi
    x = np.ascontiguousarray(x, dtype=np.float64)
    y_scale = 10000.0
    yy_scaled = yy / y_scale
    c = np.ascontiguousarray(yy_scaled.astype(np.complex128))
    ms = next_even(nufft_modes)
    Fk = finufft.nufft1d1(x, c, ms, eps=eps, isign=-1)
    k = np.arange(-ms // 2, ms // 2, dtype=np.int64)
    freqs = k.astype(np.float64) / Tspan
    pos = freqs >= 0
    f_pos = freqs[pos]
    P_pos = (np.abs(Fk[pos]) ** 2)
    dt = np.diff(np.sort(t_rel))
    dt_pos = dt[dt > 0]
    dt_med = float(np.median(dt_pos)) if dt_pos.size else Tspan / len(t_rel)
    fmax = 0.5 / max(dt_med, 1e-12)
    if shared_freqs is not None:
        shared_arr = np.asarray(shared_freqs, dtype=np.float64)
        shared_arr = shared_arr[np.isfinite(shared_arr) & (shared_arr > ignore_dc_hz) & (shared_arr <= fmax)]
        if shared_arr.size > 0:
            freqs_sel = shared_arr
        else:
            preferred_freqs = _preferred_periods_to_freqs(preferred_periods_days, time_unit="seconds")
            spectral_top_k_eff = spectral_top_k if spectral_top_k > 0 else num_peaks
            freqs_sel = select_frequencies(
                f_pos=f_pos, P_pos=P_pos, fmax=fmax,
                selection_mode=frequency_selection,
                preferred_freqs=preferred_freqs,
                preferred_top_k=preferred_top_k,
                spectral_top_k=spectral_top_k_eff,
                spectral_merge_tol=spectral_merge_tol,
                power_cum=power_cum,
                ignore_dc_hz=ignore_dc_hz,
                refine_peaks=refine_peaks,
            )
    else:
        preferred_freqs = _preferred_periods_to_freqs(preferred_periods_days, time_unit="seconds")
        spectral_top_k_eff = spectral_top_k if spectral_top_k > 0 else num_peaks
        freqs_sel = select_frequencies(
            f_pos=f_pos,
            P_pos=P_pos,
            fmax=fmax,
            selection_mode=frequency_selection,
            preferred_freqs=preferred_freqs,
            preferred_top_k=preferred_top_k,
            spectral_top_k=spectral_top_k_eff,
            spectral_merge_tol=spectral_merge_tol,
            power_cum=power_cum,
            ignore_dc_hz=ignore_dc_hz,
            refine_peaks=refine_peaks,
        )
    if max_freqs is not None:
        freqs_sel = freqs_sel[:max(0, int(max_freqs))]

    num_params = 1 + int(include_trend) + 2 * len(freqs_sel)
    if outlier_sigma > 0 and len(yy) >= min_obs and num_params > 0:
        min_needed = max(min_obs, num_params + 2)
        t_curr = t_rel
        y_curr = yy_scaled
        beta = None
        for _ in range(min(len(yy), 50)):
            n_obs = len(y_curr)
            if n_obs < min_needed:
                break
            X_curr = design_matrix(t_curr, freqs_sel, include_trend=include_trend, include_dc=True)
            beta_curr, _ = robust_fit_freq_ridge(X_curr, y_curr, freqs_sel, lam=ridge_lam,
                                                   iters=huber_iters, delta=huber_delta,
                                                   include_dc=True, include_trend=include_trend,
                                                   freq_weight=freq_weight)
            beta = beta_curr
            y_pred = X_curr @ beta_curr
            residuals = y_curr - y_pred
            mad = float(np.median(np.abs(residuals - np.median(residuals))))
            threshold = outlier_sigma * 1.4826 * max(mad, 1e-12)
            clean_mask = np.abs(residuals) <= threshold
            if clean_mask.all():
                break
            t_curr = t_curr[clean_mask]
            y_curr = y_curr[clean_mask]
        if beta is None:
            X = design_matrix(t_rel, freqs_sel, include_trend=include_trend, include_dc=True)
            beta, _ = robust_fit_freq_ridge(X, yy_scaled, freqs_sel, lam=ridge_lam, iters=huber_iters,
                                             delta=huber_delta, include_dc=True, include_trend=include_trend,
                                             freq_weight=freq_weight)
    else:
        X = design_matrix(t_rel, freqs_sel, include_trend=include_trend, include_dc=True)
        beta, _ = robust_fit_freq_ridge(X, yy_scaled, freqs_sel, lam=ridge_lam, iters=huber_iters, delta=huber_delta,
                                        include_dc=True, include_trend=include_trend, freq_weight=freq_weight)

    params = {
        "valid": True, "include_trend": include_trend,
        "n_freqs_used": len(freqs_sel), "t_min": float(t.min()),
        "t_rel_mean": t_rel_mean, "fill_value": fill_value,
        "freqs": freqs_sel, "beta": beta, "y_scale": y_scale,
    }
    return params


def _pad_freqs(freqs, max_freqs):
    out = np.full(max_freqs, np.nan, dtype=np.float64)
    out[:len(freqs)] = np.asarray(freqs, dtype=np.float64)
    return out


def _pad_beta(beta, max_freqs, include_trend):
    size = 1 + (1 if include_trend else 0) + 2 * max_freqs
    out = np.full(size, np.nan, dtype=np.float64)
    out[:len(beta)] = np.asarray(beta, dtype=np.float64)
    return out


def predict_nufrost_from_params(params, target_t):
    if not params.get("valid", False):
        return float(params.get("fill_value", np.nan))
    include_trend = params.get("include_trend", True)
    n_freqs = min(int(params.get("n_freqs_used", 0)), 1000)
    freqs_all = np.asarray(params.get("freqs", []))
    beta_all = np.asarray(params.get("beta", []))
    freqs_sel = freqs_all[:n_freqs]
    beta_len = 1 + int(include_trend) + 2 * n_freqs
    beta = beta_all[:beta_len]
    t_star_rel = float(target_t - float(params.get("t_min", 0.0)))
    cols = [1.0]
    if include_trend:
        cols.append(t_star_rel - float(params.get("t_rel_mean", 0.0)))
    for f in freqs_sel:
        w = 2 * np.pi * f
        cols.append(_math.cos(w * t_star_rel))
        cols.append(_math.sin(w * t_star_rel))
    return float(np.asarray(cols, dtype=np.float64) @ beta[:len(cols)] * float(params.get("y_scale", 1.0)))


def predict_nufrost_curve_from_params(params, target_t_secs):
    return np.array([predict_nufrost_from_params(params, float(t)) for t in target_t_secs], dtype=np.float64)


def predict_curve_pixel(
    t_sec, y, target_t_secs, nufft_modes, eps, num_peaks, power_cum, ignore_dc_hz,
    frequency_selection="spectral", preferred_periods_days="365.25,182.625,91.3125,30.4375",
    preferred_top_k=4, spectral_top_k=4, spectral_merge_tol=0.15,
    refine_peaks=True, include_trend=True, ridge_lam=0.01, freq_weight=2.0,
    huber_iters=3, huber_delta=1.5, min_obs=12,
):
    params = fit_nufrost_pixel_params(
        t_sec, y, nufft_modes=nufft_modes, eps=eps, num_peaks=num_peaks,
        power_cum=power_cum, ignore_dc_hz=ignore_dc_hz,
        refine_peaks=refine_peaks, include_trend=include_trend,
        ridge_lam=ridge_lam, freq_weight=freq_weight,
        huber_iters=huber_iters, huber_delta=huber_delta, min_obs=min_obs,
    )
    return predict_nufrost_curve_from_params(params, target_t_secs)

# ── Padded versions for model_params ──

def _padded_fit_nufrost_pixel_params(
    t_sec, y, target_t=None,
    nufft_modes=4096, eps=1e-12, num_peaks=10, power_cum=0.7, ignore_dc_hz=1e-10,
    frequency_selection="spectral", preferred_periods_days="365.25,182.625,91.3125,30.4375",
    preferred_top_k=4, spectral_top_k=4, spectral_merge_tol=0.15,
    refine_peaks=True, include_trend=True, ridge_lam=0.01, freq_weight=2.0,
    huber_iters=3, huber_delta=1.5, min_obs=12, max_freqs=10, shared_freqs=None,
    outlier_sigma=2.0,
):
    raw = _orig_fit(
        t_sec, y, nufft_modes=nufft_modes, eps=eps, num_peaks=num_peaks,
        power_cum=power_cum, ignore_dc_hz=ignore_dc_hz,
        frequency_selection=frequency_selection,
        preferred_periods_days=preferred_periods_days,
        preferred_top_k=preferred_top_k,
        spectral_top_k=spectral_top_k,
        spectral_merge_tol=spectral_merge_tol,
        refine_peaks=refine_peaks, include_trend=include_trend,
        ridge_lam=ridge_lam, freq_weight=freq_weight,
        huber_iters=huber_iters, huber_delta=huber_delta, min_obs=min_obs,
        max_freqs=max_freqs, shared_freqs=shared_freqs,
        outlier_sigma=outlier_sigma,
    )
    raw["freqs"] = _pad_freqs(raw["freqs"], max(1, max_freqs or 10))
    raw["beta"] = _pad_beta(raw["beta"], max(1, max_freqs or 10), bool(raw.get("include_trend", True)))
    return raw


import sys as _sys
_orig_fit = fit_nufrost_pixel_params
fit_nufrost_pixel_params = _padded_fit_nufrost_pixel_params
fit_nufrost_pixel_params.__name__ = "fit_nufrost_pixel_params"

# Make original predict_single_pixel accept newer keyword arguments
_orig_predict_single_pixel = predict_single_pixel
_predict_single_pixel_defined = True
def predict_single_pixel(
    t_sec, y, target_t,
    nufft_modes=4096, eps=1e-12,
    num_peaks=10, power_cum=0.7, ignore_dc_hz=1e-10,
    frequency_selection="spectral", preferred_periods_days="365.25,182.625,91.3125,30.4375",
    preferred_top_k=4, spectral_top_k=4, spectral_merge_tol=0.15,
    refine_peaks=True, include_trend=True,
    ridge_lam=0.01, freq_weight=2.0, huber_iters=3, huber_delta=1.5,
    min_obs=12, max_freqs=None, shared_freqs=None, outlier_sigma=2.0,
):
    params = fit_nufrost_pixel_params(
        t_sec, y,
        nufft_modes=nufft_modes, eps=eps,
        num_peaks=num_peaks, power_cum=power_cum, ignore_dc_hz=ignore_dc_hz,
        frequency_selection=frequency_selection,
        preferred_periods_days=preferred_periods_days,
        preferred_top_k=preferred_top_k,
        spectral_top_k=spectral_top_k,
        spectral_merge_tol=spectral_merge_tol,
        refine_peaks=refine_peaks, include_trend=include_trend,
        ridge_lam=ridge_lam, freq_weight=freq_weight,
        huber_iters=huber_iters, huber_delta=huber_delta,
        min_obs=min_obs,
        max_freqs=max_freqs or max(num_peaks, preferred_top_k + spectral_top_k, 1),
        shared_freqs=shared_freqs,
        outlier_sigma=outlier_sigma,
    )
    return predict_nufrost_from_params(params, target_t), int(params.get("n_freqs_used", 0))
