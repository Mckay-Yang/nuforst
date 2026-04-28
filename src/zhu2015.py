import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from typing import Tuple, Optional, Union, Any, Dict as DictType
from datetime import datetime
from .data_loader import RSCube
from .nufrost import timestamps_to_seconds
from config import Args, build_args
import rasterio
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import os

# Constants
DAYS_PER_YEAR = 365.25

def _get_julian_date(timestamps: np.ndarray) -> np.ndarray:
    """Convert timestamps (seconds) to Julian dates (days)."""
    # Simply convert seconds to days. The exact reference epoch doesn't matter
    # as long as we use the same reference for training and prediction.
    # We'll use the first timestamp as t=0 to keep numbers small.
    return timestamps / 86400.0

def make_design_matrix(x: np.ndarray, order: int) -> np.ndarray:
    """
    Construct design matrix for harmonic model with linear trend.
    x: array of time points (days)
    order: 1 (Simple), 2 (Advanced), 3 (Full)

    Columns: [cos(2pi*x/T), sin(2pi*x/T), ..., x]
    Intercept is handled by Lasso fit_intercept=True.
    """
    cols = []
    # Linear trend
    # Paper Equation 1: c1 * x
    # We add 'x' as a feature.

    # Harmonics
    w = 2 * np.pi / DAYS_PER_YEAR
    for k in range(1, order + 1):
        cols.append(np.cos(k * w * x))
        cols.append(np.sin(k * w * x))

    cols.append(x)
    return np.column_stack(cols)

def fit_predict_pixel(
    t_days: np.ndarray,
    y: np.ndarray,
    target_t_day: float,
    lasso_alpha: float = 0.001
) -> Tuple[float, int]:
    """
    Fit model for a single pixel and predict at target time.
    Returns (prediction, model_used)
    model_used: 0=Median, 1=Simple, 2=Advanced, 3=Full
    """
    # Filter valid data
    valid_mask = np.isfinite(y)
    if not np.any(valid_mask):
        return np.nan, 0

    t_valid = t_days[valid_mask]
    y_valid = y[valid_mask]
    n_obs = len(y_valid)

    # Model Selection Logic (Zhu et al. 2015)
    # < 6: Median
    # 6 <= N < 18: Simple (Order 1)
    # 18 <= N < 24: Advanced (Order 2)
    # >= 24: Full (Order 3)

    if n_obs < 6:
        return np.median(y_valid), 0

    if 6 <= n_obs < 18:
        order = 1
        model_id = 1
    elif 18 <= n_obs < 24:
        order = 2
        model_id = 2
    else: # n_obs >= 24
        order = 3
        model_id = 3

    # Build design matrix
    X = make_design_matrix(t_valid, order)

    # Fit LASSO
    # Note: Zhu 2015 uses LASSO.
    # alpha controls regularization.
    # Paper doesn't specify alpha, we assume a small value or CV.
    # Using fixed small alpha for now as in many remote sensing implementations.
    # We should normalize X? sklearn Lasso does not normalize by default,
    # but harmonic terms are bound [-1, 1], time x is unbounded.
    # Usually standardizing 'x' is good.
    # However, for simplicity and consistency with harmonic scales, we might just run it.
    # Let's standardize x internally if needed? Lasso(normalize=False) is default.

    clf = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=2000)
    clf.fit(X, y_valid)

    # Predict
    X_target = make_design_matrix(np.array([target_t_day]), order)
    y_pred = clf.predict(X_target)[0]

    return y_pred, model_id

def reconstruct_zhu2015(
    image: str,
    target_time: str,
    output_path: Optional[str] = None,
    lasso_alpha: float = 0.0001,
    n_jobs: int = -1,
    cache_dir: str = "./cache",
    force_refresh: bool = False
) -> np.ndarray:
    """
    Reconstruct Landsat image using Zhu et al. (2015) method.
    """
    # 1. Load Data
    loader = RSCube(image, cache_dir=cache_dir, force_refresh=force_refresh)
    data = loader.load()
    cube = data["cube"]
    timestamps = data["timestamps"] # seconds

    # 2. Prepare Time
    # Use pandas for robust parsing similar to src/algorithms.py
    try:
        dt_target = pd.to_datetime(target_time, utc=True)
    except:
        dt_target = pd.to_datetime(target_time)

    # We need a reference time (t0) to convert everything to days relative to t0
    # Let's use the min timestamp in the cube as t0
    timestamps_sec = timestamps_to_seconds(timestamps)
    t0_sec = np.min(timestamps_sec)
    t_days = (timestamps_sec - t0_sec) / 86400.0

    target_ts_sec = dt_target.timestamp()
    target_t_day = (target_ts_sec - t0_sec) / 86400.0

    bands, H, W = cube.shape
    out = np.full((H, W), np.nan, dtype=np.float32)

    # 3. Parallel Processing
    if n_jobs <= 0:
        n_jobs = cpu_count()

    print(f"[Zhu2015] Reconstructing {image} at {target_time} using LASSO (alpha={lasso_alpha})...")

    def _process_row(i):
        row_pred = np.full(W, np.nan, dtype=np.float32)
        for j in range(W):
            y = cube[:, i, j]
            pred, _ = fit_predict_pixel(t_days, y, target_t_day, lasso_alpha=lasso_alpha)
            row_pred[j] = pred
        return i, row_pred

    # Use tqdm for progress
    results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_process_row)(i) for i in range(H)
    )

    for i, row_pred in tqdm(results_gen, total=H, desc="Processing Rows"):
        out[i, :] = row_pred

    # 4. Save Output
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        transform = None
        if "transform" in data:
            transform = rasterio.Affine(*data["transform"])

        with rasterio.open(
            out_p, "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype=out.dtype,
            crs=data.get("crs_wkt"),
            transform=transform,
        ) as dst:
            dst.write(out, 1)
        print(f"[Success] Saved to: {out_p}")

    return out


# ── Compatibility bridge ──

MAX_ZHU_ORDER = 3


def _select_model_order(n_obs):
    if n_obs < 6:
        return 0
    if n_obs < 18:
        return 1
    if n_obs < 24:
        return 2
    return 3


def _select_unit_qa(n_obs, perennial_snow=False):
    if perennial_snow:
        return 3
    if n_obs >= 12:
        return 0
    if n_obs >= 6:
        return 1
    return 2


def _pack_zhu_coefficients(coef, order, max_order=MAX_ZHU_ORDER):
    packed = np.zeros(2 * max_order + 1, dtype=np.float64)
    if order <= 0:
        return packed
    harmonic_terms = min(2 * order, max(0, len(coef) - 1))
    if harmonic_terms > 0:
        packed[:harmonic_terms] = coef[:harmonic_terms]
    packed[-1] = coef[-1]
    return packed


def _make_full_design_matrix(x, max_order=MAX_ZHU_ORDER):
    cols = []
    w = 2 * np.pi / DAYS_PER_YEAR
    for k in range(1, max_order + 1):
        cols.append(np.cos(k * w * x))
        cols.append(np.sin(k * w * x))
    cols.append(x)
    return np.column_stack(cols)


def fit_zhu2015_pixel_params(
    t_days, y, lasso_alpha=0.001, max_segments=64, max_order=MAX_ZHU_ORDER,
):
    segment_count = max(1, max_segments)
    params = {
        "valid": False, "n_segments": 0, "max_order": int(max_order),
        "segment_start_days": np.full(segment_count, np.nan, dtype=np.float64),
        "segment_end_days": np.full(segment_count, np.nan, dtype=np.float64),
        "segment_orders": np.zeros(segment_count, dtype=np.int16),
        "segment_unit_qas": np.full(segment_count, 255, dtype=np.int16),
        "segment_has_model": np.zeros(segment_count, dtype=np.int8),
        "segment_median_values": np.full(segment_count, np.nan, dtype=np.float64),
        "segment_intercepts": np.zeros(segment_count, dtype=np.float64),
        "segment_coefficients": np.zeros((segment_count, 2 * max_order + 1), dtype=np.float64),
    }
    valid_mask = np.isfinite(y) & np.isfinite(t_days)
    if not np.any(valid_mask):
        return params
    t_valid = np.asarray(t_days[valid_mask], dtype=np.float64)
    y_valid = np.asarray(y[valid_mask], dtype=np.float64)
    n_obs = len(y_valid)
    order = _select_model_order(n_obs)
    unit_qa = _select_unit_qa(n_obs)
    if order == 0:
        params["valid"] = True
        params["n_segments"] = 1
        params["segment_start_days"][0] = float(t_valid[0])
        params["segment_end_days"][0] = float(t_valid[-1])
        params["segment_orders"][0] = 0
        params["segment_unit_qas"][0] = unit_qa
        params["segment_has_model"][0] = 0
        params["segment_median_values"][0] = float(np.median(y_valid))
        return params
    X = make_design_matrix(t_valid, order)
    clf = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=2000)
    clf.fit(X, y_valid)
    params["valid"] = True
    params["n_segments"] = 1
    params["segment_start_days"][0] = float(t_valid[0])
    params["segment_end_days"][0] = float(t_valid[-1])
    params["segment_orders"][0] = order
    params["segment_unit_qas"][0] = unit_qa
    params["segment_has_model"][0] = 1
    params["segment_intercepts"][0] = float(clf.intercept_)
    params["segment_coefficients"][0] = _pack_zhu_coefficients(
        np.asarray(clf.coef_, dtype=np.float64), order, max_order=max_order,
    )
    return params


def predict_zhu2015_from_params(params, target_t_day):
    if not params.get("valid", False) or int(params.get("n_segments", 0)) == 0:
        return np.nan, 255
    n_segments = int(params["n_segments"])
    starts = params["segment_start_days"][:n_segments]
    ends = params["segment_end_days"][:n_segments]
    unit_qas = params["segment_unit_qas"][:n_segments]
    has_model = params["segment_has_model"][:n_segments]
    median_vals = params["segment_median_values"][:n_segments]
    intercepts = params["segment_intercepts"][:n_segments]
    coeffs = params["segment_coefficients"][:n_segments]
    seg_idx = 0
    if target_t_day < starts[0]:
        qa_prefix = 1
    elif target_t_day > ends[0]:
        qa_prefix = 2
    else:
        qa_prefix = 0
    qa = qa_prefix * 10 + int(unit_qas[seg_idx])
    if not has_model[seg_idx]:
        return float(median_vals[seg_idx]), qa
    max_order = int(params.get("max_order", MAX_ZHU_ORDER))
    w = 2 * np.pi / DAYS_PER_YEAR
    ncols = 2 * max_order + 1
    row = np.empty(ncols, dtype=np.float64)
    for k in range(1, max_order + 1):
        row[2 * (k - 1)] = np.cos(k * w * target_t_day)
        row[2 * k - 1] = np.sin(k * w * target_t_day)
    row[-1] = target_t_day
    pred = float(intercepts[seg_idx] + row @ coeffs[seg_idx])
    return pred, qa


def predict_curve_pixel(t_days, y, target_t_days, lasso_alpha=0.001):
    params = fit_zhu2015_pixel_params(t_days, y, lasso_alpha=lasso_alpha)
    preds = np.zeros(len(target_t_days), dtype=np.float32)
    for i, target_t in enumerate(target_t_days):
        pred, _ = predict_zhu2015_from_params(params, float(target_t))
        preds[i] = pred
    return preds

# ── Additional compatibility functions ──

def extract_segments(t_days, y, lasso_alpha=0.001):
    valid_mask = np.isfinite(y) & np.isfinite(t_days)
    if not np.any(valid_mask):
        return []
    t_valid = np.asarray(t_days[valid_mask], dtype=np.float64)
    y_valid = np.asarray(y[valid_mask], dtype=np.float64)
    n_obs = len(y_valid)
    order = _select_model_order(n_obs)
    unit_qa = _select_unit_qa(n_obs)
    if order == 0:
        return [{
            "start_idx": 0, "end_idx": n_obs - 1,
            "clf": None, "order": 0, "unit_qa": unit_qa,
            "median_val": float(np.median(y_valid)),
        }]
    X = make_design_matrix(t_valid, order)
    clf = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=2000)
    clf.fit(X, y_valid)
    return [{
        "start_idx": 0, "end_idx": n_obs - 1,
        "clf": clf, "order": order, "unit_qa": unit_qa,
        "median_val": 0.0,
    }]


def predict_target(segments, t_days, target_t_day):
    if not segments:
        return np.nan, 255
    seg = segments[0]
    qa = 0 * 10 + seg["unit_qa"]
    if seg["clf"] is None:
        return seg["median_val"], qa
    pred = seg["clf"].predict(make_design_matrix(np.array([target_t_day]), seg["order"]))[0]
    return pred, qa
