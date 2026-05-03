import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from typing import Tuple, Optional, Union, Any, Dict as DictType, List
from datetime import datetime
from .data_loader import RSCube
from .nufrost import timestamps_to_seconds
from config import Args, Zhu2015Args, build_args
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

def make_design_matrix(x: np.ndarray, order: int, ref_x_mean: float = None) -> np.ndarray:
    cols = []
    w = 2 * np.pi / DAYS_PER_YEAR
    for k in range(1, order + 1):
        cols.append(np.cos(k * w * x))
        cols.append(np.sin(k * w * x))
    if ref_x_mean is not None:
        cols.append(np.asarray(x) - float(ref_x_mean))
    else:
        cols.append(np.asarray(x))
    return np.column_stack(cols)


def fit_model(t_days: np.ndarray, y: np.ndarray, lasso_alpha: float):
    n_obs = len(y)
    unit_qa = _select_unit_qa(n_obs)
    order = _select_model_order(n_obs)
    if order == 0:
        return None, unit_qa, 0, 0.0, float(np.mean(t_days)) if len(t_days) else 0.0

    x_mean = float(np.mean(t_days))
    X = make_design_matrix(t_days, order, ref_x_mean=x_mean)
    clf = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, y)

    y_pred = clf.predict(X)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return clf, unit_qa, order, rmse, x_mean


def _predict_raw(coef: np.ndarray, intercept: float, X: np.ndarray) -> np.ndarray:
    return intercept + X @ coef


def _predict_single(coef: np.ndarray, intercept: float, x_day: float, order: int, x_mean: float) -> float:
    X = make_design_matrix(np.array([x_day], dtype=np.float64), order, ref_x_mean=x_mean)
    return float(intercept + X[0] @ coef)

def fit_predict_pixel(
    t_days: np.ndarray,
    y: np.ndarray,
    target_t_day: float,
    lasso_alpha: float = 0.001
) -> Tuple[float, int]:
    valid_mask = np.isfinite(y)
    if not np.any(valid_mask):
        return np.nan, 0

    t_valid = t_days[valid_mask]
    y_valid = y[valid_mask]
    n_obs = len(y_valid)

    if n_obs < 6:
        return np.median(y_valid), 0

    if 6 <= n_obs < 18:
        order = 1
        model_id = 1
    elif 18 <= n_obs < 24:
        order = 2
        model_id = 2
    else:
        order = 3
        model_id = 3

    x_mean = float(np.mean(t_valid))
    X = make_design_matrix(t_valid, order, ref_x_mean=x_mean)

    clf = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=1000)
    clf.fit(X, y_valid)

    X_target = make_design_matrix(np.array([target_t_day]), order, ref_x_mean=x_mean)
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


def _make_full_design_matrix(x, max_order=MAX_ZHU_ORDER, ref_x_mean=None):
    cols = []
    w = 2 * np.pi / DAYS_PER_YEAR
    for k in range(1, max_order + 1):
        cols.append(np.cos(k * w * x))
        cols.append(np.sin(k * w * x))
    if ref_x_mean is not None:
        cols.append(np.asarray(x) - float(ref_x_mean))
    else:
        cols.append(np.asarray(x))
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
        "segment_x_means": np.zeros(segment_count, dtype=np.float64),
        "segment_intercepts": np.zeros(segment_count, dtype=np.float64),
        "segment_coefficients": np.zeros((segment_count, 2 * max_order + 1), dtype=np.float64),
        "x_mean": np.float64(0.0),
    }
    valid_mask = np.isfinite(y) & np.isfinite(t_days)
    if not np.any(valid_mask):
        return params
    t_valid = np.asarray(t_days[valid_mask], dtype=np.float64)
    y_valid = np.asarray(y[valid_mask], dtype=np.float64)
    n_obs = len(y_valid)

    if n_obs >= 12:
        clf_pre, unit_qa, order, base_rmse, x_mean = fit_model(t_valid, y_valid, lasso_alpha)
        if clf_pre is not None and base_rmse > 0:
            coef_pre = np.asarray(clf_pre.coef_, dtype=np.float64)
            intercept_pre = float(clf_pre.intercept_)
            consecutive = 0
            has_break = False
            for i in range(n_obs):
                pred = _predict_single(coef_pre, intercept_pre, t_valid[i], order, x_mean)
                if abs(y_valid[i] - pred) > max(2.0 * base_rmse, 0.01):
                    consecutive += 1
                    if consecutive >= 6:
                        has_break = True
                        break
                else:
                    consecutive = 0
            if not has_break:
                params["valid"] = True
                params["n_segments"] = 1
                params["segment_start_days"][0] = float(t_valid[0])
                params["segment_end_days"][0] = float(t_valid[-1])
                params["segment_orders"][0] = order
                params["segment_unit_qas"][0] = unit_qa
                params["segment_has_model"][0] = 1
                params["segment_intercepts"][0] = float(clf_pre.intercept_)
                params["segment_coefficients"][0] = _pack_zhu_coefficients(
                    coef_pre, order, max_order=max_order,
                )
                params["segment_x_means"][0] = float(x_mean)
                params["x_mean"] = np.float64(x_mean)
                return params

    segments = extract_segments(t_valid, y_valid, lasso_alpha=lasso_alpha)
    if not segments:
        return params

    params["valid"] = True
    params["n_segments"] = min(len(segments), segment_count)
    for idx, seg in enumerate(segments[:segment_count]):
        params["segment_start_days"][idx] = float(t_valid[seg["start_idx"]])
        params["segment_end_days"][idx] = float(t_valid[seg["end_idx"]])
        params["segment_orders"][idx] = int(seg["order"])
        params["segment_unit_qas"][idx] = int(seg["unit_qa"])
        params["segment_median_values"][idx] = float(seg["median_val"])
        params["segment_x_means"][idx] = float(seg.get("x_mean", 0.0))
        if seg["clf"] is not None:
            params["segment_has_model"][idx] = 1
            params["segment_intercepts"][idx] = float(seg["clf"].intercept_)
            params["segment_coefficients"][idx] = _pack_zhu_coefficients(
                np.asarray(seg["clf"].coef_, dtype=np.float64), int(seg["order"]), max_order=max_order,
            )
    params["x_mean"] = np.float64(0.0)
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
    x_means = params.get("segment_x_means", np.zeros(n_segments, dtype=np.float64))[:n_segments]
    intercepts = params["segment_intercepts"][:n_segments]
    coeffs = params["segment_coefficients"][:n_segments]
    seg_idx = -1
    qa_prefix = 0
    for idx in range(n_segments):
        if starts[idx] <= target_t_day <= ends[idx]:
            seg_idx = idx
            break
    if seg_idx < 0:
        if target_t_day < starts[0]:
            seg_idx = 0
            qa_prefix = 1
        else:
            seg_idx = n_segments - 1
            qa_prefix = 2
    qa = qa_prefix * 10 + int(unit_qas[seg_idx])
    if not has_model[seg_idx]:
        return float(median_vals[seg_idx]), qa
    x_mean = float(x_means[seg_idx])
    max_order = int(params.get("max_order", MAX_ZHU_ORDER))
    w = 2 * np.pi / DAYS_PER_YEAR
    ncols = 2 * max_order + 1
    row = np.empty(ncols, dtype=np.float64)
    for k in range(1, max_order + 1):
        row[2 * (k - 1)] = np.cos(k * w * target_t_day)
        row[2 * k - 1] = np.sin(k * w * target_t_day)
    row[-1] = target_t_day - x_mean
    pred = float(intercepts[seg_idx] + row @ coeffs[seg_idx])
    return pred, qa


def predict_curve_pixel(t_days, y, target_t_days, lasso_alpha=0.001):
    params = fit_zhu2015_pixel_params(t_days, y, lasso_alpha=lasso_alpha)
    preds = np.zeros(len(target_t_days), dtype=np.float32)
    for i, target_t in enumerate(target_t_days):
        pred, _ = predict_zhu2015_from_params(params, float(target_t))
        preds[i] = pred
    return preds


def fit_predict_pixel_segments(
    t_days: np.ndarray,
    y: np.ndarray,
    target_t_day: float,
    lasso_alpha: float = 0.001,
) -> Tuple[float, int]:
    params = fit_zhu2015_pixel_params(t_days, y, lasso_alpha=lasso_alpha)
    return predict_zhu2015_from_params(params, target_t_day)

# ── Additional compatibility functions ──

def extract_segments(t_days, y, lasso_alpha=0.001):
    valid_mask = np.isfinite(y) & np.isfinite(t_days)
    if not np.any(valid_mask):
        return []
    t_valid = np.asarray(t_days[valid_mask], dtype=np.float64)
    y_valid = np.asarray(y[valid_mask], dtype=np.float64)
    n_obs = len(y_valid)
    segments = []

    if n_obs < 12:
        clf, unit_qa, order, _, x_mean = fit_model(t_valid, y_valid, lasso_alpha)
        return [{
            "start_idx": 0, "end_idx": n_obs - 1,
            "clf": clf, "order": order, "unit_qa": unit_qa,
            "median_val": float(np.median(y_valid)) if clf is None else 0.0,
            "x_mean": x_mean,
        }]

    start_idx = 0
    while start_idx < n_obs:
        remaining = n_obs - start_idx
        if remaining < 6:
            segments.append({
                "start_idx": start_idx, "end_idx": n_obs - 1,
                "clf": None, "order": 0, "unit_qa": 2,
                "median_val": float(np.median(y_valid[start_idx:])),
                "x_mean": float(np.mean(t_valid[start_idx:])),
            })
            break

        init_end = min(start_idx + 24, n_obs)
        if init_end - start_idx < 12:
            init_end = min(start_idx + 12, n_obs)

        clf, unit_qa, order, base_rmse, x_mean = fit_model(
            t_valid[start_idx:init_end], y_valid[start_idx:init_end], lasso_alpha,
        )
        if clf is None:
            segments.append({
                "start_idx": start_idx, "end_idx": n_obs - 1,
                "clf": None, "order": 0, "unit_qa": 2,
                "median_val": float(np.median(y_valid[start_idx:init_end])),
                "x_mean": float(np.mean(t_valid[start_idx:init_end])),
            })
            break
        if unit_qa != 0:
            segments.append({
                "start_idx": start_idx, "end_idx": n_obs - 1,
                "clf": clf, "order": order, "unit_qa": unit_qa,
                "median_val": 0.0, "x_mean": x_mean,
            })
            break

        coef = np.asarray(clf.coef_, dtype=np.float64)
        intercept = float(clf.intercept_)
        break_idx = -1
        consecutive_anomalies = 0
        for i in range(init_end, n_obs):
            pred = _predict_single(coef, intercept, t_valid[i], order, x_mean)
            if i - start_idx >= 24:
                t_seg = t_valid[start_idx:i]
                y_seg = y_valid[start_idx:i]
                target_doy = t_valid[i] % DAYS_PER_YEAR
                doy_seg = t_seg % DAYS_PER_YEAR
                diff = np.abs(doy_seg - target_doy)
                diff = np.minimum(diff, DAYS_PER_YEAR - diff)
                nearest_idx = np.argsort(diff)[:24]
                X_seg = make_design_matrix(t_seg, order, ref_x_mean=x_mean)
                res = y_seg - _predict_raw(coef, intercept, X_seg)
                current_rmse = float(np.sqrt(np.mean(res[nearest_idx] ** 2)))
            else:
                current_rmse = base_rmse
            threshold = max(2.0 * current_rmse, 0.01)
            if abs(y_valid[i] - pred) > threshold:
                consecutive_anomalies += 1
                if consecutive_anomalies >= 6:
                    break_idx = i - 5
                    break
            else:
                consecutive_anomalies = 0

        if break_idx != -1:
            seg_end = break_idx - 1
            seg_t = t_valid[start_idx:seg_end + 1]
            seg_y = y_valid[start_idx:seg_end + 1]
            if len(seg_y) < 6:
                segments.append({
                    "start_idx": start_idx, "end_idx": seg_end,
                    "clf": None, "order": 0, "unit_qa": 2,
                    "median_val": float(np.median(seg_y)),
                    "x_mean": float(np.mean(seg_t)),
                })
            else:
                clf_final, unit_qa, order, _, x_mean = fit_model(seg_t, seg_y, lasso_alpha)
                segments.append({
                    "start_idx": start_idx, "end_idx": seg_end,
                    "clf": clf_final, "order": order, "unit_qa": unit_qa,
                    "median_val": float(np.median(seg_y)) if clf_final is None else 0.0,
                    "x_mean": x_mean,
                })
            start_idx = break_idx
        else:
            seg_t = t_valid[start_idx:n_obs]
            seg_y = y_valid[start_idx:n_obs]
            clf_final, unit_qa, order, _, x_mean = fit_model(seg_t, seg_y, lasso_alpha)
            segments.append({
                "start_idx": start_idx, "end_idx": n_obs - 1,
                "clf": clf_final, "order": order, "unit_qa": unit_qa,
                "median_val": float(np.median(seg_y)) if clf_final is None else 0.0,
                "x_mean": x_mean,
            })
            break

    return segments


def predict_target(segments, t_days, target_t_day):
    if not segments:
        return np.nan, 255
    seg = None
    qa_prefix = 0
    for candidate in segments:
        if t_days[candidate["start_idx"]] <= target_t_day <= t_days[candidate["end_idx"]]:
            seg = candidate
            break
    if seg is None:
        if target_t_day < t_days[segments[0]["start_idx"]]:
            seg = segments[0]
            qa_prefix = 1
        else:
            seg = segments[-1]
            qa_prefix = 2
    qa = qa_prefix * 10 + seg["unit_qa"]
    if seg["clf"] is None:
        return seg["median_val"], qa
    x_mean = seg.get("x_mean", 0.0)
    pred = seg["clf"].predict(make_design_matrix(np.array([target_t_day]), seg["order"], ref_x_mean=x_mean))[0]
    return pred, qa
