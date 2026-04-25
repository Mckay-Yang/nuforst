import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from typing import Tuple, Optional, Union, List, Dict, Any
from datetime import datetime
from .data_loader import RSCube
from .nufrost import timestamps_to_seconds
from config import Args, build_args
from .logger import log as _log
import rasterio
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# Constants
DAYS_PER_YEAR = 365.25
MAX_ZHU_ORDER = 3


def _select_model_order(n_obs: int) -> int:
    """
    Select Zhu et al. (2015) harmonic model complexity.

    Returns
    -------
    0 : no harmonic model, use median fallback
    1 : simple model  (annual harmonic + trend)
    2 : advanced model (up to semi-annual harmonic + trend)
    3 : full model    (up to tri-modal harmonic + trend)
    """
    if n_obs < 6:
        return 0
    if n_obs < 18:
        return 1
    if n_obs < 24:
        return 2
    return 3


def _select_unit_qa(n_obs: int, perennial_snow: bool = False) -> int:
    if perennial_snow:
        return 3
    if n_obs >= 12:
        return 0
    if n_obs >= 6:
        return 1
    return 2

def _get_julian_date(timestamps: np.ndarray) -> np.ndarray:
    """Convert timestamps (seconds) to Julian dates (days)."""
    return timestamps / 86400.0

def make_design_matrix(x: np.ndarray, order: int) -> np.ndarray:
    """
    Construct design matrix for harmonic model with linear trend.
    x: array of time points (days)
    order: 1 (Simple), 2 (Advanced), 3 (Full)
    """
    cols = []
    w = 2 * np.pi / DAYS_PER_YEAR
    for k in range(1, order + 1):
        cols.append(np.cos(k * w * x))
        cols.append(np.sin(k * w * x))
    cols.append(x)
    return np.column_stack(cols)

def fit_model(t_days: np.ndarray, y: np.ndarray, lasso_alpha: float) -> Tuple[Optional[Lasso], int, int, float]:
    n_obs = len(y)

    unit_qa = _select_unit_qa(n_obs)
    order = _select_model_order(n_obs)

    if order == 0:
        return None, unit_qa, 0, 0.0

    X = make_design_matrix(t_days, order)
    clf = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, y)

    y_pred = clf.predict(X)
    res = y - y_pred
    rmse = np.sqrt(np.mean(res**2))

    return clf, unit_qa, order, rmse


def _predict_raw(coef: np.ndarray, intercept: float, X: np.ndarray) -> np.ndarray:
    return intercept + X @ coef


def _predict_single(coef: np.ndarray, intercept: float, x_day: float, order: int) -> float:
    w = 2 * np.pi / DAYS_PER_YEAR
    row = np.empty(2 * order + 1, dtype=np.float64)
    for k in range(1, order + 1):
        row[2 * (k - 1)] = np.cos(k * w * x_day)
        row[2 * k - 1] = np.sin(k * w * x_day)
    row[-1] = x_day
    return intercept + row @ coef

def extract_segments(t_days: np.ndarray, y: np.ndarray, lasso_alpha: float = 0.001) -> List[Dict[str, Any]]:
    n_obs = len(y)
    segments = []

    if n_obs < 12:
        clf, unit_qa, order, _ = fit_model(t_days, y, lasso_alpha)
        if clf is None:
            return [{
                'start_idx': 0,
                'end_idx': n_obs - 1,
                'clf': None,
                'order': 0,
                'unit_qa': unit_qa,
                'median_val': float(np.median(y))
            }]
        return [{
            'start_idx': 0,
            'end_idx': n_obs - 1,
            'clf': clf,
            'order': order,
            'unit_qa': unit_qa,
            'median_val': 0.0
        }]

    start_idx = 0
    while start_idx < n_obs:
        remaining = n_obs - start_idx
        if remaining < 6:
            segments.append({
                'start_idx': start_idx,
                'end_idx': n_obs - 1,
                'clf': None,
                'order': 0,
                'unit_qa': 2,
                'median_val': np.median(y[start_idx:])
            })
            break

        init_end = min(start_idx + 24, n_obs)
        if init_end - start_idx < 12:
            init_end = min(start_idx + 12, n_obs)

        t_init = t_days[start_idx:init_end]
        y_init = y[start_idx:init_end]

        clf, unit_qa, order, base_rmse = fit_model(t_init, y_init, lasso_alpha)
        if clf is None:
            segments.append({
                'start_idx': start_idx,
                'end_idx': n_obs - 1,
                'clf': None,
                'order': 0,
                'unit_qa': 2,
                'median_val': np.median(y_init)
            })
            break

        if unit_qa != 0:
            segments.append({
                'start_idx': start_idx,
                'end_idx': n_obs - 1,
                'clf': clf,
                'order': order,
                'unit_qa': unit_qa,
                'median_val': 0.0 if clf is not None else float(np.median(y_init))
            })
            break

        coef = np.asarray(clf.coef_, dtype=np.float64)
        intercept = float(clf.intercept_)
        break_idx = -1
        consecutive_anomalies = 0

        for i in range(init_end, n_obs):
            pred = _predict_single(coef, intercept, t_days[i], order)

            if i - start_idx >= 24:
                t_seg = t_days[start_idx:i]
                y_seg = y[start_idx:i]
                target_doy = t_days[i] % DAYS_PER_YEAR
                doy_seg = t_seg % DAYS_PER_YEAR
                diff = np.abs(doy_seg - target_doy)
                diff = np.minimum(diff, DAYS_PER_YEAR - diff)
                nearest_idx = np.argsort(diff)[:24]
                X_seg = make_design_matrix(t_seg, order)
                y_pred_seg = _predict_raw(coef, intercept, X_seg)
                res = y_seg - y_pred_seg
                current_rmse = np.sqrt(np.mean(res[nearest_idx]**2))
            else:
                current_rmse = base_rmse

            threshold = 2 * current_rmse
            threshold = max(threshold, 0.01)

            if abs(y[i] - pred) > threshold:
                consecutive_anomalies += 1
                if consecutive_anomalies >= 6:
                    break_idx = i - 5
                    break
            else:
                consecutive_anomalies = 0

        if break_idx != -1:
            seg_end = break_idx - 1
            if seg_end - start_idx < 6:
                 segments.append({
                     'start_idx': start_idx,
                     'end_idx': seg_end,
                     'clf': None,
                     'order': 0,
                     'unit_qa': 2,
                     'median_val': np.median(y[start_idx:seg_end+1])
                 })
            else:
                 clf_final, u_qa, ord_f, _ = fit_model(t_days[start_idx:seg_end+1], y[start_idx:seg_end+1], lasso_alpha)
                 if clf_final is None:
                     segments.append({
                         'start_idx': start_idx,
                         'end_idx': seg_end,
                         'clf': None,
                         'order': 0,
                         'unit_qa': 2,
                         'median_val': np.median(y[start_idx:seg_end+1])
                     })
                 else:
                     segments.append({
                         'start_idx': start_idx,
                         'end_idx': seg_end,
                         'clf': clf_final,
                         'order': ord_f,
                         'unit_qa': u_qa,
                         'median_val': 0.0
                     })
            start_idx = break_idx
        else:
            clf_final, u_qa, ord_f, _ = fit_model(t_days[start_idx:n_obs], y[start_idx:n_obs], lasso_alpha)
            if clf_final is None:
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': n_obs - 1,
                    'clf': None,
                    'order': 0,
                    'unit_qa': 2,
                    'median_val': np.median(y[start_idx:n_obs])
                })
            else:
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': n_obs - 1,
                    'clf': clf_final,
                    'order': ord_f,
                    'unit_qa': u_qa,
                    'median_val': 0.0
                })
            break

    return segments


def _pack_zhu_coefficients(coef: np.ndarray, order: int, max_order: int = MAX_ZHU_ORDER) -> np.ndarray:
    packed = np.zeros(2 * max_order + 1, dtype=np.float64)
    if order <= 0:
        return packed
    harmonic_terms = min(2 * order, max(0, len(coef) - 1))
    if harmonic_terms > 0:
        packed[:harmonic_terms] = coef[:harmonic_terms]
    packed[-1] = coef[-1]
    return packed


def _make_full_design_matrix(x: np.ndarray, max_order: int = MAX_ZHU_ORDER) -> np.ndarray:
    cols = []
    w = 2 * np.pi / DAYS_PER_YEAR
    for k in range(1, max_order + 1):
        cols.append(np.cos(k * w * x))
        cols.append(np.sin(k * w * x))
    cols.append(x)
    return np.column_stack(cols)


def fit_zhu2015_pixel_params(
    t_days: np.ndarray,
    y: np.ndarray,
    lasso_alpha: float = 0.001,
    max_segments: int = 10,
    max_order: int = MAX_ZHU_ORDER,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "valid": False,
        "n_segments": 0,
        "max_order": int(max_order),
        "segment_start_days": np.full(max_segments, np.nan, dtype=np.float64),
        "segment_end_days": np.full(max_segments, np.nan, dtype=np.float64),
        "segment_orders": np.zeros(max_segments, dtype=np.int16),
        "segment_unit_qas": np.full(max_segments, 255, dtype=np.int16),
        "segment_has_model": np.zeros(max_segments, dtype=np.int8),
        "segment_median_values": np.full(max_segments, np.nan, dtype=np.float64),
        "segment_intercepts": np.zeros(max_segments, dtype=np.float64),
        "segment_coefficients": np.zeros((max_segments, 2 * max_order + 1), dtype=np.float64),
    }

    valid_mask = np.isfinite(y) & np.isfinite(t_days)
    if not np.any(valid_mask):
        return params

    t_valid = np.asarray(t_days[valid_mask], dtype=np.float64)
    y_valid = np.asarray(y[valid_mask], dtype=np.float64)
    segments = extract_segments(t_valid, y_valid, lasso_alpha)
    if not segments:
        return params

    n_segments = min(len(segments), max_segments)
    params["valid"] = True
    params["n_segments"] = n_segments

    for idx, seg in enumerate(segments[:n_segments]):
        params["segment_start_days"][idx] = float(t_valid[seg['start_idx']])
        params["segment_end_days"][idx] = float(t_valid[seg['end_idx']])
        params["segment_orders"][idx] = int(seg['order'])
        params["segment_unit_qas"][idx] = int(seg['unit_qa'])
        params["segment_median_values"][idx] = float(seg['median_val'])
        if seg['clf'] is not None:
            params["segment_has_model"][idx] = 1
            params["segment_intercepts"][idx] = float(seg['clf'].intercept_)
            params["segment_coefficients"][idx] = _pack_zhu_coefficients(np.asarray(seg['clf'].coef_, dtype=np.float64), int(seg['order']), max_order=max_order)

    return params


def predict_zhu2015_from_params(params: Dict[str, Any], target_t_day: float) -> Tuple[float, int]:
    if not bool(params.get("valid", False)) or int(params.get("n_segments", 0)) == 0:
        return np.nan, 255

    n_segments = int(params["n_segments"])
    starts = params["segment_start_days"][:n_segments]
    ends = params["segment_end_days"][:n_segments]
    unit_qas = params["segment_unit_qas"][:n_segments]
    has_model = params["segment_has_model"][:n_segments]
    median_vals = params["segment_median_values"][:n_segments]
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

    max_order = int(params["max_order"])
    w = 2 * np.pi / DAYS_PER_YEAR
    ncols = 2 * max_order + 1
    row = np.empty(ncols, dtype=np.float64)
    for k in range(1, max_order + 1):
        row[2 * (k - 1)] = np.cos(k * w * target_t_day)
        row[2 * k - 1] = np.sin(k * w * target_t_day)
    row[-1] = target_t_day
    pred = float(intercepts[seg_idx] + row @ coeffs[seg_idx])
    return pred, qa

def predict_target(segments: List[Dict[str, Any]], t_days: np.ndarray, target_t_day: float) -> Tuple[float, int]:
    if not segments:
        return np.nan, 255

    for seg in segments:
        if t_days[seg['start_idx']] <= target_t_day <= t_days[seg['end_idx']]:
            qa = 0 * 10 + seg['unit_qa']
            if seg['clf'] is None:
                return seg['median_val'], qa
            else:
                pred = seg['clf'].predict(make_design_matrix(np.array([target_t_day]), seg['order']))[0]
                return pred, qa

    if target_t_day < t_days[segments[0]['start_idx']]:
        seg = segments[0]
        qa = 1 * 10 + seg['unit_qa']
        if seg['clf'] is None:
            return seg['median_val'], qa
        else:
            pred = seg['clf'].predict(make_design_matrix(np.array([target_t_day]), seg['order']))[0]
            return pred, qa

    seg = segments[-1]
    qa = 2 * 10 + seg['unit_qa']
    if seg['clf'] is None:
        return seg['median_val'], qa
    else:
        pred = seg['clf'].predict(make_design_matrix(np.array([target_t_day]), seg['order']))[0]
        return pred, qa

def fit_predict_pixel(
    t_days: np.ndarray,
    y: np.ndarray,
    target_t_day: float,
    lasso_alpha: float = 0.001
) -> Tuple[float, int]:
    params = fit_zhu2015_pixel_params(t_days, y, lasso_alpha=lasso_alpha)
    return predict_zhu2015_from_params(params, target_t_day)

def predict_curve_pixel(
    t_days: np.ndarray,
    y: np.ndarray,
    target_t_days: np.ndarray,
    lasso_alpha: float = 0.001
) -> np.ndarray:
    params = fit_zhu2015_pixel_params(t_days, y, lasso_alpha=lasso_alpha)
    preds = np.zeros(len(target_t_days), dtype=np.float32)
    for i, target_t in enumerate(target_t_days):
        pred, _ = predict_zhu2015_from_params(params, float(target_t))
        preds[i] = pred
    return preds

def reconstruct_zhu2015(
    image: Union[str, Path],
    target_time: str,
    output_path: Optional[Union[str, Path]] = None,
    lasso_alpha: float = 0.0001,
    n_jobs: int = -1,
    cache_dir: Union[str, Path] = "data/cache/local",
    force_refresh: bool = False
) -> np.ndarray:
    """
    Reconstruct Landsat image using Zhu et al. (2015) method.
    Returns: A 3D numpy array of shape (2, H, W).
             Band 1 is the predicted reflectance.
             Band 2 is the QA band.
    """
    # 1. Load Data
    loader = RSCube(image, cache_dir=cache_dir, force_refresh=force_refresh)
    data = loader.load()
    cube = np.ma.filled(data["cube"], np.nan)
    timestamps = data["timestamps"] # seconds

    # 2. Prepare Time
    try:
        dt_target = pd.to_datetime(target_time, utc=True)
    except:
        dt_target = pd.to_datetime(target_time)

    timestamps_sec = timestamps_to_seconds(timestamps)
    t0_sec = np.min(timestamps_sec)
    t_days = (timestamps_sec - t0_sec) / 86400.0

    target_ts_sec = dt_target.timestamp()
    target_t_day = (target_ts_sec - t0_sec) / 86400.0

    bands, H, W = cube.shape
    out = np.full((2, H, W), np.nan, dtype=np.float32)

    # 3. Parallel Processing
    if n_jobs <= 0:
        n_jobs = cpu_count()

    _log("reconstruct_zhu2015", f"Reconstructing {image} at {target_time} using LASSO (alpha={lasso_alpha})")

    def _process_row(i):
        row_pred = np.full((2, W), np.nan, dtype=np.float32)
        for j in range(W):
            y = cube[:, i, j]
            pred, qa = fit_predict_pixel(t_days, y, target_t_day, lasso_alpha=lasso_alpha)
            row_pred[0, j] = pred
            row_pred[1, j] = qa
        return i, row_pred

    results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_process_row)(i) for i in range(H)
    )

    for i, row_pred in tqdm(results_gen, total=H, desc="Processing Rows"):
        out[:, i, :] = row_pred

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
            count=2,
            dtype=out.dtype,
            crs=data.get("crs_wkt"),
            transform=transform,
        ) as dst:
            dst.write(out[0], 1)
            dst.write(out[1], 2)
        _log("reconstruct_zhu2015", f"Saved to: {out_p}")

    return out
