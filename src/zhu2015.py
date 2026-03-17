import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from typing import Tuple, Optional, Union
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
