import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Any, Dict as DictType
from .data_loader import RSCube
from .nufrost import timestamps_to_seconds
from config import Args, build_args
import rasterio
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# Constants
DAYS_PER_YEAR = 365.25

def make_harmonic_matrix(t: np.ndarray, frequencies: List[float]) -> np.ndarray:
    """
    Construct design matrix for harmonic analysis.
    t: time points (e.g. days)
    frequencies: list of frequencies to fit (e.g. [1, 2] for annual and semi-annual if t in years)
                 Or if t is in days, frequencies might be [1/365.25, 2/365.25].

    The paper says: "NOF... define how many frequencies are used and how large their corresponding period... is"
    Usually NOF=3 implies: Mean (freq=0), Annual, Semi-annual.

    We'll assume t is in DAYS.
    Base frequency w = 2 * pi / T.
    The paper uses specific periods.
    """
    cols = []
    # Mean (Frequency 0) is handled by intercept or column of ones
    cols.append(np.ones_like(t))

    for f in frequencies:
        if f == 0:
            continue
        # Omega = 2 * pi * f
        # If f is "1 cycle per year" and t is days: f = 1/365.25
        w = 2 * np.pi * f
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))

    return np.column_stack(cols)

def hants_pixel(
    t: np.ndarray,
    y: np.ndarray,
    target_t: float,
    nof: int = 3,
    sf: str = 'low',
    valid_min: float = None,
    valid_max: float = None,
    fet: float = 0.05,
    dod: int = 5,
    period: float = 365.25,
) -> float:
    params = fit_hants_pixel_params(
        t, y, nof=nof, sf=sf, valid_min=valid_min, valid_max=valid_max,
        fet=fet, dod=dod, period=period,
    )
    return predict_hants_from_params(params, target_t)

def reconstruct_hants(
    image: str,
    target_time: str,
    output_path: Optional[str] = None,
    nof: int = 3,
    sf: str = 'low',
    valid_min: float = None,
    valid_max: float = None,
    fet: float = 0.05,
    dod: int = 5,
    n_jobs: int = -1,
    cache_dir: str = "./cache",
    force_refresh: bool = False,
) -> np.ndarray:
    # 1. Load Data
    loader = RSCube(image, cache_dir=cache_dir, force_refresh=force_refresh)
    data = loader.load()
    cube = data["cube"]
    timestamps = data["timestamps"]

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

    _, H, W = cube.shape
    out = np.full((H, W), np.nan, dtype=np.float32)

    # 3. Parallel Processing
    if n_jobs <= 0:
        n_jobs = cpu_count()

    print(f"[HANTS] Reconstructing {image} at {target_time} (NOF={nof}, SF={sf}, FET={fet})...")

    def _process_row(i):
        row_pred = np.full(W, np.nan, dtype=np.float32)
        for j in range(W):
            y = cube[:, i, j]
            pred = hants_pixel(
                t_days, y, target_t_day,
                nof=nof, sf=sf,
                valid_min=valid_min, valid_max=valid_max,
                fet=fet, dod=dod,
            )
            row_pred[j] = pred
        return i, row_pred

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


def _apply_hants_valid_mask(y, valid_min=None, valid_max=None):
    valid_mask = np.isfinite(y)
    if valid_min is not None:
        valid_mask &= (y >= valid_min)
    if valid_max is not None:
        valid_mask &= (y <= valid_max)
    return valid_mask


def _fit_hants_coeffs(t_curr, y_curr, freqs):
    if len(y_curr) == 0:
        return None
    X = make_harmonic_matrix(t_curr, freqs)
    if X.shape[0] < X.shape[1]:
        return None
    coeffs, _, _, _ = np.linalg.lstsq(X, y_curr, rcond=None)
    return coeffs


def fit_hants_pixel_params(t, y, nof=3, sf='low', valid_min=None, valid_max=None, fet=0.05, dod=5, period=365.25):
    coeff_count = 1 + 2 * max(0, nof - 1)
    params = {
        "valid": False, "nof": int(nof), "period": float(period),
        "coeffs": np.full(coeff_count, np.nan, dtype=np.float64),
        "fill_value": float(np.nanmedian(y)) if np.isfinite(y).any() else np.nan,
        "n_iterations": 0,
    }
    freqs = [i / period for i in range(1, nof)]
    num_params = 1 + 2 * (nof - 1)

    valid_mask = np.isfinite(y)
    if valid_min is not None:
        valid_mask &= (y >= valid_min)
    if valid_max is not None:
        valid_mask &= (y <= valid_max)
    if np.sum(valid_mask) == 0:
        return params

    t_curr = t[valid_mask].copy()
    y_curr = y[valid_mask].copy()
    if len(y_curr) < num_params + dod:
        return params

    coeffs = None
    for it in range(min(len(y_curr), 50)):
        n_obs = len(y_curr)
        if n_obs < num_params + dod:
            break
        X = make_harmonic_matrix(t_curr, freqs)
        XTX = X.T @ X
        XTy = X.T @ y_curr
        try:
            coeffs = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            break
        y_pred_curr = X @ coeffs
        residuals = y_curr - y_pred_curr
        if sf == 'low':
            bad = residuals < -fet
        elif sf == 'high':
            bad = residuals > fet
        else:
            bad = np.abs(residuals) > fet
        if not np.any(bad):
            params["n_iterations"] = it + 1
            break
        mask_keep = ~bad
        t_curr = t_curr[mask_keep]
        y_curr = y_curr[mask_keep]
        params["n_iterations"] = it + 1

    if len(y_curr) < num_params + dod:
        return params
    if coeffs is None:
        coeffs = _fit_hants_coeffs(t_curr, y_curr, freqs)
    if coeffs is None:
        return params
    params["valid"] = True
    params["coeffs"][:len(coeffs)] = coeffs
    return params


def predict_hants_from_params(params, target_t):
    if not bool(params.get("valid", False)):
        return float(params.get("fill_value", np.nan))
    coeffs = np.asarray(params["coeffs"], dtype=np.float64)
    nof = int(params["nof"])
    period = float(params["period"])
    freqs = [i / period for i in range(1, nof)]
    X_target = make_harmonic_matrix(np.array([target_t], dtype=np.float64), freqs)
    return float((X_target @ coeffs[:X_target.shape[1]])[0])


def predict_hants_curve_from_params(params, target_t_array):
    return np.array([predict_hants_from_params(params, float(t)) for t in target_t_array], dtype=np.float64)


def hants_curve_pixel(t, y, target_t_array, nof=3, sf='low', valid_min=None, valid_max=None, fet=0.05, dod=5, period=365.25):
    params = fit_hants_pixel_params(t, y, nof=nof, sf=sf, valid_min=valid_min, valid_max=valid_max, fet=fet, dod=dod, period=period)
    return predict_hants_curve_from_params(params, target_t_array)
