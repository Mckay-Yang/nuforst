import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from .data_loader import RSCube
from .nufrost import timestamps_to_seconds
from config import Args, build_args
import rasterio
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# Constants
DAYS_PER_YEAR = 365.25


def _apply_hants_valid_mask(
    y: np.ndarray,
    sf: str,
    idrt: Optional[float],
) -> np.ndarray:
    """
    Build the validity mask used by HANTS.

    Notes
    -----
    Roerink et al. (2000) describe IDRT as an *invalid data rejection*
    threshold. In their NDVI example, with ``sf='low'`` they still reject very
    high values (IDRT = 0.7). This means IDRT works in the opposite direction
    of the suppression flag:

    - ``sf='low'``  -> reject implausibly high values above IDRT
    - ``sf='high'`` -> reject implausibly low values below IDRT
    """
    valid_mask = np.isfinite(y)
    if idrt is None:
        return valid_mask

    if sf == 'low':
        valid_mask &= (y <= idrt)
    elif sf == 'high':
        valid_mask &= (y >= idrt)
    else:
        valid_mask &= np.isfinite(y)

    return valid_mask


def _fit_hants_coeffs(
    t_curr: np.ndarray,
    y_curr: np.ndarray,
    freqs: List[float],
) -> Optional[np.ndarray]:
    if len(y_curr) == 0:
        return None
    X = make_harmonic_matrix(t_curr, freqs)
    if X.shape[0] < X.shape[1]:
        return None
    coeffs, _, _, _ = np.linalg.lstsq(X, y_curr, rcond=None)
    return coeffs

def make_harmonic_matrix(t: np.ndarray, frequencies: List[float]) -> np.ndarray:
    """
    Construct design matrix for harmonic analysis.
    """
    cols = []
    cols.append(np.ones_like(t))
    for f in frequencies:
        if f == 0:
            continue
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
    idrt: float = None,
    fet: float = 0.05,
    dod: int = 5,
    period: float = 365.25
) -> float:
    """
    Apply HANTS algorithm to a single pixel.
    """
    valid_mask = _apply_hants_valid_mask(y, sf=sf, idrt=idrt) & np.isfinite(t)

    t_valid = t[valid_mask]
    y_valid = y[valid_mask]

    t_curr = t[valid_mask]
    y_curr = y[valid_mask]

    freqs = [i / period for i in range(1, nof)]
    num_params = 1 + 2 * (nof - 1)

    coeffs = None
    max_iter = len(y_curr) # Allow removing up to all points
    for _ in range(max_iter):
        n_obs = len(y_curr)
        if n_obs < num_params + dod:
            break

        X = make_harmonic_matrix(t_curr, freqs)
        coeffs, _, _, _ = np.linalg.lstsq(X, y_curr, rcond=None)

        y_pred_curr = X @ coeffs
        residuals = y_curr - y_pred_curr

        if sf == 'low':
            worst_idx = np.argmin(residuals)
            if residuals[worst_idx] < -fet:
                has_bad = True
            else:
                has_bad = False
        elif sf == 'high':
            worst_idx = np.argmax(residuals)
            if residuals[worst_idx] > fet:
                has_bad = True
            else:
                has_bad = False
        else:
            worst_idx = np.argmax(np.abs(residuals))
            if np.abs(residuals)[worst_idx] > fet:
                has_bad = True
            else:
                has_bad = False

        if not has_bad:
            break

        mask_keep = np.ones(n_obs, dtype=bool)
        mask_keep[worst_idx] = False
        t_curr = t_curr[mask_keep]
        y_curr = y_curr[mask_keep]

    if coeffs is None:
        coeffs = _fit_hants_coeffs(t_curr, y_curr, freqs)

    if coeffs is None:
        return np.nan

    X_target = make_harmonic_matrix(np.array([target_t]), freqs)
    y_target = (X_target @ coeffs)[0]

    return y_target

def hants_curve_pixel(
    t: np.ndarray,
    y: np.ndarray,
    target_t_array: np.ndarray,
    nof: int = 3,
    sf: str = 'low',
    idrt: float = None,
    fet: float = 0.05,
    dod: int = 5,
    period: float = 365.25
) -> np.ndarray:
    """
    Apply HANTS algorithm to a single pixel and predict for an array of times.
    """
    valid_mask = _apply_hants_valid_mask(y, sf=sf, idrt=idrt) & np.isfinite(t)

    if np.sum(valid_mask) == 0:
        return np.full(len(target_t_array), np.nan)

    t_curr = t[valid_mask]
    y_curr = y[valid_mask]

    freqs = [i / period for i in range(1, nof)]
    num_params = 1 + 2 * (nof - 1)

    coeffs = None
    max_iter = len(y_curr)
    for _ in range(max_iter):
        n_obs = len(y_curr)
        if n_obs < num_params + dod:
            break

        X = make_harmonic_matrix(t_curr, freqs)
        coeffs, _, _, _ = np.linalg.lstsq(X, y_curr, rcond=None)

        y_pred_curr = X @ coeffs
        residuals = y_curr - y_pred_curr

        if sf == 'low':
            worst_idx = np.argmin(residuals)
            if residuals[worst_idx] < -fet:
                has_bad = True
            else:
                has_bad = False
        elif sf == 'high':
            worst_idx = np.argmax(residuals)
            if residuals[worst_idx] > fet:
                has_bad = True
            else:
                has_bad = False
        else:
            worst_idx = np.argmax(np.abs(residuals))
            if np.abs(residuals)[worst_idx] > fet:
                has_bad = True
            else:
                has_bad = False

        if not has_bad:
            break

        mask_keep = np.ones(n_obs, dtype=bool)
        mask_keep[worst_idx] = False
        t_curr = t_curr[mask_keep]
        y_curr = y_curr[mask_keep]

    if coeffs is None:
        coeffs = _fit_hants_coeffs(t_curr, y_curr, freqs)

    if coeffs is None:
        return np.full(len(target_t_array), np.nan)

    X_target = make_harmonic_matrix(target_t_array, freqs)
    return X_target @ coeffs

def reconstruct_hants(
    image: Union[str, Path],
    target_time: str,
    output_path: Optional[Union[str, Path]] = None,
    nof: int = 3,
    sf: str = 'low',
    fet: float = 0.05,
    dod: int = 5,
    n_jobs: int = -1,
    cache_dir: Union[str, Path] = "./cache",
    force_refresh: bool = False
) -> np.ndarray:
    """
    Reconstruct Landsat image using HANTS (Roerink et al. 2000).

    Parameters:
    - nof: Number of frequencies (including mean). Default 3 (Mean, Annual, Semi-Annual).
    - sf: Suppression flag ('low' or 'high'). 'low' rejects low outliers (e.g. cloud shadows or clouds in NDVI).
          For Surface Reflectance clouds are usually High.
    - fet: Fit Error Tolerance.
    - dod: Degree of Overdeterminedness.
    """
    # 1. Load Data
    loader = RSCube(image, cache_dir=cache_dir, force_refresh=force_refresh)
    data = loader.load()
    cube = np.ma.filled(data["cube"], np.nan)
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
                nof=nof, sf=sf, fet=fet, dod=dod
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
