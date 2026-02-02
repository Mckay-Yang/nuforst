import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from .data_loader import RSCube
from .config import Args, build_args
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
    nof: int = 3, # Number of frequencies. 3 means: 0, 1/T, 2/T
    sf: str = 'low', # 'low' means reject low values (e.g. clouds in NDVI)
    idrt: float = None, # Invalid data rejection threshold
    fet: float = 0.05, # Fit error tolerance
    dod: int = 5, # Degree of overdeterminedness
    period: float = 365.25
) -> float:
    """
    Apply HANTS algorithm to a single pixel.
    """
    # 1. Initial valid data check
    valid_mask = np.isfinite(y)
    if idrt is not None:
        if sf == 'low':
            valid_mask &= (y > idrt)
        elif sf == 'high':
            valid_mask &= (y < idrt)

    if np.sum(valid_mask) == 0:
        return np.nan

    t_curr = t[valid_mask]
    y_curr = y[valid_mask]

    # 2. Setup frequencies
    # NOF includes 0 freq. If NOF=3, we use freq 0, 1/period, 2/period
    freqs = [i / period for i in range(1, nof)] # 0 is handled by intercept column

    # Number of parameters: 1 (mean) + 2 * (NOF-1)
    # Paper: 2 * NOF - 1. (NOF includes zero).
    # e.g. NOF=3 -> Mean, Amp1, Ph1, Amp2, Ph2 -> 5 params.
    # 1 + 2*(3-1) = 5. Correct.
    num_params = 1 + 2 * (nof - 1)

    # Iteration
    max_iter = 20 # Safety break
    for _ in range(max_iter):
        n_obs = len(y_curr)
        if n_obs < num_params + dod:
            # Not enough points
            break

        # Build Matrix
        X = make_harmonic_matrix(t_curr, freqs)

        # Fit Least Squares
        # lstsq returns: x, residuals, rank, s
        coeffs, _, _, _ = np.linalg.lstsq(X, y_curr, rcond=None)

        # Calculate fit values for CURRENT points to check outliers
        y_pred_curr = X @ coeffs
        residuals = y_curr - y_pred_curr

        # Check outliers
        # SF = 'low' means reject low values. Low values have y_curr < y_pred => residual < 0
        # Paper: "large positive or negative deviation ... removed"
        # Paper (Sec 2): "Hi/Lo suppression flag (SF)... indicates whether high or low values (outliers) should be rejected"
        # Example: "SF = low; ... cloudy observations lead to low NDVI values."
        # This implies we reject points where y_obs is significantly LOWER than y_fit.
        # i.e. residual (y_obs - y_fit) is large NEGATIVE.

        candidates_to_reject = np.zeros(n_obs, dtype=bool)

        if sf == 'low':
            # Reject if y_obs is too low -> residual is negative and magnitude > FET
            # Paper: "absolute difference in the Hi/Lo direction ... determined"
            # "Iteration stops when the difference of all remaining points becomes smaller than the FET"
            diffs = residuals
            # We care about negative residuals
            bad_indices = (diffs < -fet)
        elif sf == 'high':
            # Reject if y_obs is too high -> residual is positive and > FET
            diffs = residuals
            bad_indices = (diffs > fet)
        else:
            # Reject both? Paper implies one direction usually.
            # "HANTS cannot reject outliers in the opposite direction of the SF"
            bad_indices = (np.abs(residuals) > fet)

        if not np.any(bad_indices):
            # Convergence: No points exceed FET in the specified direction
            break

        # Identify the WORST outlier to remove? Or all?
        # Paper says: "Input data points that have a large ... deviation ... are removed".
        # Usually HANTS removes *all* outliers outside tolerance in one step?
        # Or one by one?
        # "After recalculation ... the procedure is repeated".
        # Let's remove all that violate FET.

        mask_keep = ~bad_indices
        t_curr = t_curr[mask_keep]
        y_curr = y_curr[mask_keep]

    # Final Prediction
    # If loop finished or broke, we use the last coeffs.
    # Need to handle case where loop breaks due to low N *before* computing coeffs?
    # Actually we compute coeffs at start of loop.

    # If we exited loop because N < limit, we might want to use the LAST valid fit?
    # But if N < limit initially?
    if 'coeffs' not in locals():
        return np.nan

    X_target = make_harmonic_matrix(np.array([target_t]), freqs)
    y_target = (X_target @ coeffs)[0]

    return y_target

def reconstruct_hants(
    image: str,
    target_time: str,
    output_path: Optional[str] = None,
    nof: int = 3,
    sf: str = 'low',
    fet: float = 0.05,
    dod: int = 5,
    n_jobs: int = -1,
    cache_dir: str = "./cache",
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
    cube = data["cube"]
    timestamps = data["timestamps"]

    # 2. Prepare Time
    try:
        dt_target = pd.to_datetime(target_time, utc=True)
    except:
        dt_target = pd.to_datetime(target_time)

    t0_sec = np.min(timestamps)
    t_days = (timestamps - t0_sec) / 86400.0

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
