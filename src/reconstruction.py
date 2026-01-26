import os
import time
from typing import Tuple, Sequence, cast
import numpy as np

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

from .config import Args
from .algorithms import timestamps_to_seconds, parse_timestamp_str, predict_single_pixel

def revive(cube: np.ndarray, timestamps: np.ndarray, target_time: str, args: Args) -> np.ndarray:
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
                args.modes, args.eps,
                args.num_peaks, args.power_cum, args.ignore_dc_hz,
                args.refine_peaks, args.include_trend,
                args.ridge, args.freq_weight, args.huber_iters, args.huber_delta,
                args.min_obs
            )
            row[j] = pred
        return i, row

    n_jobs = args.n_jobs
    if n_jobs == 0:
        if cpu_count is not None:
            n_jobs = max(1, int(cpu_count()))
        else:
            n_jobs = max(1, int(os.cpu_count() or 1))
    n_jobs = min(n_jobs, H)

    print(f"[System] Starting reconstruction with {n_jobs} jobs...")

    if not JOBLIB_AVAILABLE or n_jobs == 1:
        if (not JOBLIB_AVAILABLE) and n_jobs != 1:
            print("[Info] joblib not installed; falling back to serial execution.")
        progress_every = max(1, int(args.progress_every))
        start = time.perf_counter()
        if args.show_progress and TQDM_AVAILABLE and tqdm is not None:
            for i in tqdm(range(H), total=H, desc="Rows", unit="row"):
                _, row = _predict_row(i)
                out[i, :] = row
        else:
            for i in range(H):
                _, row = _predict_row(i)
                out[i, :] = row
                if args.show_progress and (i + 1) % progress_every == 0:
                    elapsed = time.perf_counter() - start
                    rate = (i + 1) / max(elapsed, 1e-9)
                    remaining = (H - (i + 1)) / max(rate, 1e-9)
                    print(f"Rows {i+1}/{H} | ETA ~ {remaining/60:.1f} min")
        return out

    assert Parallel is not None and delayed is not None
    if args.show_progress and TQDM_AVAILABLE and tqdm is not None:
        try:
            results_iter = Parallel(
                n_jobs=n_jobs,
                prefer="processes",
                max_nbytes="256M",
                mmap_mode="r",
                return_as="generator",
            )(delayed(_predict_row)(i) for i in range(H))
            results_iter = cast(Sequence[Tuple[int, np.ndarray]], results_iter)
            for i, row in tqdm(results_iter, total=H, desc="Rows", unit="row"):
                out[i, :] = row
            return out
        except TypeError:
            print("[Info] joblib return_as not supported; progress will be approximate.")

    results = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
        max_nbytes="256M",
        mmap_mode="r",
    )(delayed(_predict_row)(i) for i in range(H))
    results = cast(Sequence[Tuple[int, np.ndarray]], results)
    for i, row in results:
        out[i, :] = row
    return out
