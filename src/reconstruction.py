import os
import time
from typing import Tuple, Sequence, cast, Optional
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

def revive(cube: np.ndarray, timestamps: np.ndarray, target_time: str, args: Optional[Args] = None, **kwargs) -> np.ndarray:
    """核心重建函数，支持直接传入参数或 Args 对象"""
    # 合并参数
    if args is None:
        from .config import Args
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
                refine_peaks=args.refine_peaks, include_trend=args.include_trend,
                ridge_lam=args.ridge, freq_weight=args.freq_weight, huber_iters=args.huber_iters, huber_delta=args.huber_delta,
                min_obs=args.min_obs
            )
            row[j] = pred
        return i, row

    # 确定并行任务数
    n_jobs = args.n_jobs
    if n_jobs <= 0:
        n_jobs = max(1, int(os.cpu_count() or 1)) if cpu_count is None else max(1, int(cpu_count()))
    n_jobs = min(n_jobs, H)

    print(f"[System] Starting reconstruction with {n_jobs} jobs...")

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
