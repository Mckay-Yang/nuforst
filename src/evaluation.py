import numpy as np
import pandas as pd
import time
import os
from contextlib import contextmanager
from typing import Iterator, Tuple, Dict, List, Optional, Union
from pathlib import Path
import warnings
from collections import defaultdict

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ModuleNotFoundError:
    Parallel = None
    delayed = None
    cpu_count = None
    JOBLIB_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ModuleNotFoundError:
    tqdm = None
    TQDM_AVAILABLE = False

from .nufrost import timestamps_to_seconds, parse_timestamp_str, predict_single_pixel
from .data_loader import RSCube, TimeSeriesRasterSource
from .hants import hants_pixel
from .zhu2015 import fit_predict_pixel
from config import Args


def _image_name_from_path(image_path: Union[str, Path, List[Union[str, Path]]]) -> str:
    img_name = Path(image_path[0] if isinstance(image_path, list) else image_path).name
    import re
    match = re.search(r"([A-Z0-9]+_lon[0-9.]+_lat[0-9.]+)", img_name)
    return match.group(1) if match else img_name


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs <= 0:
        return max(1, int(os.cpu_count() or 1)) if cpu_count is None else max(1, int(cpu_count()))
    return n_jobs


def load_evaluation_cube(
    image_path: Union[str, Path, List[Union[str, Path]]],
    args: Args,
) -> Dict[str, object]:
    loader = RSCube(image_path, cache_dir=args.cache_dir, force_refresh=getattr(args, "force_refresh", False))
    data = loader.load()
    cube = np.ma.filled(data["cube"], np.nan)
    timestamps = data["timestamps"]

    t_sec = timestamps_to_seconds(timestamps, unit="seconds")
    t0_sec = np.nanmin(t_sec)
    t_days = (t_sec - t0_sec) / 86400.0

    return {
        "cube": cube,
        "timestamps": timestamps,
        "t_sec": t_sec,
        "t_days": t_days,
        "meta": {k: v for k, v in data.items() if k != "cube"},
    }


@contextmanager
def open_evaluation_source(
    image_path: Union[str, Path, List[Union[str, Path]]],
    args: Args,
) -> Iterator[Dict[str, object]]:
    with TimeSeriesRasterSource(image_path, cache_dir=args.cache_dir) as source:
        meta = source.metadata()
        timestamps = meta["timestamps"]
        t_sec = timestamps_to_seconds(timestamps, unit="seconds")
        t0_sec = np.nanmin(t_sec)
        t_days = (t_sec - t0_sec) / 86400.0
        yield {
            "source": source,
            "timestamps": timestamps,
            "t_sec": t_sec,
            "t_days": t_days,
            "meta": meta,
        }


def _scan_valid_counts_from_source(
    source: TimeSeriesRasterSource,
    t_days: np.ndarray,
    block_shape: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    meta = source.metadata()
    valid_counts = np.zeros((int(meta["height"]), int(meta["width"])), dtype=np.int32)
    t_valid_mask = np.isfinite(t_days)
    for row_slice, col_slice in source.iter_windows(block_shape=block_shape):
        arr = source.read_window(row_slice, col_slice)
        valid_mask = np.isfinite(arr) & t_valid_mask[:, np.newaxis, np.newaxis]
        valid_counts[row_slice, col_slice] = np.sum(valid_mask, axis=0)
    return valid_counts


def scan_pixel_stats_from_source(
    source: TimeSeriesRasterSource,
    t_days: np.ndarray,
    block_shape: Tuple[int, int] = (256, 256),
    log_interval: int = 0,
) -> Dict[str, np.ndarray]:
    meta = source.metadata()
    h, w = int(meta["height"]), int(meta["width"])
    t_valid_mask = np.isfinite(t_days)
    n_valid_t = int(np.sum(t_valid_mask))

    valid_counts = np.zeros((h, w), dtype=np.int32)
    missing_ratios = np.ones((h, w), dtype=np.float32)
    native_gap_days = np.full((h, w), np.inf, dtype=np.float32)

    windows = list(source.iter_windows(block_shape=block_shape))
    n_windows = len(windows)
    if log_interval <= 0:
        log_interval = max(1, n_windows // 10)

    for wi, (row_slice, col_slice) in enumerate(windows):
        if wi > 0 and wi % log_interval == 0:
            print(f"  scan_pixel_stats: {wi}/{n_windows} windows", flush=True)
        arr = source.read_window(row_slice, col_slice)
        valid_mask = np.isfinite(arr) & t_valid_mask[:, np.newaxis, np.newaxis]
        vc = np.sum(valid_mask, axis=0).astype(np.int32)
        valid_counts[row_slice, col_slice] = vc

        rh = row_slice.stop - row_slice.start
        cw = col_slice.stop - col_slice.start
        has_obs = vc > 0

        mr = np.ones((rh, cw), dtype=np.float32)
        mr[has_obs] = 1.0 - vc[has_obs].astype(np.float32) / n_valid_t
        missing_ratios[row_slice, col_slice] = mr

        ngd = np.full((rh, cw), np.inf, dtype=np.float32)
        need_gap = np.argwhere(vc >= 2)
        for dr, dc in need_gap:
            ts_valid = np.sort(t_days[valid_mask[:, dr, dc]])
            if len(ts_valid) > 1:
                ngd[dr, dc] = float(np.max(np.diff(ts_valid)))
        native_gap_days[row_slice, col_slice] = ngd

    return {
        "valid_counts": valid_counts,
        "missing_ratios": missing_ratios,
        "native_gap_days": native_gap_days,
    }


def sample_random_points_from_source(
    source: TimeSeriesRasterSource,
    t_days: np.ndarray,
    min_obs: int,
    num_points: int,
    seed: int = 42,
    block_shape: Tuple[int, int] = (256, 256),
    precomputed_stats: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    if precomputed_stats is not None:
        stats = precomputed_stats
    else:
        stats = scan_pixel_stats_from_source(source, t_days, block_shape=block_shape)
    valid_counts = stats["valid_counts"]
    t_valid_mask = np.isfinite(t_days)

    valid_pixels = np.argwhere(valid_counts >= max(min_obs + 1, 3))
    if len(valid_pixels) == 0:
        return np.empty((0, 3), dtype=int)

    rng = np.random.RandomState(seed)
    weights = valid_counts[valid_pixels[:, 0], valid_pixels[:, 1]].astype(np.float64)
    weights = weights / weights.sum()

    valid_time_indices = np.flatnonzero(t_valid_mask)

    selected: List[List[int]] = []
    seen = set()
    max_attempts = max(num_points * 20, len(valid_pixels))
    attempts = 0
    while len(selected) < min(num_points, len(valid_pixels)) and attempts < max_attempts:
        pix_idx = int(rng.choice(len(valid_pixels), p=weights))
        row, col = [int(v) for v in valid_pixels[pix_idx]]
        t_idx = int(rng.choice(valid_time_indices))
        key = (t_idx, row, col)
        if key not in seen:
            seen.add(key)
            selected.append([t_idx, row, col])
        attempts += 1
    return np.array(selected, dtype=int) if selected else np.empty((0, 3), dtype=int)


def sample_gap_pixels_from_source(
    source: TimeSeriesRasterSource,
    t_days: np.ndarray,
    min_obs: int,
    num_samples: int,
    seed: int = 42,
    block_shape: Tuple[int, int] = (256, 256),
    precomputed_stats: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    if precomputed_stats is not None:
        valid_counts = precomputed_stats["valid_counts"]
    else:
        valid_counts = _scan_valid_counts_from_source(source, t_days, block_shape=block_shape)
    valid_pixels = np.argwhere(valid_counts >= max(min_obs + 3, 15))
    if len(valid_pixels) == 0:
        return np.empty((0, 2), dtype=int)
    if len(valid_pixels) > num_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(valid_pixels), num_samples, replace=False)
        return valid_pixels[indices]
    return valid_pixels


def scan_gap_candidates_from_source(
    source: TimeSeriesRasterSource,
    t_days: np.ndarray,
    min_obs: int,
    max_candidates: int = 50000,
    seed: int = 42,
    block_shape: Tuple[int, int] = (256, 256),
    precomputed_stats: Optional[Dict[str, np.ndarray]] = None,
) -> List[Tuple[int, int, float, float]]:
    if precomputed_stats is not None:
        stats = precomputed_stats
    else:
        stats = scan_pixel_stats_from_source(source, t_days, block_shape=block_shape)
    valid_counts = stats["valid_counts"]
    missing_ratios = stats["missing_ratios"]
    native_gap_days = stats["native_gap_days"]

    threshold = max(min_obs + 3, 15)
    valid_pixels = np.argwhere(valid_counts >= threshold)
    if len(valid_pixels) == 0:
        return []

    if len(valid_pixels) > max_candidates:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(valid_pixels), max_candidates, replace=False)
        valid_pixels = valid_pixels[indices]

    candidates = []
    for r, c in valid_pixels:
        candidates.append((int(r), int(c), float(missing_ratios[r, c]), float(native_gap_days[r, c])))
    return candidates


def sample_random_points(
    cube: np.ndarray,
    t_days: np.ndarray,
    min_obs: int,
    num_points: int,
    seed: int = 42,
) -> np.ndarray:
    t_valid_mask = np.isfinite(t_days)
    valid_mask = np.isfinite(cube) & t_valid_mask[:, np.newaxis, np.newaxis]
    valid_counts = np.sum(valid_mask, axis=0)
    valid_pixels_mask = valid_counts >= max(min_obs + 1, 3)
    candidate_mask = valid_mask & np.broadcast_to(valid_pixels_mask[np.newaxis, :, :], cube.shape)
    candidate_indices = np.argwhere(candidate_mask)

    if len(candidate_indices) == 0:
        return np.empty((0, 3), dtype=int)

    if len(candidate_indices) > num_points:
        rng = np.random.RandomState(seed)
        selected_idx = rng.choice(len(candidate_indices), num_points, replace=False)
        return candidate_indices[selected_idx]

    return candidate_indices


def sample_gap_pixels(
    cube: np.ndarray,
    t_days: np.ndarray,
    min_obs: int,
    num_samples: int,
    seed: int = 42,
) -> np.ndarray:
    t_valid_mask = np.isfinite(t_days)
    valid_mask = np.isfinite(cube) & t_valid_mask[:, np.newaxis, np.newaxis]
    valid_counts = np.sum(valid_mask, axis=0)
    valid_pixels = np.argwhere(valid_counts >= max(min_obs + 3, 15))

    if len(valid_pixels) == 0:
        return np.empty((0, 2), dtype=int)

    if len(valid_pixels) > num_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(valid_pixels), num_samples, replace=False)
        return valid_pixels[indices]

    return valid_pixels


def evaluate_algorithms_from_source(
    source: TimeSeriesRasterSource,
    t_sec: np.ndarray,
    t_days: np.ndarray,
    args: Args,
    num_points: Optional[int] = None,
    sampled_points: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    if sampled_points is None:
        if num_points is None:
            raise ValueError("Either num_points or sampled_points must be provided.")
        sampled_points = sample_random_points_from_source(source, t_days, args.min_obs, num_points)

    if len(sampled_points) == 0:
        print("Not enough valid points to evaluate.")
        return pd.DataFrame()

    print(f"Selected {len(sampled_points)} valid points for evaluation.")
    rc_to_t_idx = defaultdict(list)
    for t_idx, r, c in sampled_points:
        rc_to_t_idx[(int(r), int(c))].append(int(t_idx))

    start_time = time.time()
    n_jobs_resolved = _resolve_n_jobs(n_jobs)
    print(f"--> Running predictions with {n_jobs_resolved} parallel workers...")

    ordered_points = []
    tasks = []
    for (r, c), t_idxs in rc_to_t_idx.items():
        y_ts = source.read_pixel_series(r, c).copy()
        for t in t_idxs:
            y_ts[t] = np.nan
        for target_t_idx in t_idxs:
            if JOBLIB_AVAILABLE:
                tasks.append(delayed(_process_random_point)(target_t_idx, r, c, y_ts, t_sec, t_days, args))
            else:
                tasks.append((_process_random_point, (target_t_idx, r, c, y_ts, t_sec, t_days, args), {}))
            ordered_points.append((target_t_idx, r, c))

    point_results = []
    if JOBLIB_AVAILABLE and n_jobs_resolved > 1:
        if TQDM_AVAILABLE:
            try:
                gen = Parallel(n_jobs=n_jobs_resolved, return_as="generator")(tasks)
                point_results = list(tqdm(gen, total=len(tasks), desc="Evaluating Points"))
            except TypeError:
                point_results = Parallel(n_jobs=n_jobs_resolved)(tasks)
        else:
            point_results = Parallel(n_jobs=n_jobs_resolved)(tasks)
    else:
        iterator = tasks
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(tasks), desc="Evaluating Points")
        for task in iterator:
            func, func_args, func_kwargs = task
            point_results.append(func(*func_args, **func_kwargs))

    print(f"--> Prediction finished in {time.time() - start_time:.2f}s.")

    true_all = []
    pred_all_nufrost = []
    pred_all_zhu = []
    pred_all_hants = []
    for (t_idx, r, c), res in zip(ordered_points, point_results):
        true_all.append(source.read_pixel_series(r, c)[t_idx])
        pred_all_nufrost.append(res["nufrost"])
        pred_all_zhu.append(res["zhu"])
        pred_all_hants.append(res["hants"])

    metrics_nufrost = compute_metrics(np.array(true_all), np.array(pred_all_nufrost))
    metrics_zhu = compute_metrics(np.array(true_all), np.array(pred_all_zhu))
    metrics_hants = compute_metrics(np.array(true_all), np.array(pred_all_hants))
    return pd.DataFrame([
        {"Algorithm": "NuFrost", "RMSE": metrics_nufrost["RMSE"], "MAE": metrics_nufrost["MAE"], "R": metrics_nufrost["R"], "OutlierRatio": metrics_nufrost.get("OutlierRatio", np.nan)},
        {"Algorithm": "Zhu2015", "RMSE": metrics_zhu["RMSE"], "MAE": metrics_zhu["MAE"], "R": metrics_zhu["R"], "OutlierRatio": metrics_zhu.get("OutlierRatio", np.nan)},
        {"Algorithm": "HANTS", "RMSE": metrics_hants["RMSE"], "MAE": metrics_hants["MAE"], "R": metrics_hants["R"], "OutlierRatio": metrics_hants.get("OutlierRatio", np.nan)},
    ])


def evaluate_timeseries_from_source(
    source: TimeSeriesRasterSource,
    t_sec: np.ndarray,
    t_days: np.ndarray,
    args: Args,
    simulate_gap_days: int,
    num_samples: Optional[int] = None,
    sampled_pixels: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    if sampled_pixels is None:
        if num_samples is None:
            raise ValueError("Either num_samples or sampled_pixels must be provided.")
        sampled_pixels = sample_gap_pixels_from_source(source, t_days, args.min_obs, num_samples)

    if len(sampled_pixels) == 0:
        print("Not enough valid pixels to evaluate.")
        return pd.DataFrame()

    print(f"Selected {len(sampled_pixels)} valid pixels for time-series evaluation.")
    start_time = time.time()
    pixel_results = []
    for r, c in sampled_pixels:
        y_ts = source.read_pixel_series(int(r), int(c)).copy()
        pixel_results.append(_process_pixel_ts(int(r), int(c), y_ts, t_days, t_sec, len(t_days), simulate_gap_days, args))
    true_all_nufrost, pred_all_nufrost = [], []
    true_all_zhu, pred_all_zhu = [], []
    true_all_hants, pred_all_hants = [], []
    pixel_stats = []
    for res in pixel_results:
        if res is None:
            continue
        pixel_stats.append(res["stat"])
        true_all_nufrost.extend(res["true"])
        pred_all_nufrost.extend(res["pred_nufrost"])
        true_all_zhu.extend(res["true"])
        pred_all_zhu.extend(res["pred_zhu"])
        true_all_hants.extend(res["true"])
        pred_all_hants.extend(res["pred_hants"])
    print(f"--> Time-series evaluation finished in {time.time() - start_time:.2f}s.")
    if not pixel_stats:
        return pd.DataFrame()
    metrics_n = compute_metrics(np.array(true_all_nufrost), np.array(pred_all_nufrost))
    metrics_z = compute_metrics(np.array(true_all_zhu), np.array(pred_all_zhu))
    metrics_h = compute_metrics(np.array(true_all_hants), np.array(pred_all_hants))
    return pd.DataFrame([
        {"Algorithm": "NuFrost", "RMSE": metrics_n["RMSE"], "MAE": metrics_n["MAE"], "R": metrics_n["R"], "OutlierRatio": metrics_n["OutlierRatio"]},
        {"Algorithm": "Zhu2015", "RMSE": metrics_z["RMSE"], "MAE": metrics_z["MAE"], "R": metrics_z["R"], "OutlierRatio": metrics_z["OutlierRatio"]},
        {"Algorithm": "HANTS", "RMSE": metrics_h["RMSE"], "MAE": metrics_h["MAE"], "R": metrics_h["R"], "OutlierRatio": metrics_h["OutlierRatio"]},
    ])


def evaluate_algorithms_on_cube(
    cube: np.ndarray,
    t_sec: np.ndarray,
    t_days: np.ndarray,
    args: Args,
    num_points: Optional[int] = None,
    sampled_points: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    if sampled_points is None:
        if num_points is None:
            raise ValueError("Either num_points or sampled_points must be provided.")
        sampled_points = sample_random_points(cube, t_days, args.min_obs, num_points)

    if len(sampled_points) == 0:
        print("Not enough valid points to evaluate.")
        return pd.DataFrame()

    print(f"Selected {len(sampled_points)} valid points for evaluation.")

    rc_to_t_idx = defaultdict(list)
    for t_idx, r, c in sampled_points:
        rc_to_t_idx[(r, c)].append(t_idx)

    start_time = time.time()
    n_jobs = _resolve_n_jobs(n_jobs)
    print(f"--> Running predictions with {n_jobs} parallel workers...")

    tasks = []
    ordered_points = []
    for (r, c), t_idxs in rc_to_t_idx.items():
        y_ts = cube[:, r, c].copy()
        for t in t_idxs:
            y_ts[t] = np.nan

        for target_t_idx in t_idxs:
            if JOBLIB_AVAILABLE:
                tasks.append(delayed(_process_random_point)(target_t_idx, r, c, y_ts, t_sec, t_days, args))
            else:
                tasks.append((_process_random_point, (target_t_idx, r, c, y_ts, t_sec, t_days, args), {}))
            ordered_points.append((target_t_idx, r, c))

    point_results = []
    if JOBLIB_AVAILABLE and n_jobs > 1:
        if TQDM_AVAILABLE:
            try:
                gen = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
                point_results = list(tqdm(gen, total=len(tasks), desc="Evaluating Points"))
            except TypeError:
                point_results = Parallel(n_jobs=n_jobs)(tasks)
        else:
            point_results = Parallel(n_jobs=n_jobs)(tasks)
    else:
        iterator = tasks
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(tasks), desc="Evaluating Points")
        for task in iterator:
            func, func_args, func_kwargs = task
            point_results.append(func(*func_args, **func_kwargs))

    print(f"--> Prediction finished in {time.time() - start_time:.2f}s.")

    true_all = []
    pred_all_nufrost = []
    pred_all_zhu = []
    pred_all_hants = []

    for (t_idx, r, c), res in zip(ordered_points, point_results):
        true_all.append(cube[t_idx, r, c])
        pred_all_nufrost.append(res["nufrost"])
        pred_all_zhu.append(res["zhu"])
        pred_all_hants.append(res["hants"])

    metrics_nufrost = compute_metrics(np.array(true_all), np.array(pred_all_nufrost))
    metrics_zhu = compute_metrics(np.array(true_all), np.array(pred_all_zhu))
    metrics_hants = compute_metrics(np.array(true_all), np.array(pred_all_hants))

    df_results = pd.DataFrame([
        {"Algorithm": "NuFrost", "RMSE": metrics_nufrost["RMSE"], "MAE": metrics_nufrost["MAE"], "R": metrics_nufrost["R"], "OutlierRatio": metrics_nufrost.get("OutlierRatio", np.nan)},
        {"Algorithm": "Zhu2015", "RMSE": metrics_zhu["RMSE"], "MAE": metrics_zhu["MAE"], "R": metrics_zhu["R"], "OutlierRatio": metrics_zhu.get("OutlierRatio", np.nan)},
        {"Algorithm": "HANTS", "RMSE": metrics_hants["RMSE"], "MAE": metrics_hants["MAE"], "R": metrics_hants["R"], "OutlierRatio": metrics_hants.get("OutlierRatio", np.nan)},
    ])
    print("\n========== Random 3D Points Evaluation Summary ==========")
    print(df_results.to_string(index=False))
    return df_results


def evaluate_timeseries_on_cube(
    cube: np.ndarray,
    t_sec: np.ndarray,
    t_days: np.ndarray,
    args: Args,
    simulate_gap_days: int,
    num_samples: Optional[int] = None,
    sampled_pixels: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    if sampled_pixels is None:
        if num_samples is None:
            raise ValueError("Either num_samples or sampled_pixels must be provided.")
        sampled_pixels = sample_gap_pixels(cube, t_days, args.min_obs, num_samples)

    if len(sampled_pixels) == 0:
        print("Not enough valid pixels to evaluate.")
        return pd.DataFrame()

    print(f"Selected {len(sampled_pixels)} valid pixels for time-series evaluation.")

    start_time = time.time()
    n_jobs = _resolve_n_jobs(n_jobs)
    print(f"Running predictions with {n_jobs} parallel workers...")

    pixel_results = []
    if JOBLIB_AVAILABLE and n_jobs > 1:
        tasks = [delayed(_process_pixel_ts)(r, c, cube[:, r, c].copy(), t_days, t_sec, cube.shape[0], simulate_gap_days, args) for r, c in sampled_pixels]
        if TQDM_AVAILABLE:
            try:
                gen = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
                pixel_results = list(tqdm(gen, total=len(sampled_pixels), desc="Evaluating Pixels"))
            except TypeError:
                pixel_results = Parallel(n_jobs=n_jobs)(tasks)
        else:
            pixel_results = Parallel(n_jobs=n_jobs)(tasks)
    else:
        iterator = sampled_pixels
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(sampled_pixels), desc="Evaluating Pixels")
        for r, c in iterator:
            pixel_results.append(_process_pixel_ts(r, c, cube[:, r, c].copy(), t_days, t_sec, cube.shape[0], simulate_gap_days, args))

    true_all_nufrost, pred_all_nufrost = [], []
    true_all_zhu, pred_all_zhu = [], []
    true_all_hants, pred_all_hants = [], []
    pixel_stats = []

    for res in pixel_results:
        if res is None:
            continue
        pixel_stats.append(res["stat"])
        true_all_nufrost.extend(res["true"])
        pred_all_nufrost.extend(res["pred_nufrost"])
        true_all_zhu.extend(res["true"])
        pred_all_zhu.extend(res["pred_zhu"])
        true_all_hants.extend(res["true"])
        pred_all_hants.extend(res["pred_hants"])

    print(f"--> Time-series evaluation finished in {time.time() - start_time:.2f}s.")

    if not pixel_stats:
        print("No valid pixels left after adding gap.")
        return pd.DataFrame()

    df_stats = pd.DataFrame(pixel_stats)
    print("\n[Dataset Missingness Profile (Sampled Pixels)]")
    print(f"Average Original Missing Ratio: {df_stats['Original_Missing_Ratio'].mean()*100:.2f}%")
    print(f"Average Max Continuous Gap: {df_stats['Original_Max_Gap_Days'].mean():.2f} days")

    metrics_n = compute_metrics(np.array(true_all_nufrost), np.array(pred_all_nufrost))
    metrics_z = compute_metrics(np.array(true_all_zhu), np.array(pred_all_zhu))
    metrics_h = compute_metrics(np.array(true_all_hants), np.array(pred_all_hants))

    df_results = pd.DataFrame([
        {"Algorithm": "NuFrost", "RMSE": metrics_n["RMSE"], "MAE": metrics_n["MAE"], "R": metrics_n["R"], "OutlierRatio": metrics_n["OutlierRatio"]},
        {"Algorithm": "Zhu2015", "RMSE": metrics_z["RMSE"], "MAE": metrics_z["MAE"], "R": metrics_z["R"], "OutlierRatio": metrics_z["OutlierRatio"]},
        {"Algorithm": "HANTS", "RMSE": metrics_h["RMSE"], "MAE": metrics_h["MAE"], "R": metrics_h["R"], "OutlierRatio": metrics_h["OutlierRatio"]},
    ])
    print("\n========== Comprehensive Evaluation Summary ==========")
    print(f"(Tested by simulating {simulate_gap_days}-day continuous gaps)")
    print(df_results.to_string(index=False))
    return df_results

def generate_block_mask(shape: Tuple[int, int], ratio: float = 0.3) -> np.ndarray:
    """生成一个块状掩膜（模拟大片云遮挡）。"""
    H, W = shape
    mask = np.zeros(shape, dtype=bool)
    bh, bw = int(H * ratio), int(W * ratio)
    start_h, start_w = (H - bh) // 2, (W - bw) // 2
    mask[start_h:start_h+bh, start_w:start_w+bw] = True
    return mask

def generate_random_mask(shape: Tuple[int, int], ratio: float = 0.3) -> np.ndarray:
    """生成一个随机掩膜（模拟零散缺失）。"""
    return np.random.rand(*shape) < ratio

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算误差指标"""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"RMSE": np.nan, "MAE": np.nan, "R": np.nan, "OutlierRatio": np.nan}
    
    yt = y_true[valid]
    yp = y_pred[valid]
    
    mse = np.mean((yt - yp) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(yt - yp))
    
    if np.std(yt) == 0 or np.std(yp) == 0:
        r = 0.0
    else:
        r = np.corrcoef(yt, yp)[0, 1]
        
    # 计算异常值比例：预测值超出真实值合理范围 (例如 3倍标准差) 的比例
    mean_yt = np.mean(yt)
    std_yt = np.std(yt)
    if std_yt > 0:
        outliers = np.abs(yp - mean_yt) > max(3 * std_yt, 0.2 * np.abs(mean_yt)) # 防止 std_yt 太小
    else:
        outliers = np.abs(yp - yt) > 0.2 * np.abs(yt)
    outlier_ratio = float(np.mean(outliers))
        
    metrics = {"RMSE": float(rmse), "MAE": float(mae), "R": float(r), "OutlierRatio": outlier_ratio}
    
    if ssim is not None and y_true.ndim == 2:
        yt_img = np.nan_to_num(y_true)
        yp_img = np.nan_to_num(y_pred)
        data_range = max(yt_img.max() - yt_img.min(), 1.0)
        metrics["SSIM"] = float(ssim(yt_img, yp_img, data_range=data_range))
        
    return metrics

def _process_pixel_ts(r: int, c: int, y_ts: np.ndarray, t_days: np.ndarray, t_sec: np.ndarray, T: int, simulate_gap_days: int, args: Args) -> Optional[Dict]:
    """处理单一像元的时间序列评估（用于并行计算）"""
    valid_mask_orig = np.isfinite(y_ts)
    t_valid_days = t_days[valid_mask_orig]
    
    if len(t_valid_days) == 0:
        return None
        
    original_missing_count = T - len(t_valid_days)
    original_missing_ratio = original_missing_count / T
    
    if len(t_valid_days) > 1:
        gaps = np.diff(t_valid_days)
        max_gap_orig = float(np.max(gaps))
    else:
        max_gap_orig = float(np.max(t_days) - np.min(t_days))
        
    pixel_stat = {
        "Original_Valid_Count": len(t_valid_days),
        "Original_Missing_Ratio": original_missing_ratio,
        "Original_Max_Gap_Days": max_gap_orig
    }
        
    min_t, max_t = t_valid_days[0], t_valid_days[-1]
    
    # 使用基于坐标的随机种子以确保结果可重复（特别是在并行下）
    rng = np.random.RandomState(r * 10000 + c)
    gap_start = rng.uniform(min_t, max(min_t + 1, max_t - simulate_gap_days))
    gap_end = gap_start + simulate_gap_days
    
    eval_mask = valid_mask_orig & (t_days >= gap_start) & (t_days <= gap_end)
    
    if not np.any(eval_mask):
        return None 
        
    y_corrupted = y_ts.copy()
    y_corrupted[eval_mask] = np.nan 
    
    y_true_eval = y_ts[eval_mask]
    t_eval_days = t_days[eval_mask]
    t_eval_secs = t_sec[eval_mask]
    
    preds_nufrost = []
    preds_zhu = []
    preds_hants = []
    
    for t_target_day, t_target_sec in zip(t_eval_days, t_eval_secs):
        # NuFrost
        pred_n, _ = predict_single_pixel(
            t_sec, y_corrupted, t_target_sec,
            nufft_modes=args.modes, eps=args.eps,
            num_peaks=args.num_peaks, power_cum=args.power_cum, ignore_dc_hz=args.ignore_dc_hz,
            refine_peaks=args.refine_peaks, include_trend=args.include_trend,
            ridge_lam=args.ridge, freq_weight=args.freq_weight, huber_iters=args.huber_iters, huber_delta=args.huber_delta,
            min_obs=args.min_obs
        )
        preds_nufrost.append(pred_n)
        
        # Zhu2015
        pred_z, _ = fit_predict_pixel(t_days, y_corrupted, t_target_day, lasso_alpha=0.0001)
        preds_zhu.append(pred_z)
        
        # HANTS
        pred_h = hants_pixel(t_days, y_corrupted, t_target_day, nof=3, sf='low', fet=0.05, dod=5)
        preds_hants.append(pred_h)
        
    return {
        "stat": pixel_stat,
        "true": y_true_eval,
        "pred_nufrost": preds_nufrost,
        "pred_zhu": preds_zhu,
        "pred_hants": preds_hants
    }

def evaluate_timeseries_comprehensive(
    image_path: Union[str, Path, List[Union[str, Path]]],
    args: Args,
    num_samples: int = 1000,
    simulate_gap_days: int = 60,
    n_jobs: int = -1,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """
    综合时序评估方法 (支持并行)：
    1. 统计影像本身的时间序列缺失率、最大连续缺失天数。
    2. 对于每个选定的像元，在时间序列上人为引入一段时间的连续缺失（模拟长周期云覆盖）。
    3. 分别用各种算法预测这些人为挖去的点，并计算综合指标。
    """
    print(f"\n========== Starting Comprehensive Time-Series Evaluation ==========")
    img_name = _image_name_from_path(image_path)
    print(f"Image: {img_name}")
    print(f"Simulating continuous gap: {simulate_gap_days} days")
    prepared = load_evaluation_cube(image_path, args)
    sampled_pixels = sample_gap_pixels(prepared["cube"], prepared["t_days"], args.min_obs, num_samples, seed=sample_seed)
    return evaluate_timeseries_on_cube(
        prepared["cube"],
        prepared["t_sec"],
        prepared["t_days"],
        args,
        simulate_gap_days=simulate_gap_days,
        sampled_pixels=sampled_pixels,
        n_jobs=n_jobs,
    )


def _process_random_point(t_idx: int, r: int, c: int, y_ts: np.ndarray, t_sec: np.ndarray, t_days: np.ndarray, args: Args) -> Dict:
    """处理单个随机三维点的预测"""
    target_t_sec = t_sec[t_idx]
    target_t_day = t_days[t_idx]

    pred_n, _ = predict_single_pixel(
        t_sec, y_ts, target_t_sec,
        nufft_modes=args.modes, eps=args.eps,
        num_peaks=args.num_peaks, power_cum=args.power_cum, ignore_dc_hz=args.ignore_dc_hz,
        refine_peaks=args.refine_peaks, include_trend=args.include_trend,
        ridge_lam=args.ridge, freq_weight=args.freq_weight, huber_iters=args.huber_iters, huber_delta=args.huber_delta,
        min_obs=args.min_obs
    )
    pred_z, _ = fit_predict_pixel(t_days, y_ts, target_t_day, lasso_alpha=0.0001)
    pred_h = hants_pixel(t_days, y_ts, target_t_day, nof=3, sf='low', fet=0.05, dod=5)
    
    return {"t_idx": t_idx, "r": r, "c": c, "nufrost": pred_n, "zhu": pred_z, "hants": pred_h}


def evaluate_algorithms(
    image_path: Union[str, Path, List[Union[str, Path]]],
    args: Args,
    num_points: int = 1000,
    n_jobs: int = -1,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """
    随机三维点评估方法 (支持并行)：
    在整个时空数据集中随机选取一定数量的有效点进行敲除，然后利用剩余数据对这些点进行预测并评估。
    """
    print(f"\n========== Starting 3D Random Points Evaluation ==========")
    img_name = _image_name_from_path(image_path)
    print(f"Image: {img_name}")
    print(f"Masking {num_points} random points across all space and time.")
    prepared = load_evaluation_cube(image_path, args)
    sampled_points = sample_random_points(prepared["cube"], prepared["t_days"], args.min_obs, num_points, seed=sample_seed)
    return evaluate_algorithms_on_cube(
        prepared["cube"],
        prepared["t_sec"],
        prepared["t_days"],
        args,
        sampled_points=sampled_points,
        n_jobs=n_jobs,
    )
