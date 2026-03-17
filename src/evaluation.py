import numpy as np
import pandas as pd
import time
import os
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import warnings

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
from .data_loader import RSCube
from .hants import hants_pixel
from .zhu2015 import fit_predict_pixel
from config import Args

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
    image_path: str,
    args: Args,
    num_samples: int = 1000,
    simulate_gap_days: int = 60,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    综合时序评估方法 (支持并行)：
    1. 统计影像本身的时间序列缺失率、最大连续缺失天数。
    2. 对于每个选定的像元，在时间序列上人为引入一段时间的连续缺失（模拟长周期云覆盖）。
    3. 分别用各种算法预测这些人为挖去的点，并计算综合指标。
    """
    print(f"\n========== Starting Comprehensive Time-Series Evaluation ==========")
    print(f"Image: {Path(image_path).name}")
    print(f"Simulating continuous gap: {simulate_gap_days} days")
    
    loader = RSCube(image_path, cache_dir="./cache")
    data = loader.load()
    cube = data["cube"]
    timestamps = data["timestamps"]
    
    T, H, W = cube.shape
    t_sec = timestamps_to_seconds(timestamps, unit="seconds")
    t0_sec = np.min(t_sec)
    t_days = (t_sec - t0_sec) / 86400.0
    
    valid_counts = np.sum(np.isfinite(cube), axis=0)
    valid_pixels = np.argwhere(valid_counts >= max(args.min_obs + 3, 15))
    
    if len(valid_pixels) == 0:
        print("Not enough valid pixels to evaluate.")
        return pd.DataFrame()
        
    if len(valid_pixels) > num_samples:
        np.random.seed(42) # 保证可重复的采样
        indices = np.random.choice(len(valid_pixels), num_samples, replace=False)
        sampled_pixels = valid_pixels[indices]
    else:
        sampled_pixels = valid_pixels

    print(f"Selected {len(sampled_pixels)} valid pixels for time-series evaluation.")

    start_time = time.time()
    
    # 确定并行 worker 数量
    if n_jobs <= 0:
        n_jobs = max(1, int(os.cpu_count() or 1)) if cpu_count is None else max(1, int(cpu_count()))
    
    print(f"Running predictions with {n_jobs} parallel workers...")
    
    pixel_results = []
    if JOBLIB_AVAILABLE and n_jobs > 1:
        # 使用 joblib 进行并行处理
        tasks = [delayed(_process_pixel_ts)(r, c, cube[:, r, c].copy(), t_days, t_sec, T, simulate_gap_days, args) for r, c in sampled_pixels]
        if TQDM_AVAILABLE:
            # 兼容 tqdm 的并行输出
            try:
                gen = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
                pixel_results = list(tqdm(gen, total=len(sampled_pixels), desc="Evaluating Pixels"))
            except TypeError:
                pixel_results = Parallel(n_jobs=n_jobs)(tasks)
        else:
            pixel_results = Parallel(n_jobs=n_jobs)(tasks)
    else:
        # 串行处理
        iterator = sampled_pixels
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(sampled_pixels), desc="Evaluating Pixels")
        for r, c in iterator:
            pixel_results.append(_process_pixel_ts(r, c, cube[:, r, c].copy(), t_days, t_sec, T, simulate_gap_days, args))

    # 汇总结果
    true_all_nufrost, pred_all_nufrost = [], []
    true_all_zhu, pred_all_zhu = [], []
    true_all_hants, pred_all_hants = [], []
    pixel_stats = []
    
    for res in pixel_results:
        if res is not None:
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
    print(f"\n[Dataset Missingness Profile (Sampled Pixels)]")
    print(f"Average Original Missing Ratio: {df_stats['Original_Missing_Ratio'].mean()*100:.2f}%")
    print(f"Average Max Continuous Gap: {df_stats['Original_Max_Gap_Days'].mean():.2f} days")
    
    # 计算总体指标
    true_all_nufrost = np.array(true_all_nufrost)
    pred_all_nufrost = np.array(pred_all_nufrost)
    true_all_zhu = np.array(true_all_zhu)
    pred_all_zhu = np.array(pred_all_zhu)
    true_all_hants = np.array(true_all_hants)
    pred_all_hants = np.array(pred_all_hants)
    
    metrics_n = compute_metrics(true_all_nufrost, pred_all_nufrost)
    metrics_z = compute_metrics(true_all_zhu, pred_all_zhu)
    metrics_h = compute_metrics(true_all_hants, pred_all_hants)
    
    results = [
        {"Algorithm": "NuFrost", "RMSE": metrics_n["RMSE"], "MAE": metrics_n["MAE"], "R": metrics_n["R"], "OutlierRatio": metrics_n["OutlierRatio"]},
        {"Algorithm": "Zhu2015", "RMSE": metrics_z["RMSE"], "MAE": metrics_z["MAE"], "R": metrics_z["R"], "OutlierRatio": metrics_z["OutlierRatio"]},
        {"Algorithm": "HANTS", "RMSE": metrics_h["RMSE"], "MAE": metrics_h["MAE"], "R": metrics_h["R"], "OutlierRatio": metrics_h["OutlierRatio"]},
    ]
    
    df_results = pd.DataFrame(results)
    print("\n========== Comprehensive Evaluation Summary ==========")
    print(f"(Tested by simulating {simulate_gap_days}-day continuous gaps)")
    print(df_results.to_string(index=False))
    
    return df_results


def _process_slice_pixel(r: int, c: int, y_ts: np.ndarray, t_sec: np.ndarray, t_days: np.ndarray, target_t_sec: float, target_t_day: float, args: Args) -> Dict:
    """处理单一时间截面的单像素预测（用于并行计算）"""
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
    
    return {"r": r, "c": c, "nufrost": pred_n, "zhu": pred_z, "hants": pred_h}


def evaluate_algorithms(
    image_path: str,
    args: Args,
    mask_type: str = "block",
    mask_ratio: float = 0.3,
    top_k_slices: int = 1,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    保留原有的单期评测方法，供对比参考 (支持并行)。
    """
    print(f"\n========== Starting Evaluation for {Path(image_path).name} ==========")
    
    loader = RSCube(image_path, cache_dir="./cache")
    data = loader.load()
    cube = data["cube"]
    timestamps = data["timestamps"]
    
    T, H, W = cube.shape
    
    t_sec = timestamps_to_seconds(timestamps, unit="seconds")
    t0_sec = np.min(t_sec)
    t_days = (t_sec - t0_sec) / 86400.0
    
    nan_counts = np.sum(np.isnan(cube), axis=(1, 2))
    best_indices = np.argsort(nan_counts)[:top_k_slices]
    
    if n_jobs <= 0:
        n_jobs = max(1, int(os.cpu_count() or 1)) if cpu_count is None else max(1, int(cpu_count()))
    
    results = []
    
    for idx in best_indices:
        target_t_sec = t_sec[idx]
        target_t_day = t_days[idx]
        target_time_str = str(timestamps[idx])
        
        print(f"\n--> Evaluating on Slice {idx} (Time: {target_time_str}), Original NaNs: {nan_counts[idx]}")
        
        y_true_full = cube[idx, :, :].copy()
        
        if mask_type == "block":
            mask = generate_block_mask((H, W), ratio=mask_ratio)
        else:
            mask = generate_random_mask((H, W), ratio=mask_ratio)
            
        mask = mask & np.isfinite(y_true_full)
        mask_pixels_count = np.sum(mask)
        print(f"--> Mask generated ({mask_type}). Masked pixels: {mask_pixels_count} ({(mask_pixels_count/(H*W))*100:.1f}%)")
        
        corrupted_cube = cube.copy()
        corrupted_cube[idx, mask] = np.nan
        
        mask_indices = np.where(mask)
        y_pred_nufrost = np.full((H, W), np.nan, dtype=np.float32)
        y_pred_zhu = np.full((H, W), np.nan, dtype=np.float32)
        y_pred_hants = np.full((H, W), np.nan, dtype=np.float32)
        
        start_time = time.time()
        
        # 准备并行任务
        print(f"--> Running predictions on masked pixels with {n_jobs} parallel workers...")
        tasks = [
            delayed(_process_slice_pixel)(
                r, c, corrupted_cube[:, r, c].copy(), t_sec, t_days, target_t_sec, target_t_day, args
            )
            for r, c in zip(mask_indices[0], mask_indices[1])
        ]
        
        slice_results = []
        if JOBLIB_AVAILABLE and n_jobs > 1:
            if TQDM_AVAILABLE:
                try:
                    gen = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
                    slice_results = list(tqdm(gen, total=len(tasks), desc="Processing Mask"))
                except TypeError:
                    slice_results = Parallel(n_jobs=n_jobs)(tasks)
            else:
                slice_results = Parallel(n_jobs=n_jobs)(tasks)
        else:
            iterator = tasks
            if TQDM_AVAILABLE:
                iterator = tqdm(iterator, total=len(tasks), desc="Processing Mask")
            for t in iterator:
                # delayed() 返回一个 (func, args, kwargs) 的 tuple，可以直接调用
                func, func_args, func_kwargs = t
                slice_results.append(func(*func_args, **func_kwargs))
                
        # 填充结果矩阵
        for res in slice_results:
            r, c = res["r"], res["c"]
            y_pred_nufrost[r, c] = res["nufrost"]
            y_pred_zhu[r, c] = res["zhu"]
            y_pred_hants[r, c] = res["hants"]
            
        print(f"--> Prediction finished in {time.time() - start_time:.2f}s.")
        
        metrics_nufrost = compute_metrics(y_true_full[mask], y_pred_nufrost[mask])
        metrics_zhu = compute_metrics(y_true_full[mask], y_pred_zhu[mask])
        metrics_hants = compute_metrics(y_true_full[mask], y_pred_hants[mask])
        
        if ssim is not None:
            full_pred_nufrost = np.where(mask, y_pred_nufrost, y_true_full)
            full_pred_zhu = np.where(mask, y_pred_zhu, y_true_full)
            full_pred_hants = np.where(mask, y_pred_hants, y_true_full)
            
            metrics_nufrost["SSIM"] = compute_metrics(y_true_full, full_pred_nufrost).get("SSIM", np.nan)
            metrics_zhu["SSIM"] = compute_metrics(y_true_full, full_pred_zhu).get("SSIM", np.nan)
            metrics_hants["SSIM"] = compute_metrics(y_true_full, full_pred_hants).get("SSIM", np.nan)

        for algo, m in zip(["NuFrost", "Zhu2015", "HANTS"], [metrics_nufrost, metrics_zhu, metrics_hants]):
            results.append({
                "Algorithm": algo,
                "TimeSlice": target_time_str,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "R": m["R"],
                "OutlierRatio": m.get("OutlierRatio", np.nan),
                "SSIM": m.get("SSIM", np.nan)
            })

    df_results = pd.DataFrame(results)
    print("\n========== Single-Slice Evaluation Summary ==========")
    print(df_results.to_string(index=False))
    return df_results