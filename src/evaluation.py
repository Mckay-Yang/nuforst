import numpy as np
import pandas as pd
import time
from typing import Tuple, Dict, List
from pathlib import Path

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

from .algorithms import timestamps_to_seconds, parse_timestamp_str, predict_single_pixel
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
    """计算单个算法的误差指标"""
    # 仅计算双边都有有效值的像素
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"RMSE": np.nan, "MAE": np.nan, "R": np.nan}
    
    yt = y_true[valid]
    yp = y_pred[valid]
    
    mse = np.mean((yt - yp) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(yt - yp))
    
    # 避免方差为0导致的除零错误
    if np.std(yt) == 0 or np.std(yp) == 0:
        r = 0.0
    else:
        r = np.corrcoef(yt, yp)[0, 1]
        
    metrics = {"RMSE": rmse, "MAE": mae, "R": r}
    
    # 如果形状是2D且有ssim，计算SSIM（将无效值设为0供ssim计算）
    if ssim is not None and y_true.ndim == 2:
        yt_img = np.nan_to_num(y_true)
        yp_img = np.nan_to_num(y_pred)
        # 指定 data_range 以避免全黑图像时的缩放问题
        data_range = max(yt_img.max() - yt_img.min(), 1.0)
        metrics["SSIM"] = ssim(yt_img, yp_img, data_range=data_range)
        
    return metrics

def evaluate_algorithms(
    image_path: str,
    args: Args,
    mask_type: str = "block",
    mask_ratio: float = 0.3,
    top_k_slices: int = 1
) -> pd.DataFrame:
    """
    评测所有算法。
    逻辑：找到最完整的时间切片 -> 挖空 -> 预测 -> 对比。
    """
    print(f"\n========== Starting Evaluation for {Path(image_path).name} ==========")
    
    # 1. 载入数据
    loader = RSCube(image_path, cache_dir="./cache")
    data = loader.load()
    cube = data["cube"]
    timestamps = data["timestamps"]
    
    T, H, W = cube.shape
    
    # 2. 时间格式转换
    t_sec = timestamps_to_seconds(timestamps, unit="seconds")
    t0_sec = np.min(t_sec)
    t_days = (t_sec - t0_sec) / 86400.0
    
    # 3. 寻找最优质的一期影像 (NaN 最少的波段)
    nan_counts = np.sum(np.isnan(cube), axis=(1, 2))
    best_indices = np.argsort(nan_counts)[:top_k_slices]
    
    results = []
    
    for idx in best_indices:
        target_t_sec = t_sec[idx]
        target_t_day = t_days[idx]
        target_time_str = str(timestamps[idx])
        
        print(f"\n--> Evaluating on Slice {idx} (Time: {target_time_str}), Original NaNs: {nan_counts[idx]}")
        
        # 获取 Ground Truth
        y_true_full = cube[idx, :, :].copy()
        
        # 4. 生成掩膜并在 Cube 中模拟破坏
        if mask_type == "block":
            mask = generate_block_mask((H, W), ratio=mask_ratio)
        else:
            mask = generate_random_mask((H, W), ratio=mask_ratio)
            
        # 防止覆盖真实无数据区域（原本就是NaN的不算作被挖掉的）
        mask = mask & np.isfinite(y_true_full)
        mask_pixels_count = np.sum(mask)
        print(f"--> Mask generated ({mask_type}). Masked pixels: {mask_pixels_count} ({(mask_pixels_count/(H*W))*100:.1f}%)")
        
        # 破坏 Cube（这一期的这些像素被设为NaN）
        corrupted_cube = cube.copy()
        corrupted_cube[idx, mask] = np.nan
        
        # 5. 开始针对每个算法在 Mask 区域预测
        mask_indices = np.where(mask)
        y_pred_nufrost = np.full((H, W), np.nan, dtype=np.float32)
        y_pred_zhu = np.full((H, W), np.nan, dtype=np.float32)
        y_pred_hants = np.full((H, W), np.nan, dtype=np.float32)
        
        print("--> Running predictions on masked pixels...")
        start_time = time.time()
        
        # 逐像素预测被掩盖的部分
        for r, c in zip(mask_indices[0], mask_indices[1]):
            y_ts = corrupted_cube[:, r, c]
            
            # --- NuFrost 预测 ---
            pred_n, _ = predict_single_pixel(
                t_sec, y_ts, target_t_sec,
                nufft_modes=args.modes, eps=args.eps,
                num_peaks=args.num_peaks, power_cum=args.power_cum, ignore_dc_hz=args.ignore_dc_hz,
                refine_peaks=args.refine_peaks, include_trend=args.include_trend,
                ridge_lam=args.ridge, freq_weight=args.freq_weight, huber_iters=args.huber_iters, huber_delta=args.huber_delta,
                min_obs=args.min_obs
            )
            y_pred_nufrost[r, c] = pred_n
            
            # --- Zhu2015 预测 ---
            pred_z, _ = fit_predict_pixel(t_days, y_ts, target_t_day, lasso_alpha=0.0001)
            y_pred_zhu[r, c] = pred_z
            
            # --- HANTS 预测 ---
            pred_h = hants_pixel(t_days, y_ts, target_t_day, nof=3, sf='low', fet=0.05, dod=5)
            y_pred_hants[r, c] = pred_h
            
        print(f"--> Prediction finished in {time.time() - start_time:.2f}s.")
        
        # 6. 计算指标 (只在被遮挡的部分计算)
        metrics_nufrost = compute_metrics(y_true_full[mask], y_pred_nufrost[mask])
        metrics_zhu = compute_metrics(y_true_full[mask], y_pred_zhu[mask])
        metrics_hants = compute_metrics(y_true_full[mask], y_pred_hants[mask])
        
        # 如果需要计算SSIM，我们将整张影像合并（周围用真实值，洞用预测值）来算
        if ssim is not None:
            full_pred_nufrost = np.where(mask, y_pred_nufrost, y_true_full)
            full_pred_zhu = np.where(mask, y_pred_zhu, y_true_full)
            full_pred_hants = np.where(mask, y_pred_hants, y_true_full)
            
            metrics_nufrost["SSIM"] = compute_metrics(y_true_full, full_pred_nufrost).get("SSIM", np.nan)
            metrics_zhu["SSIM"] = compute_metrics(y_true_full, full_pred_zhu).get("SSIM", np.nan)
            metrics_hants["SSIM"] = compute_metrics(y_true_full, full_pred_hants).get("SSIM", np.nan)

        # 记录结果
        for algo, m in zip(["NuFrost", "Zhu2015", "HANTS"], [metrics_nufrost, metrics_zhu, metrics_hants]):
            results.append({
                "Algorithm": algo,
                "TimeSlice": target_time_str,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "R": m["R"],
                "SSIM": m.get("SSIM", np.nan)
            })

    df_results = pd.DataFrame(results)
    print("\n========== Evaluation Summary ==========")
    print(df_results.to_string(index=False))
    return df_results
