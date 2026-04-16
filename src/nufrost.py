import os
import math
from typing import Any, Dict, Optional, Tuple, Sequence, Union, cast
from datetime import datetime
import numpy as np
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import finufft  # type: ignore
except ModuleNotFoundError:
    raise ModuleNotFoundError("finufft is required. Install with: pip install finufft")

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

import rasterio

from config import Args, build_args
from .data_loader import RSCube


def _make_frequency_penalty(
    freqs: Optional[np.ndarray],
    freq_weight: float,
) -> np.ndarray:
    """为谐波项构建随频率递增但更平滑的 ridge 惩罚系数。"""
    if freqs is None or len(freqs) == 0:
        return np.zeros(0, dtype=np.float64)

    freqs = np.asarray(freqs, dtype=np.float64)
    positive = freqs[np.isfinite(freqs) & (freqs > 0)]
    if positive.size == 0:
        return np.ones(len(freqs), dtype=np.float64)

    base_freq = float(np.min(positive))
    if not np.isfinite(base_freq) or base_freq <= 0:
        base_freq = 1.0

    rel = np.maximum(freqs / base_freq, 1.0)
    # 使用对数型增长而非幂律增长，避免高频惩罚过强。
    # freq_weight=0 -> 所有频率等权；
    # freq_weight>0 -> 高频相对低频受到更强惩罚，但增长更平滑。
    penalty_scale = 1.0 + max(0.0, freq_weight) * np.log2(rel)
    # 这里返回 sqrt(scale)，因为后续增广矩阵里还会再乘一次平方根的 lam，
    # 最终目标函数中的惩罚项就是 lam * scale * beta^2。
    return np.sqrt(penalty_scale)

def _to_seconds_since_start(ts_utc: np.ndarray) -> np.ndarray:
    t0 = np.min(ts_utc)
    return np.ascontiguousarray(ts_utc - t0, dtype=np.float64)

def _parse_timestamp_str(ts: str) -> Optional[datetime]:
    if PANDAS_AVAILABLE:
        try:
            # 优先使用 pandas 解析，并统一转为 UTC，这与 bak/revive_test.py 逻辑一致
            dt = pd.to_datetime(ts, utc=True)
            return dt.to_pydatetime()
        except Exception:
            pass

    ts = str(ts).strip()
    fmts = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%dT%H%M%S",
        "%Y%m%d",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None

def _datetime64_to_seconds(ts: np.datetime64) -> float:
    epoch = np.datetime64("1970-01-01T00:00:00")
    return float((ts - epoch) / np.timedelta64(1, "s"))

def timestamps_to_seconds(timestamps: np.ndarray, unit: str = "seconds") -> np.ndarray:
    out = []
    for ts in timestamps:
        if isinstance(ts, datetime):
            out.append(ts.timestamp())
            continue
        if isinstance(ts, np.datetime64):
            out.append(_datetime64_to_seconds(ts))
            continue
        dt = _parse_timestamp_str(str(ts))
        if dt is None:
            out.append(np.nan)
            continue
        out.append(dt.timestamp())
    arr = np.array(out, dtype=np.float64)
    if unit == "days":
        arr = arr / 86400.0
    return arr

def parse_timestamp_str(ts: str) -> Optional[datetime]:
    return _parse_timestamp_str(ts)

def next_even(n: int) -> int:
    return int(np.ceil(n/2.0))*2

def refine_parabolic(f: np.ndarray, P: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(P)-1:
        return float(f[i])
    y0,y1,y2 = P[i-1],P[i],P[i+1]
    denom = (y0 - 2*y1 + y2)
    if denom == 0:
        return float(f[i])
    delta = 0.5*(y0 - y2)/denom
    return float(f[i] + delta*(f[i+1] - f[i]))


def _parse_preferred_periods_days(periods_spec: Union[str, Sequence[float], np.ndarray]) -> np.ndarray:
    if isinstance(periods_spec, str):
        parts = [p.strip() for p in periods_spec.split(',') if p.strip()]
        vals = [float(p) for p in parts]
    else:
        vals = [float(v) for v in periods_spec]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return np.array(vals, dtype=np.float64)


def _preferred_periods_to_freqs(periods_days: Union[str, Sequence[float], np.ndarray], time_unit: str = "seconds") -> np.ndarray:
    periods_days_arr = _parse_preferred_periods_days(periods_days)
    if periods_days_arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    if time_unit == "days":
        return 1.0 / periods_days_arr
    return 1.0 / (periods_days_arr * 86400.0)


def _snap_frequency_to_spectrum(target_freq: float, f_pos: np.ndarray, P_pos: np.ndarray, rel_tol: float) -> float:
    if not np.isfinite(target_freq) or target_freq <= 0:
        return float(target_freq)
    valid = np.isfinite(f_pos) & np.isfinite(P_pos) & (f_pos > 0)
    if not np.any(valid):
        return float(target_freq)
    cand_f = f_pos[valid]
    cand_p = P_pos[valid]
    rel_err = np.abs(cand_f - target_freq) / max(target_freq, 1e-12)
    nearby = np.where(rel_err <= max(rel_tol, 0.0))[0]
    if nearby.size == 0:
        return float(target_freq)
    best_local = nearby[np.argmax(cand_p[nearby])]
    return float(cand_f[best_local])


def select_frequencies(
    f_pos: np.ndarray,
    P_pos: np.ndarray,
    fmax: float,
    selection_mode: str,
    preferred_freqs: np.ndarray,
    preferred_top_k: int,
    spectral_top_k: int,
    spectral_merge_tol: float,
    power_cum: float,
    ignore_dc_hz: float,
    refine_peaks: bool,
) -> np.ndarray:
    """
    频率选择策略：
    - spectral: 纯谱峰选择
    - preferred: 优先使用先验周期频率，并吸附到附近谱峰
    - hybrid: 先验频率 + 数据驱动谱峰联合选择
    """
    selected: list[float] = []

    if selection_mode in ("preferred", "hybrid") and preferred_freqs.size > 0:
        preferred_valid = preferred_freqs[np.isfinite(preferred_freqs) & (preferred_freqs > ignore_dc_hz)]
        preferred_valid = preferred_valid[preferred_valid <= fmax]
        for f in preferred_valid[:max(0, preferred_top_k)]:
            selected.append(_snap_frequency_to_spectrum(f, f_pos, P_pos, spectral_merge_tol))

    if selection_mode in ("spectral", "hybrid"):
        peak_idx = select_peaks_adaptive(
            f_pos, P_pos,
            k_max=max(0, spectral_top_k),
            power_cum=power_cum,
            ignore_dc_hz=ignore_dc_hz,
            fmax=fmax,
        )
        if len(peak_idx) > 0:
            if refine_peaks:
                spectral_freqs = [refine_parabolic(f_pos, P_pos, i) for i in peak_idx]
            else:
                spectral_freqs = [float(f_pos[i]) for i in peak_idx]
            selected.extend(spectral_freqs)

    if not selected:
        return np.zeros(0, dtype=np.float64)

    selected = sorted(float(f) for f in selected if np.isfinite(f) and f > ignore_dc_hz and f <= fmax)
    merged: list[float] = []
    for f in selected:
        if not merged:
            merged.append(f)
            continue
        rel = abs(f - merged[-1]) / max(merged[-1], 1e-12)
        if rel <= max(spectral_merge_tol, 0.0):
            merged[-1] = 0.5 * (merged[-1] + f)
        else:
            merged.append(f)
    return np.array(merged, dtype=np.float64)

def select_peaks_adaptive(f_pos: np.ndarray, P_pos: np.ndarray, k_max: int, power_cum: float, ignore_dc_hz: float, fmax: float) -> np.ndarray:
    lower = max(ignore_dc_hz, 0.0)
    if not np.isfinite(fmax) or fmax <= 0:
        fmax = np.nanmax(f_pos[np.isfinite(f_pos)]) or 1.0
    valid = np.isfinite(f_pos) & np.isfinite(P_pos) & (f_pos > lower) & (f_pos <= fmax)
    if not np.any(valid):
        return np.array([], dtype=int)
    idx = np.where(valid)[0]
    order = np.argsort(-P_pos[idx])
    idx_sorted = idx[order]
    cum = np.cumsum(P_pos[idx_sorted])
    thr = np.clip(power_cum, 0.0, 1.0) * cum[-1]
    take = np.searchsorted(cum, thr) + 1
    take = min(take, k_max, len(idx_sorted))
    return idx_sorted[:take]

def design_matrix(t: np.ndarray, freqs: Union[Sequence[float], np.ndarray], include_trend: bool = True, include_dc: bool = True) -> np.ndarray:
    cols = []
    if include_dc:
        cols.append(np.ones_like(t))
    if include_trend:
        cols.append(t - t.mean())
    for f in freqs:
        w = 2*np.pi*f
        cols.append(np.cos(w*t))
        cols.append(np.sin(w*t))
    return np.vstack(cols).T if cols else np.empty((len(t),0))

def huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    w = np.ones_like(r)
    m = a > delta
    w[m] = (delta / a[m])
    return w

def _safe_lstsq(X: np.ndarray, y: np.ndarray, rcond: Optional[float] = None) -> np.ndarray:
    """更稳健的 lstsq：失败则小抖动回退；再失败提升 ridge。"""
    try:
        return np.linalg.lstsq(X, y, rcond=rcond)[0]
    except Exception:
        # 加一个小的对角抖动
        try:
            eps = 1e-8
            p = X.shape[1]
            return np.linalg.lstsq(
                np.vstack([X, np.sqrt(eps)*np.eye(p)]),
                np.concatenate([y, np.zeros(p)]),
                rcond=None
            )[0]
        except Exception:
            # 再失败就进一步增大 ridge
            eps2 = 1e-4
            p = X.shape[1]
            return np.linalg.lstsq(
                np.vstack([X, np.sqrt(eps2)*np.eye(p)]),
                np.concatenate([y, np.zeros(p)]),
                rcond=None
            )[0]

def ridge_with_freq_weights(X: np.ndarray, y: np.ndarray, freqs: Optional[np.ndarray], lam: float, include_dc: bool = True, include_trend: bool = True, freq_weight: float = 2.0, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return np.zeros(0, dtype=float), np.full_like(y, np.nanmean(y))
    if w is None:
        w = np.ones_like(y)
    W = np.sqrt(w)
    Xw = X * W[:, None]
    yw = y * W

    if lam <= 0:
        beta = _safe_lstsq(Xw, yw)
        y_hat = X @ beta
        return beta, y_hat

    p = X.shape[1]
    R = np.zeros(p, dtype=np.float64)
    col = 0
    if include_dc:
        col += 1
    if include_trend:
        col += 1
    if freqs is not None and len(freqs) > 0:
        penalty = _make_frequency_penalty(np.asarray(freqs, dtype=np.float64), freq_weight)
        for w_f in penalty:
            if col < p:
                R[col] = w_f
                col += 1
            if col < p:
                R[col] = w_f
                col += 1

    if np.any(R > 0):
        X_aug = np.vstack([Xw, np.diag(np.sqrt(lam)*R)])
    else:
        X_aug = np.vstack([Xw, np.sqrt(lam)*np.eye(p)])
    y_aug = np.concatenate([yw, np.zeros(p, dtype=np.float64)])

    beta = _safe_lstsq(X_aug, y_aug)
    y_hat = X @ beta
    return beta, y_hat

def robust_fit_freq_ridge(X: np.ndarray, y: np.ndarray, freqs: np.ndarray, lam: float, iters: int, delta: float, include_dc: bool, include_trend: bool, freq_weight: float) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[1] == 0:
        return np.zeros(0), np.full_like(y, np.nanmean(y))
    w = np.ones_like(y)
    y_hat = np.zeros_like(y)
    for _ in range(max(0, iters)):
        r = y - y_hat
        w = huber_weights(r, max(1e-8, delta))
        _, y_hat = ridge_with_freq_weights(X, y, freqs, lam,
                                           include_dc, include_trend, freq_weight, w=w)
    beta, y_hat = ridge_with_freq_weights(X, y, freqs, lam,
                                          include_dc, include_trend, freq_weight, w=w)
    return beta, y_hat


def fit_nufrost_pixel_params(t_sec: np.ndarray, y: np.ndarray, target_t: Optional[float] = None,
                             nufft_modes: int = 4096, eps: float = 1e-12,
                             num_peaks: int = 10, power_cum: float = 0.7, ignore_dc_hz: float = 1e-10,
                              frequency_selection: str = "hybrid", preferred_periods_days: Union[str, Sequence[float], np.ndarray] = "365.25,182.625,91.3125,30.4375", preferred_top_k: int = 4, spectral_top_k: int = 4, spectral_merge_tol: float = 0.15,
                              refine_peaks: bool = True, include_trend: bool = True,
                              ridge_lam: float = 0.005, freq_weight: float = 2.0, huber_iters: int = 3, huber_delta: float = 1.5,
                              min_obs: int = 12, max_freqs: int = 10) -> Dict[str, Any]:
    beta_size = 1 + (1 if include_trend else 0) + 2 * max(0, max_freqs)
    params: Dict[str, Any] = {
        "valid": False,
        "include_trend": bool(include_trend),
        "n_freqs_used": 0,
        "t_min": np.nan,
        "t_rel_mean": np.nan,
        "fill_value": np.nan,
        "freqs": np.full(max(0, max_freqs), np.nan, dtype=np.float64),
        "beta": np.full(beta_size, np.nan, dtype=np.float64),
    }

    m = np.isfinite(y) & np.isfinite(t_sec)
    if m.sum() < max(3, min_obs):
        return params
    t = np.asarray(t_sec[m], dtype=np.float64)
    yy = np.asarray(y[m], dtype=np.float64)
    params["fill_value"] = float(np.nanmean(yy))

    t_rel = _to_seconds_since_start(t)
    t_rel_mean = float(t_rel.mean())
    Tspan = float(t_rel.max() - t_rel.min())
    if not np.isfinite(Tspan) or Tspan <= 0:
        return params

    x = 2 * np.pi * (t_rel - t_rel.min()) / Tspan - np.pi
    x = np.ascontiguousarray(x, dtype=np.float64)
    c = np.ascontiguousarray(yy.astype(np.complex128))
    ms = next_even(nufft_modes)
    Fk = finufft.nufft1d1(x, c, ms, eps=eps, isign=-1)
    k = np.arange(-ms // 2, ms // 2, dtype=np.int64)
    freqs = k.astype(np.float64) / Tspan

    pos = freqs >= 0
    f_pos = freqs[pos]
    P_pos = np.abs(Fk[pos]) ** 2

    dt = np.diff(np.sort(t_rel))
    dt_pos = dt[dt > 0]
    dt_med = float(np.median(dt_pos)) if dt_pos.size else Tspan / len(t_rel)
    fmax = 0.5 / max(dt_med, 1e-12)

    preferred_freqs = _preferred_periods_to_freqs(preferred_periods_days, time_unit="seconds")
    spectral_top_k_eff = spectral_top_k if spectral_top_k > 0 else num_peaks
    freqs_sel = select_frequencies(
        f_pos=f_pos,
        P_pos=P_pos,
        fmax=fmax,
        selection_mode=frequency_selection,
        preferred_freqs=preferred_freqs,
        preferred_top_k=preferred_top_k,
        spectral_top_k=spectral_top_k_eff,
        spectral_merge_tol=spectral_merge_tol,
        power_cum=power_cum,
        ignore_dc_hz=ignore_dc_hz,
        refine_peaks=refine_peaks,
    )
    freqs_sel = np.asarray(freqs_sel[:max(0, max_freqs)], dtype=np.float64)
    X = design_matrix(t_rel, freqs_sel, include_trend=include_trend, include_dc=True)
    beta, _ = robust_fit_freq_ridge(
        X,
        yy,
        freqs_sel,
        lam=ridge_lam,
        iters=huber_iters,
        delta=huber_delta,
        include_dc=True,
        include_trend=include_trend,
        freq_weight=freq_weight,
    )

    params["valid"] = True
    params["n_freqs_used"] = int(len(freqs_sel))
    params["t_min"] = float(t.min())
    params["t_rel_mean"] = t_rel_mean
    if len(freqs_sel) > 0:
        params["freqs"][: len(freqs_sel)] = freqs_sel
    params["beta"][: len(beta)] = beta
    return params


def predict_nufrost_from_params(params: Dict[str, Any], target_t: float) -> float:
    if not bool(params.get("valid", False)):
        return float(params.get("fill_value", np.nan))
    include_trend = bool(params["include_trend"])
    n_freqs = int(params["n_freqs_used"])
    freqs = np.asarray(params["freqs"], dtype=np.float64)[:n_freqs]
    beta = np.asarray(params["beta"], dtype=np.float64)
    t_star_rel = float(target_t - float(params["t_min"]))
    cols = [1.0]
    if include_trend:
        cols.append(t_star_rel - float(params["t_rel_mean"]))
    for f in freqs:
        w = 2 * np.pi * f
        cols.append(math.cos(w * t_star_rel))
        cols.append(math.sin(w * t_star_rel))
    return float(np.asarray(cols, dtype=np.float64) @ beta[: len(cols)])


def predict_nufrost_curve_from_params(params: Dict[str, Any], target_t_secs: np.ndarray) -> np.ndarray:
    return np.array([predict_nufrost_from_params(params, float(target_t)) for target_t in target_t_secs], dtype=np.float64)

def predict_single_pixel(t_sec: np.ndarray, y: np.ndarray, target_t: float,
                         nufft_modes: int, eps: float,
                         num_peaks: int, power_cum: float, ignore_dc_hz: float,
                          frequency_selection: str = "hybrid", preferred_periods_days: Union[str, Sequence[float], np.ndarray] = "365.25,182.625,91.3125,30.4375", preferred_top_k: int = 4, spectral_top_k: int = 4, spectral_merge_tol: float = 0.15,
                          refine_peaks: bool = True, include_trend: bool = True,
                          ridge_lam: float = 0.005, freq_weight: float = 2.0, huber_iters: int = 3, huber_delta: float = 1.5,
                          min_obs: int = 12) -> Tuple[float, int]:
    params = fit_nufrost_pixel_params(
        t_sec,
        y,
        nufft_modes=nufft_modes,
        eps=eps,
        num_peaks=num_peaks,
        power_cum=power_cum,
        ignore_dc_hz=ignore_dc_hz,
        frequency_selection=frequency_selection,
        preferred_periods_days=preferred_periods_days,
        preferred_top_k=preferred_top_k,
        spectral_top_k=spectral_top_k,
        spectral_merge_tol=spectral_merge_tol,
        refine_peaks=refine_peaks,
        include_trend=include_trend,
        ridge_lam=ridge_lam,
        freq_weight=freq_weight,
        huber_iters=huber_iters,
        huber_delta=huber_delta,
        min_obs=min_obs,
        max_freqs=max(num_peaks, spectral_top_k, preferred_top_k, 1),
    )
    return predict_nufrost_from_params(params, target_t), int(params["n_freqs_used"])

def predict_curve_pixel(t_sec: np.ndarray, y: np.ndarray, target_t_secs: np.ndarray,
                         nufft_modes: int, eps: float,
                         num_peaks: int, power_cum: float, ignore_dc_hz: float,
                          frequency_selection: str = "hybrid", preferred_periods_days: Union[str, Sequence[float], np.ndarray] = "365.25,182.625,91.3125,30.4375", preferred_top_k: int = 4, spectral_top_k: int = 4, spectral_merge_tol: float = 0.15,
                          refine_peaks: bool = True, include_trend: bool = True,
                          ridge_lam: float = 0.005, freq_weight: float = 2.0, huber_iters: int = 3, huber_delta: float = 1.5,
                          min_obs: int = 12) -> np.ndarray:
    params = fit_nufrost_pixel_params(
        t_sec,
        y,
        nufft_modes=nufft_modes,
        eps=eps,
        num_peaks=num_peaks,
        power_cum=power_cum,
        ignore_dc_hz=ignore_dc_hz,
        frequency_selection=frequency_selection,
        preferred_periods_days=preferred_periods_days,
        preferred_top_k=preferred_top_k,
        spectral_top_k=spectral_top_k,
        spectral_merge_tol=spectral_merge_tol,
        refine_peaks=refine_peaks,
        include_trend=include_trend,
        ridge_lam=ridge_lam,
        freq_weight=freq_weight,
        huber_iters=huber_iters,
        huber_delta=huber_delta,
        min_obs=min_obs,
        max_freqs=max(num_peaks, spectral_top_k, preferred_top_k, 1),
    )
    return predict_nufrost_curve_from_params(params, target_t_secs)

def reconstruct_nufrost(
    image: Union[str, Path],
    target_time: str,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> np.ndarray:
    """
    一键重建接口。

    参数:
        image: 输入的多波段 TIF 路径
        target_time: 目标重建时间 (如 '2023-06-15')
        output_path: 输出 TIF 路径 (可选)
        **kwargs: 其他算法参数 (如 n_jobs, ridge, etc.)
    """
    # 1. 构建参数
    overrides = {**kwargs, "image": image, "target_time": target_time}
    if output_path:
        overrides["output_path"] = output_path

    args = build_args(overrides=overrides)

    # 2. 加载数据
    loader = RSCube(args.image, cache_dir=args.cache_dir, force_refresh=args.force_refresh)
    data = loader.load()
    cube = np.ma.filled(data["cube"], np.nan)
    timestamps = data["timestamps"]

    # 3. 执行重建
    recon = nufrost_core(cube, timestamps, args.target_time, args=args)

    # 4. 保存结果
    if args.output_path:
        out_p = Path(args.output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        transform = None
        if "transform" in data:
            transform = rasterio.Affine(*data["transform"])

        with rasterio.open(
            out_p, "w",
            driver="GTiff",
            height=recon.shape[0],
            width=recon.shape[1],
            count=1,
            dtype=recon.dtype,
            crs=data.get("crs_wkt"),
            transform=transform,
        ) as dst:
            dst.write(recon, 1)
        print(f"[Success] Saved to: {out_p}")

    return recon

def nufrost_core(cube: np.ndarray, timestamps: np.ndarray, target_time: str, args: Optional[Args] = None, **kwargs) -> np.ndarray:
    """核心重建函数，支持直接传入参数或 Args 对象"""
    # 合并参数
    if args is None:
        from config import Args
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
                frequency_selection=args.frequency_selection,
                preferred_periods_days=args.preferred_periods_days,
                preferred_top_k=args.preferred_top_k,
                spectral_top_k=args.spectral_top_k,
                spectral_merge_tol=args.spectral_merge_tol,
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

    print(f"[System] Starting NuFrost reconstruction with {n_jobs} jobs...")

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
        if args.show_progress and TQDM_AVAILABLE:
            # 尝试使用 generator 模式配合 tqdm 显示进度
            try:
                results_gen = Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator")(
                    delayed(_predict_row)(i) for i in range(H)
                )
                for i, row in tqdm(results_gen, total=H, desc="Processing Rows"):
                    out[i, :] = row
                return out
            except TypeError:
                # 如果 joblib 版本太老不支持 return_as="generator"，回退到普通模式
                pass

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
