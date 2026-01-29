import math
from typing import Optional, Tuple, Sequence, Union, cast
from datetime import datetime
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import finufft  # type: ignore
except ModuleNotFoundError:
    raise ModuleNotFoundError("finufft is required. Install with: pip install finufft")

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
        fmax_local = np.max(freqs) if np.max(freqs) > 0 else 1.0
        if not np.isfinite(fmax_local) or fmax_local <= 0:
            fmax_local = 1.0
        for f in freqs:
            w_f = (max(f, 0.0) / fmax_local) ** max(0.0, freq_weight)
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

def predict_single_pixel(t_sec: np.ndarray, y: np.ndarray, target_t: float,
                         nufft_modes: int, eps: float,
                         num_peaks: int, power_cum: float, ignore_dc_hz: float,
                         refine_peaks: bool, include_trend: bool,
                         ridge_lam: float, freq_weight: float, huber_iters: int, huber_delta: float,
                         min_obs: int) -> Tuple[float, int]:
    m = np.isfinite(y) & np.isfinite(t_sec)
    if m.sum() < max(3, min_obs):
        return np.nan, 0
    t = np.asarray(t_sec[m], dtype=np.float64)
    yy = np.asarray(y[m], dtype=np.float64)

    t_rel = _to_seconds_since_start(t)
    t_rel_mean = float(t_rel.mean())
    Tspan = float(t_rel.max() - t_rel.min())
    if not np.isfinite(Tspan) or Tspan <= 0:
        return np.nan, 0

    x = 2*np.pi*(t_rel - t_rel.min())/Tspan - np.pi
    x = np.ascontiguousarray(x, dtype=np.float64)
    c = np.ascontiguousarray(yy.astype(np.complex128))
    ms = next_even(nufft_modes)
    Fk = finufft.nufft1d1(x, c, ms, eps=eps, isign=-1)
    k = np.arange(-ms//2, ms//2, dtype=np.int64)
    freqs = k.astype(np.float64)/Tspan

    pos = freqs >= 0
    f_pos = freqs[pos]
    P_pos = (np.abs(Fk[pos])**2)

    dt = np.diff(np.sort(t_rel))
    dt_pos = dt[dt > 0]
    dt_med = float(np.median(dt_pos)) if dt_pos.size else Tspan/len(t_rel)
    fmax = 0.5 / max(dt_med, 1e-12)

    pos_idx = select_peaks_adaptive(f_pos, P_pos, k_max=num_peaks,
                                    power_cum=power_cum,
                                    ignore_dc_hz=ignore_dc_hz, fmax=fmax)
    if len(pos_idx) == 0:
        freqs_sel = np.array([], dtype=np.float64)
    else:
        if refine_peaks:
            freqs_sel = np.array([refine_parabolic(f_pos, P_pos, i) for i in pos_idx], dtype=np.float64)
        else:
            freqs_sel = np.array(f_pos[pos_idx], dtype=np.float64)

    X = design_matrix(t_rel, freqs_sel, include_trend=include_trend, include_dc=True)
    beta, _ = robust_fit_freq_ridge(X, yy, freqs_sel,
                                    lam=ridge_lam, iters=huber_iters, delta=huber_delta,
                                    include_dc=True, include_trend=include_trend, freq_weight=freq_weight)

    t_star_rel = float(target_t - t.min())
    cols = [1.0]
    if include_trend:
        cols.append(t_star_rel - t_rel_mean)
    for f in freqs_sel:
        w = 2*np.pi*f
        cols.append(math.cos(w*t_star_rel))
        cols.append(math.sin(w*t_star_rel))
    X_star = np.array(cols, dtype=np.float64).reshape(1, -1) if len(cols) else np.zeros((1,0))

    if X.shape[1] > 0:
        y_star = float((X_star @ beta).item())
    else:
        y_star = float(np.nanmean(yy))
    return y_star, len(freqs_sel)
