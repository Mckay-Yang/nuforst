from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import Args, build_args
from .data_loader import TimeSeriesRasterSource
from .hants import fit_hants_pixel_params, predict_hants_from_params
from .nufrost import fit_nufrost_pixel_params, parse_timestamp_str, predict_nufrost_from_params, timestamps_to_seconds
from .zhu2015 import fit_zhu2015_pixel_params, predict_zhu2015_from_params


def _block_slices(height: int, width: int, block_shape: Tuple[int, int]) -> list[Tuple[slice, slice]]:
    block_h, block_w = block_shape
    return [
        (slice(row0, min(row0 + block_h, height)), slice(col0, min(col0 + block_w, width)))
        for row0 in range(0, height, block_h)
        for col0 in range(0, width, block_w)
    ]


def _target_time_to_algorithm_time(target_time: str, params: Dict[str, Any]) -> float:
    dt = parse_timestamp_str(target_time)
    if dt is None:
        raise ValueError(f"Unrecognized target_time: {target_time}")
    target_sec = float(dt.timestamp())
    if str(params["algorithm"]) == "nufrost":
        return target_sec
    return (target_sec - float(params["time_origin_sec"])) / 86400.0


def fit_cube_params(
    cube: np.ndarray,
    timestamps: np.ndarray,
    algorithm: str,
    args: Optional[Args] = None,
    block_shape: Tuple[int, int] = (64, 64),
    max_freqs: int = 10,
    max_segments: int = 10,
) -> Dict[str, Any]:
    if args is None:
        args = build_args(algorithm.lower(), {})

    algorithm = algorithm.lower()
    _, height, width = cube.shape
    t_sec = timestamps_to_seconds(timestamps, unit=args.time_unit)
    time_origin_sec = float(np.nanmin(t_sec))
    t_days = (t_sec - time_origin_sec) / 86400.0

    params: Dict[str, Any] = {
        "algorithm": np.array(algorithm),
        "shape": np.array([height, width], dtype=np.int32),
        "time_origin_sec": np.array(time_origin_sec, dtype=np.float64),
    }

    if algorithm == "nufrost":
        beta_size = 1 + (1 if args.include_trend else 0) + 2 * max_freqs
        params.update(
            {
                "valid": np.zeros((height, width), dtype=np.int8),
                "n_freqs_used": np.zeros((height, width), dtype=np.int16),
                "t_min": np.full((height, width), np.nan, dtype=np.float64),
                "t_rel_mean": np.full((height, width), np.nan, dtype=np.float64),
                "fill_value": np.full((height, width), np.nan, dtype=np.float64),
                "freqs": np.full((height, width, max_freqs), np.nan, dtype=np.float64),
                "beta": np.full((height, width, beta_size), np.nan, dtype=np.float64),
                "include_trend": np.array(bool(args.include_trend), dtype=np.int8),
            }
        )
    elif algorithm == "hants":
        coeff_count = 1 + 2 * (3 - 1)
        params.update(
            {
                "valid": np.zeros((height, width), dtype=np.int8),
                "coeffs": np.full((height, width, coeff_count), np.nan, dtype=np.float64),
                "fill_value": np.full((height, width), np.nan, dtype=np.float64),
                "nof": np.full((height, width), 3, dtype=np.int16),
                "period": np.full((height, width), 365.25, dtype=np.float64),
            }
        )
    elif algorithm == "zhu2015":
        params.update(
            {
                "valid": np.zeros((height, width), dtype=np.int8),
                "n_segments": np.zeros((height, width), dtype=np.int16),
                "segment_start_days": np.full((height, width, max_segments), np.nan, dtype=np.float64),
                "segment_end_days": np.full((height, width, max_segments), np.nan, dtype=np.float64),
                "segment_orders": np.zeros((height, width, max_segments), dtype=np.int16),
                "segment_has_model": np.zeros((height, width, max_segments), dtype=np.int8),
                "segment_median_values": np.full((height, width, max_segments), np.nan, dtype=np.float64),
                "segment_intercepts": np.zeros((height, width, max_segments), dtype=np.float64),
                "segment_coefficients": np.zeros((height, width, max_segments, 2 * 3 + 1), dtype=np.float64),
                "max_order": np.array(3, dtype=np.int16),
            }
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    for row_slice, col_slice in _block_slices(height, width, block_shape):
        for i in range(row_slice.start, row_slice.stop):
            for j in range(col_slice.start, col_slice.stop):
                y = cube[:, i, j]
                if algorithm == "nufrost":
                    pixel = fit_nufrost_pixel_params(
                        t_sec,
                        y,
                        nufft_modes=args.modes,
                        eps=args.eps,
                        num_peaks=args.num_peaks,
                        power_cum=args.power_cum,
                        ignore_dc_hz=args.ignore_dc_hz,
                        frequency_selection=args.frequency_selection,
                        preferred_periods_days=args.preferred_periods_days,
                        preferred_top_k=args.preferred_top_k,
                        spectral_top_k=args.spectral_top_k,
                        spectral_merge_tol=args.spectral_merge_tol,
                        refine_peaks=args.refine_peaks,
                        include_trend=args.include_trend,
                        ridge_lam=args.ridge,
                        freq_weight=args.freq_weight,
                        huber_iters=args.huber_iters,
                        huber_delta=args.huber_delta,
                        min_obs=args.min_obs,
                        max_freqs=max_freqs,
                    )
                    params["valid"][i, j] = int(pixel["valid"])
                    params["n_freqs_used"][i, j] = int(pixel["n_freqs_used"])
                    params["t_min"][i, j] = float(pixel["t_min"])
                    params["t_rel_mean"][i, j] = float(pixel["t_rel_mean"])
                    params["fill_value"][i, j] = float(pixel["fill_value"])
                    params["freqs"][i, j, :] = np.asarray(pixel["freqs"], dtype=np.float64)
                    params["beta"][i, j, :] = np.asarray(pixel["beta"], dtype=np.float64)
                elif algorithm == "hants":
                    pixel = fit_hants_pixel_params(t_days, y, nof=3, sf="none", dod=1)
                    params["valid"][i, j] = int(pixel["valid"])
                    params["coeffs"][i, j, :] = np.asarray(pixel["coeffs"], dtype=np.float64)
                    params["fill_value"][i, j] = float(pixel["fill_value"])
                else:
                    pixel = fit_zhu2015_pixel_params(t_days, y, max_segments=max_segments)
                    params["valid"][i, j] = int(pixel["valid"])
                    params["n_segments"][i, j] = int(pixel["n_segments"])
                    params["segment_start_days"][i, j, :] = np.asarray(pixel["segment_start_days"], dtype=np.float64)
                    params["segment_end_days"][i, j, :] = np.asarray(pixel["segment_end_days"], dtype=np.float64)
                    params["segment_orders"][i, j, :] = np.asarray(pixel["segment_orders"], dtype=np.int16)
                    params["segment_has_model"][i, j, :] = np.asarray(pixel["segment_has_model"], dtype=np.int8)
                    params["segment_median_values"][i, j, :] = np.asarray(pixel["segment_median_values"], dtype=np.float64)
                    params["segment_intercepts"][i, j, :] = np.asarray(pixel["segment_intercepts"], dtype=np.float64)
                    params["segment_coefficients"][i, j, :, :] = np.asarray(pixel["segment_coefficients"], dtype=np.float64)

    return params


def fit_param_cube_from_source(
    source: TimeSeriesRasterSource,
    algorithm: str,
    args: Optional[Args] = None,
    block_shape: Tuple[int, int] = (64, 64),
    max_freqs: int = 10,
    max_segments: int = 10,
) -> Dict[str, Any]:
    meta = source.metadata()
    height = int(meta["height"])
    width = int(meta["width"])
    timestamps = np.asarray(meta["timestamps"])
    if args is None:
        args = build_args(algorithm.lower(), {})

    algorithm = algorithm.lower()
    t_sec = timestamps_to_seconds(timestamps, unit=args.time_unit)
    time_origin_sec = float(np.nanmin(t_sec))
    t_days = (t_sec - time_origin_sec) / 86400.0

    params: Dict[str, Any] = {
        "algorithm": np.array(algorithm),
        "shape": np.array([height, width], dtype=np.int32),
        "time_origin_sec": np.array(time_origin_sec, dtype=np.float64),
    }

    if algorithm == "nufrost":
        beta_size = 1 + (1 if args.include_trend else 0) + 2 * max_freqs
        params.update(
            {
                "valid": np.zeros((height, width), dtype=np.int8),
                "n_freqs_used": np.zeros((height, width), dtype=np.int16),
                "t_min": np.full((height, width), np.nan, dtype=np.float64),
                "t_rel_mean": np.full((height, width), np.nan, dtype=np.float64),
                "fill_value": np.full((height, width), np.nan, dtype=np.float64),
                "freqs": np.full((height, width, max_freqs), np.nan, dtype=np.float64),
                "beta": np.full((height, width, beta_size), np.nan, dtype=np.float64),
                "include_trend": np.array(bool(args.include_trend), dtype=np.int8),
            }
        )
    elif algorithm == "hants":
        coeff_count = 1 + 2 * (3 - 1)
        params.update(
            {
                "valid": np.zeros((height, width), dtype=np.int8),
                "coeffs": np.full((height, width, coeff_count), np.nan, dtype=np.float64),
                "fill_value": np.full((height, width), np.nan, dtype=np.float64),
                "nof": np.full((height, width), 3, dtype=np.int16),
                "period": np.full((height, width), 365.25, dtype=np.float64),
            }
        )
    elif algorithm == "zhu2015":
        params.update(
            {
                "valid": np.zeros((height, width), dtype=np.int8),
                "n_segments": np.zeros((height, width), dtype=np.int16),
                "segment_start_days": np.full((height, width, max_segments), np.nan, dtype=np.float64),
                "segment_end_days": np.full((height, width, max_segments), np.nan, dtype=np.float64),
                "segment_orders": np.zeros((height, width, max_segments), dtype=np.int16),
                "segment_has_model": np.zeros((height, width, max_segments), dtype=np.int8),
                "segment_median_values": np.full((height, width, max_segments), np.nan, dtype=np.float64),
                "segment_intercepts": np.zeros((height, width, max_segments), dtype=np.float64),
                "segment_coefficients": np.zeros((height, width, max_segments, 2 * 3 + 1), dtype=np.float64),
                "max_order": np.array(3, dtype=np.int16),
            }
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    for row_slice, col_slice in _block_slices(height, width, block_shape):
        for i in range(row_slice.start, row_slice.stop):
            for j in range(col_slice.start, col_slice.stop):
                y = source.read_pixel_series(i, j)
                if algorithm == "nufrost":
                    pixel = fit_nufrost_pixel_params(
                        t_sec,
                        y,
                        nufft_modes=args.modes,
                        eps=args.eps,
                        num_peaks=args.num_peaks,
                        power_cum=args.power_cum,
                        ignore_dc_hz=args.ignore_dc_hz,
                        frequency_selection=args.frequency_selection,
                        preferred_periods_days=args.preferred_periods_days,
                        preferred_top_k=args.preferred_top_k,
                        spectral_top_k=args.spectral_top_k,
                        spectral_merge_tol=args.spectral_merge_tol,
                        refine_peaks=args.refine_peaks,
                        include_trend=args.include_trend,
                        ridge_lam=args.ridge,
                        freq_weight=args.freq_weight,
                        huber_iters=args.huber_iters,
                        huber_delta=args.huber_delta,
                        min_obs=args.min_obs,
                        max_freqs=max_freqs,
                    )
                    params["valid"][i, j] = int(pixel["valid"])
                    params["n_freqs_used"][i, j] = int(pixel["n_freqs_used"])
                    params["t_min"][i, j] = float(pixel["t_min"])
                    params["t_rel_mean"][i, j] = float(pixel["t_rel_mean"])
                    params["fill_value"][i, j] = float(pixel["fill_value"])
                    params["freqs"][i, j, :] = np.asarray(pixel["freqs"], dtype=np.float64)
                    params["beta"][i, j, :] = np.asarray(pixel["beta"], dtype=np.float64)
                elif algorithm == "hants":
                    pixel = fit_hants_pixel_params(t_days, y, nof=3, sf="none", dod=1)
                    params["valid"][i, j] = int(pixel["valid"])
                    params["coeffs"][i, j, :] = np.asarray(pixel["coeffs"], dtype=np.float64)
                    params["fill_value"][i, j] = float(pixel["fill_value"])
                else:
                    pixel = fit_zhu2015_pixel_params(t_days, y, max_segments=max_segments)
                    params["valid"][i, j] = int(pixel["valid"])
                    params["n_segments"][i, j] = int(pixel["n_segments"])
                    params["segment_start_days"][i, j, :] = np.asarray(pixel["segment_start_days"], dtype=np.float64)
                    params["segment_end_days"][i, j, :] = np.asarray(pixel["segment_end_days"], dtype=np.float64)
                    params["segment_orders"][i, j, :] = np.asarray(pixel["segment_orders"], dtype=np.int16)
                    params["segment_has_model"][i, j, :] = np.asarray(pixel["segment_has_model"], dtype=np.int8)
                    params["segment_median_values"][i, j, :] = np.asarray(pixel["segment_median_values"], dtype=np.float64)
                    params["segment_intercepts"][i, j, :] = np.asarray(pixel["segment_intercepts"], dtype=np.float64)
                    params["segment_coefficients"][i, j, :, :] = np.asarray(pixel["segment_coefficients"], dtype=np.float64)

    return params


def predict_cube_from_params(params: Dict[str, Any], target_time: str, block_shape: Tuple[int, int] = (64, 64)) -> np.ndarray:
    algorithm = str(params["algorithm"])
    height, width = [int(v) for v in np.asarray(params["shape"]).tolist()]
    target_value = _target_time_to_algorithm_time(target_time, params)
    out = np.full((height, width), np.nan, dtype=np.float32)

    for row_slice, col_slice in _block_slices(height, width, block_shape):
        for i in range(row_slice.start, row_slice.stop):
            for j in range(col_slice.start, col_slice.stop):
                if algorithm == "nufrost":
                    pixel = {
                        "valid": bool(params["valid"][i, j]),
                        "include_trend": bool(params["include_trend"]),
                        "n_freqs_used": int(params["n_freqs_used"][i, j]),
                        "t_min": float(params["t_min"][i, j]),
                        "t_rel_mean": float(params["t_rel_mean"][i, j]),
                        "fill_value": float(params["fill_value"][i, j]),
                        "freqs": params["freqs"][i, j, :],
                        "beta": params["beta"][i, j, :],
                    }
                    out[i, j] = predict_nufrost_from_params(pixel, target_value)
                elif algorithm == "hants":
                    pixel = {
                        "valid": bool(params["valid"][i, j]),
                        "nof": int(params["nof"][i, j]),
                        "period": float(params["period"][i, j]),
                        "coeffs": params["coeffs"][i, j, :],
                        "fill_value": float(params["fill_value"][i, j]),
                    }
                    out[i, j] = predict_hants_from_params(pixel, target_value)
                else:
                    pixel = {
                        "valid": bool(params["valid"][i, j]),
                        "n_segments": int(params["n_segments"][i, j]),
                        "max_order": int(params["max_order"]),
                        "segment_start_days": params["segment_start_days"][i, j, :],
                        "segment_end_days": params["segment_end_days"][i, j, :],
                        "segment_orders": params["segment_orders"][i, j, :],
                        "segment_has_model": params["segment_has_model"][i, j, :],
                        "segment_median_values": params["segment_median_values"][i, j, :],
                        "segment_intercepts": params["segment_intercepts"][i, j, :],
                        "segment_coefficients": params["segment_coefficients"][i, j, :, :],
                    }
                    out[i, j] = predict_zhu2015_from_params(pixel, target_value)

    return out


def save_param_cube(path: Path | str, params: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: (value if isinstance(value, np.ndarray) else np.array(value)) for key, value in params.items()}
    np.savez_compressed(path, **arrays)


def load_param_cube(path: Path | str) -> Dict[str, Any]:
    loaded = np.load(Path(path), allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}
