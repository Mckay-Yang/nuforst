import numpy as np

from config import build_args
from src.data_loader import TimeSeriesRasterSource
from src.hants import fit_hants_pixel_params, hants_pixel, predict_hants_from_params
from src.model_params import (
    fit_cube_params,
    fit_param_cube_from_source,
    load_param_cube,
    predict_cube_from_params,
    save_param_cube,
)
from src.nufrost import fit_nufrost_pixel_params, predict_nufrost_from_params, predict_single_pixel
from src.zhu2015 import fit_predict_pixel, fit_zhu2015_pixel_params, predict_zhu2015_from_params


def test_nufrost_pixel_params_reproduce_direct_prediction(
    synthetic_t_sec: np.ndarray,
    synthetic_signal: np.ndarray,
) -> None:
    target_t = float(synthetic_t_sec[5])
    y = synthetic_signal.copy()
    y[5] = np.nan

    direct_pred, direct_n_freqs = predict_single_pixel(
        synthetic_t_sec,
        y,
        target_t,
        nufft_modes=64,
        eps=1e-12,
        num_peaks=4,
        power_cum=0.9,
        ignore_dc_hz=1e-9,
        min_obs=6,
    )
    params = fit_nufrost_pixel_params(
        synthetic_t_sec,
        y,
        nufft_modes=64,
        eps=1e-12,
        num_peaks=4,
        power_cum=0.9,
        ignore_dc_hz=1e-9,
        min_obs=6,
        max_freqs=10,
    )
    param_pred = predict_nufrost_from_params(params, target_t)

    assert np.isfinite(param_pred)
    assert abs(param_pred - direct_pred) < 1e-6
    assert int(params["n_freqs_used"]) == direct_n_freqs


def test_hants_pixel_params_reproduce_direct_prediction(
    synthetic_t_days: np.ndarray,
    synthetic_signal: np.ndarray,
) -> None:
    target_t = float(synthetic_t_days[4])

    direct_pred = hants_pixel(synthetic_t_days, synthetic_signal, target_t=target_t, nof=3, sf="none", dod=1)
    params = fit_hants_pixel_params(synthetic_t_days, synthetic_signal, nof=3, sf="none", dod=1)
    param_pred = predict_hants_from_params(params, target_t)

    assert np.isfinite(param_pred)
    assert abs(param_pred - direct_pred) < 1e-6


def test_zhu2015_pixel_params_reproduce_direct_prediction(
    synthetic_t_days: np.ndarray,
    synthetic_signal: np.ndarray,
) -> None:
    target_t = float(synthetic_t_days[3])

    direct_pred, direct_qa = fit_predict_pixel(synthetic_t_days, synthetic_signal, target_t_day=target_t)
    params = fit_zhu2015_pixel_params(synthetic_t_days, synthetic_signal, max_segments=10)
    param_pred, param_qa = predict_zhu2015_from_params(params, target_t)

    assert np.isfinite(param_pred)
    assert abs(param_pred - direct_pred) < 1e-6
    assert param_qa == direct_qa


def test_cube_param_roundtrip_reconstructs_complete_image(tmp_path) -> None:
    timestamps = np.array([
        "2020-01-01T00:00:00",
        "2020-02-01T00:00:00",
        "2020-03-01T00:00:00",
        "2020-04-01T00:00:00",
        "2020-05-01T00:00:00",
        "2020-06-01T00:00:00",
        "2020-07-01T00:00:00",
        "2020-08-01T00:00:00",
        "2020-09-01T00:00:00",
        "2020-10-01T00:00:00",
        "2020-11-01T00:00:00",
        "2020-12-01T00:00:00",
    ])
    base = np.linspace(0.1, 0.8, len(timestamps), dtype=np.float32)
    cube = np.stack(
        [
            np.array([[val, val + 0.02], [val + 0.03, val + 0.05]], dtype=np.float32)
            for val in base
        ],
        axis=0,
    )
    cube[2, 0, 0] = np.nan

    args = build_args({"n_jobs": 1, "show_progress": False, "min_obs": 6, "modes": 64, "num_peaks": 4})
    params = fit_cube_params(cube, timestamps, algorithm="nufrost", args=args, max_freqs=10)

    path = tmp_path / "nufrost_params.npz"
    save_param_cube(path, params)
    loaded = load_param_cube(path)
    out = predict_cube_from_params(loaded, "2020-06-15T00:00:00")

    assert out.shape == (2, 2)
    assert np.isfinite(out).any()


def test_param_cube_from_streaming_source_avoids_npz(single_tile_path: str, cache_dir, tmp_path) -> None:
    args = build_args({"cache_dir": cache_dir, "n_jobs": 1, "show_progress": False, "min_obs": 6, "modes": 64, "num_peaks": 4})

    with TimeSeriesRasterSource([single_tile_path], cache_dir=cache_dir) as source:
        params = fit_param_cube_from_source(source, algorithm="nufrost", args=args, max_freqs=10)

    path = tmp_path / "streaming_nufrost_params.npz"
    save_param_cube(path, params)
    loaded = load_param_cube(path)
    out = predict_cube_from_params(loaded, "2020-06-15T00:00:00")

    assert out.ndim == 2
    assert np.isfinite(out).any()
    assert not any((cache_dir / "npz").glob("*.npz"))
