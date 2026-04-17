import numpy as np

from config import build_args
from src.nufrost import nufrost_core, predict_single_pixel


def test_predict_single_pixel_returns_finite_on_clean_periodic_signal(
    synthetic_t_days: np.ndarray,
    synthetic_t_sec: np.ndarray,
    synthetic_signal: np.ndarray,
) -> None:
    target_t = synthetic_t_sec[5]
    y = synthetic_signal.copy()
    true_value = y[5]
    y[5] = np.nan

    pred, n_freqs = predict_single_pixel(
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

    assert np.isfinite(pred)
    assert n_freqs >= 1
    assert abs(pred - true_value) < 0.05


def test_nufrost_core_returns_image_for_small_cube() -> None:
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

    args = build_args(
        {
            "n_jobs": 1,
            "show_progress": False,
            "min_obs": 6,
            "modes": 64,
            "num_peaks": 4,
        }
    )
    out = nufrost_core(cube, timestamps, "2020-06-15T00:00:00", args=args)

    assert out.shape == (2, 2)
    assert np.isfinite(out).any()
