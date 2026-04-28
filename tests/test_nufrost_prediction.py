import numpy as np

from config import build_args
from src.nufrost import fit_nufrost_pixel_params, nufrost_core, predict_single_pixel


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

    args = build_args("nufrost",
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


def test_preferred_frequency_selection_uses_requested_period() -> None:
    t_sec = np.arange(0, 4 * 365, 30, dtype=np.float64) * 86400.0
    y = np.ones_like(t_sec, dtype=np.float64)

    params = fit_nufrost_pixel_params(
        t_sec,
        y,
        nufft_modes=128,
        eps=1e-12,
        frequency_selection="preferred",
        preferred_periods_days="365.25",
        preferred_top_k=1,
        spectral_top_k=0,
        min_obs=6,
        max_freqs=1,
    )

    assert params["valid"] is True
    assert int(params["n_freqs_used"]) == 1
    assert abs(float(params["freqs"][0]) - 1.0 / (365.25 * 86400.0)) < 1e-9


def test_fit_nufrost_pixel_params_uses_shared_frequencies_when_provided() -> None:
    t_sec = np.arange(0, 4 * 365, 30, dtype=np.float64) * 86400.0
    y = 1000.0 + 50.0 * np.sin(2 * np.pi * t_sec / (365.25 * 86400.0))
    shared_freqs = np.array([1.0 / (120.0 * 86400.0), 1.0 / (240.0 * 86400.0)], dtype=np.float64)

    params = fit_nufrost_pixel_params(
        t_sec,
        y,
        nufft_modes=128,
        eps=1e-12,
        frequency_selection="preferred",
        preferred_periods_days="365.25",
        preferred_top_k=1,
        spectral_top_k=0,
        min_obs=6,
        max_freqs=2,
        shared_freqs=shared_freqs,
    )

    assert params["valid"] is True
    assert int(params["n_freqs_used"]) == 2
    np.testing.assert_allclose(params["freqs"][:2], shared_freqs)


def test_frequency_penalty_uses_log_growth() -> None:
    from src.nufrost import _make_frequency_penalty

    freqs = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    penalty = _make_frequency_penalty(freqs, freq_weight=2.0)

    np.testing.assert_allclose(penalty, np.sqrt([1.0, 3.0, 5.0]))
