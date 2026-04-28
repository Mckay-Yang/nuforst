import numpy as np

from src.hants import (
    _apply_hants_valid_mask,
    fit_hants_pixel_params,
    hants_curve_pixel,
    hants_pixel,
    make_harmonic_matrix,
    predict_hants_from_params,
)


def test_apply_hants_valid_mask_lower_upper() -> None:
    y = np.array([0.1, 0.5, 0.9, np.nan])
    lower_mask = _apply_hants_valid_mask(y, valid_min=0.7)
    upper_mask = _apply_hants_valid_mask(y, valid_max=0.3)

    assert lower_mask.tolist() == [False, False, True, False]
    assert upper_mask.tolist() == [True, False, False, False]


def test_make_harmonic_matrix_shape() -> None:
    X = make_harmonic_matrix(np.array([0.0, 1.0]), [1 / 365.25, 2 / 365.25])
    assert X.shape == (2, 5)


def test_hants_pixel_and_curve_return_finite_values(
    synthetic_t_days: np.ndarray,
    synthetic_signal: np.ndarray,
) -> None:
    pred = hants_pixel(synthetic_t_days, synthetic_signal, target_t=float(synthetic_t_days[4]), nof=3, sf="none", dod=1)
    curve = hants_curve_pixel(synthetic_t_days, synthetic_signal, synthetic_t_days[:3], nof=3, sf="none", dod=1)

    assert np.isfinite(pred)
    assert curve.shape == (3,)
    assert np.isfinite(curve).all()


def test_fit_hants_pixel_params_requires_paper_minimum_observations_after_rejection() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)

    params = fit_hants_pixel_params(t, y, nof=1, sf="none", fet=0.1, dod=3, period=365.25)

    assert params["valid"] is False


def test_hants_suppression_flag_is_one_sided() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0, 1.0, 1.0, 10.0], dtype=np.float64)

    low_pred = hants_pixel(t, y, target_t=1.5, nof=1, sf="low", fet=0.1, dod=0, period=365.25)
    high_pred = hants_pixel(t, y, target_t=1.5, nof=1, sf="high", fet=0.1, dod=0, period=365.25)

    assert low_pred > 2.0
    assert abs(high_pred - 1.0) < 1e-6


def test_valid_min_filters_invalid_low_before_hants_fit() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0.2, 0.4, 0.8, 0.9], dtype=np.float64)

    params = fit_hants_pixel_params(t, y, nof=1, sf="low", valid_min=0.7, fet=0.1, dod=0, period=365.25)
    pred = predict_hants_from_params(params, target_t=1.5)

    assert params["valid"] is True
    assert abs(pred - 0.85) < 1e-6


def test_hants_iterative_rejection_removes_outliers() -> None:
    t = np.arange(10, dtype=np.float64)
    y = np.ones(10, dtype=np.float64) * 0.5
    y[5] = -5.0

    params = fit_hants_pixel_params(t, y, nof=1, sf="low", fet=0.1, dod=0, period=365.25)

    assert params["valid"] is True
    pred = predict_hants_from_params(params, target_t=5.0)
    assert abs(pred - 0.5) < 0.5


def test_hants_nof_3_uses_5_parameters() -> None:
    params = fit_hants_pixel_params(
        np.arange(20, dtype=np.float64),
        np.ones(20, dtype=np.float64),
        nof=3,
        sf="low",
        fet=0.1,
        dod=5,
        period=365.25,
    )
    assert params["valid"] is True
    assert len(params["coeffs"]) == 5


def test_hants_stops_when_all_residuals_within_fet() -> None:
    t = np.arange(8, dtype=np.float64)
    y = np.sin(2 * np.pi * t / 365.25) * 0.1 + 0.5

    params = fit_hants_pixel_params(t, y, nof=1, sf="low", fet=1.0, dod=0, period=365.25)

    assert params["valid"] is True


def test_hants_iterative_rejection_handles_multiple_outliers() -> None:
    t = np.arange(100, dtype=np.float64)
    y = np.ones(100, dtype=np.float64) * 0.5
    y[20] = -10.0
    y[40] = -10.0
    y[60] = -10.0

    params = fit_hants_pixel_params(t, y, nof=1, sf="low", fet=0.1, dod=0, period=365.25)

    assert params["valid"] is True
    pred = predict_hants_from_params(params, target_t=50.0)
    assert abs(pred - 0.5) < 0.5
    assert int(params.get("n_iterations", 100)) >= 1


def test_hants_converges_with_iteration_cap() -> None:
    t = np.arange(200, dtype=np.float64)
    y = np.ones(200, dtype=np.float64) * 0.5
    y[::7] = -5.0

    params = fit_hants_pixel_params(t, y, nof=1, sf="low", fet=0.1, dod=0, period=365.25)
    assert params["valid"] is True
    n_iters = int(params.get("n_iterations", 0))
    assert n_iters >= 1
    pred = predict_hants_from_params(params, target_t=100.0)
    assert abs(pred - 0.5) < 0.5
