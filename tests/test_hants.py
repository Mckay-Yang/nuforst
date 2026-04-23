import numpy as np

from src.hants import _apply_hants_valid_mask, hants_curve_pixel, hants_pixel, make_harmonic_matrix


def test_apply_hants_valid_mask_respects_idrt_direction() -> None:
    y = np.array([0.1, 0.5, 0.9, np.nan])
    low_mask = _apply_hants_valid_mask(y, sf="low", idrt=0.7)
    high_mask = _apply_hants_valid_mask(y, sf="high", idrt=0.3)

    assert low_mask.tolist() == [True, True, False, False]
    assert high_mask.tolist() == [False, True, True, False]


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
