import numpy as np

from src.zhu2015 import (
    _select_model_order,
    _select_unit_qa,
    extract_segments,
    fit_predict_pixel,
    make_design_matrix,
)


def test_select_model_order_thresholds() -> None:
    assert _select_model_order(5) == 0
    assert _select_model_order(10) == 1
    assert _select_model_order(20) == 2
    assert _select_model_order(30) == 3


def test_select_unit_qa_thresholds() -> None:
    assert _select_unit_qa(15) == 0
    assert _select_unit_qa(8) == 1
    assert _select_unit_qa(3) == 2
    assert _select_unit_qa(20, perennial_snow=True) == 3


def test_make_design_matrix_for_order_two() -> None:
    x = np.array([0.0, 10.0, 20.0])
    X = make_design_matrix(x, order=2)
    assert X.shape == (3, 5)


def test_extract_segments_short_series_uses_single_segment() -> None:
    t_days = np.arange(8, dtype=np.float64)
    y = np.linspace(0.2, 0.4, 8, dtype=np.float64)
    segments = extract_segments(t_days, y)
    assert len(segments) == 1
    assert segments[0]["start_idx"] == 0
    assert segments[0]["end_idx"] == 7


def test_fit_predict_pixel_returns_finite_prediction(
    synthetic_t_days: np.ndarray,
    synthetic_signal: np.ndarray,
) -> None:
    pred, qa = fit_predict_pixel(synthetic_t_days, synthetic_signal, target_t_day=float(synthetic_t_days[3]))
    assert np.isfinite(pred)
    assert 0 <= qa <= 255
