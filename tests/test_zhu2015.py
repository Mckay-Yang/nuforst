import numpy as np

from src.zhu2015 import (
    _select_model_order,
    extract_segments,
    fit_predict_pixel,
    fit_predict_pixel_segments,
    fit_zhu2015_pixel_params,
    make_design_matrix,
    predict_zhu2015_from_params,
)


def test_select_model_order_thresholds() -> None:
    assert _select_model_order(5) == 0
    assert _select_model_order(10) == 1
    assert _select_model_order(20) == 2
    assert _select_model_order(30) == 3


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


def test_extract_segments_does_not_emit_qa_keys() -> None:
    t_days = np.arange(8, dtype=np.float64)
    y = np.linspace(0.2, 0.4, 8, dtype=np.float64)
    segments = extract_segments(t_days, y)
    for seg in segments:
        assert "unit_qa" not in seg


def test_fit_predict_pixel_returns_finite_prediction(
    synthetic_t_days: np.ndarray,
    synthetic_signal: np.ndarray,
) -> None:
    pred = fit_predict_pixel(synthetic_t_days, synthetic_signal, target_t_day=float(synthetic_t_days[3]))
    assert np.isfinite(pred)
    assert isinstance(pred, float)


def test_select_model_order_paper_thresholds() -> None:
    assert _select_model_order(5) == 0
    assert _select_model_order(6) == 1
    assert _select_model_order(11) == 1
    assert _select_model_order(12) == 1
    assert _select_model_order(17) == 1
    assert _select_model_order(18) == 2
    assert _select_model_order(23) == 2
    assert _select_model_order(24) == 3


def test_predict_zhu2015_from_params_returns_scalar() -> None:
    np.random.seed(42)
    t_days = np.arange(30, dtype=np.float64) * 16
    y = np.sin(2 * np.pi * t_days / 365.25) * 0.1 + 0.5

    params = fit_zhu2015_pixel_params(t_days, y)
    pred = predict_zhu2015_from_params(params, target_t_day=float(t_days[0]))

    assert params["valid"] is True
    assert "segment_unit_qas" not in params
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_predict_zhu2015_backward_projection_returns_finite() -> None:
    np.random.seed(42)
    t_days = np.arange(30, dtype=np.float64) * 16
    y = np.sin(2 * np.pi * t_days / 365.25) * 0.1 + 0.5

    params = fit_zhu2015_pixel_params(t_days, y)
    pred = predict_zhu2015_from_params(params, target_t_day=float(t_days[0] - 100))

    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_predict_zhu2015_forward_projection_returns_finite() -> None:
    np.random.seed(42)
    t_days = np.arange(30, dtype=np.float64) * 16
    y = np.sin(2 * np.pi * t_days / 365.25) * 0.1 + 0.5

    params = fit_zhu2015_pixel_params(t_days, y)
    pred = predict_zhu2015_from_params(params, target_t_day=float(t_days[-1] + 100))

    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_break_detection_requires_six_consecutive() -> None:
    np.random.seed(42)
    t_days = np.arange(60, dtype=np.float64)
    y = np.ones(60, dtype=np.float64) * 0.5
    y[30:36] = 10.0

    segments = extract_segments(t_days, y)
    assert len(segments) == 2


def test_break_detection_five_consecutive_no_break() -> None:
    np.random.seed(42)
    t_days = np.arange(60, dtype=np.float64)
    y = np.ones(60, dtype=np.float64) * 0.5
    y[30:35] = 10.0

    segments = extract_segments(t_days, y)
    assert len(segments) == 1


def test_prediction_is_finite_for_far_future() -> None:
    np.random.seed(42)
    t_days = np.arange(50, dtype=np.float64) * 16
    y_mean = 0.02
    y = y_mean + np.random.randn(50).astype(np.float64) * 0.005
    y += -0.0005 * t_days

    params = fit_zhu2015_pixel_params(t_days, y)
    assert params["valid"] is True

    future_t = float(t_days[-1] + 5000)
    pred = predict_zhu2015_from_params(params, target_t_day=future_t)
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_fit_predict_pixel_segments_detects_break() -> None:
    np.random.seed(42)
    t_days = np.arange(60, dtype=np.float64)
    y = np.ones(60, dtype=np.float64) * 0.5
    y[30:36] = 10.0

    pred = fit_predict_pixel_segments(t_days, y, target_t_day=float(t_days[50]))
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_fit_predict_pixel_segments_short_series() -> None:
    np.random.seed(42)
    t_days = np.arange(8, dtype=np.float64)
    y = np.linspace(0.2, 0.4, 8, dtype=np.float64)

    pred = fit_predict_pixel_segments(t_days, y, target_t_day=float(t_days[4]))
    assert isinstance(pred, float)
    assert np.isfinite(pred)
