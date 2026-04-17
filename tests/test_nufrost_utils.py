import numpy as np

from src.nufrost import (
    _parse_preferred_periods_days,
    _preferred_periods_to_freqs,
    _snap_frequency_to_spectrum,
    design_matrix,
    huber_weights,
    next_even,
    parse_timestamp_str,
    select_frequencies,
    timestamps_to_seconds,
)


def test_timestamps_to_seconds_supports_multiple_input_types() -> None:
    timestamps = np.array(["2020-01-01T00:00:00", np.datetime64("2020-01-02T00:00:00")], dtype=object)
    seconds = timestamps_to_seconds(timestamps)
    assert np.isfinite(seconds).all()
    assert np.isclose(seconds[1] - seconds[0], 86400.0)


def test_parse_timestamp_str_returns_none_for_invalid_input() -> None:
    assert parse_timestamp_str("not-a-timestamp") is None


def test_next_even_rounds_up() -> None:
    assert next_even(4) == 4
    assert next_even(5) == 6


def test_preferred_period_helpers_filter_invalid_values() -> None:
    periods = _parse_preferred_periods_days("365.25, 0, -1, 30")
    assert np.allclose(periods, np.array([365.25, 30.0]))

    freqs_days = _preferred_periods_to_freqs([365.25, 30.0], time_unit="days")
    assert np.allclose(freqs_days, np.array([1.0 / 365.25, 1.0 / 30.0]))


def test_snap_frequency_to_spectrum_prefers_strong_nearby_peak() -> None:
    f_pos = np.array([0.01, 0.02, 0.03])
    p_pos = np.array([1.0, 10.0, 2.0])
    snapped = _snap_frequency_to_spectrum(0.019, f_pos, p_pos, rel_tol=0.1)
    assert np.isclose(snapped, 0.02)


def test_design_matrix_shape_matches_requested_terms() -> None:
    t = np.array([0.0, 1.0, 2.0])
    X = design_matrix(t, [0.25], include_trend=True, include_dc=True)
    assert X.shape == (3, 4)
    assert np.allclose(X[:, 0], 1.0)


def test_huber_weights_downweight_large_residuals() -> None:
    weights = huber_weights(np.array([0.1, 5.0]), delta=1.0)
    assert np.isclose(weights[0], 1.0)
    assert weights[1] < 1.0


def test_select_frequencies_merges_preferred_and_spectral_candidates() -> None:
    f_pos = np.array([0.0, 0.01, 0.02, 0.04])
    p_pos = np.array([0.1, 4.0, 10.0, 2.0])
    freqs = select_frequencies(
        f_pos=f_pos,
        P_pos=p_pos,
        fmax=0.05,
        selection_mode="hybrid",
        preferred_freqs=np.array([0.0195]),
        preferred_top_k=1,
        spectral_top_k=2,
        spectral_merge_tol=0.1,
        power_cum=1.0,
        ignore_dc_hz=1e-9,
        refine_peaks=False,
    )
    assert freqs.size >= 1
    assert np.any(np.isclose(freqs, 0.02, atol=1e-6))
