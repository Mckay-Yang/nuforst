import numpy as np
import pytest

from src.nufrost import _classify_freq_tier


def _period_to_freq_per_sec(period_days):
    return 1.0 / (period_days * 86400.0)


def test_classify_freq_tier_marks_short_periods_as_high():
    freqs = np.array([
        _period_to_freq_per_sec(365.25),  # annual, low
        _period_to_freq_per_sec(182.625), # semiannual, low
        _period_to_freq_per_sec(60.0),    # exact threshold, low
        _period_to_freq_per_sec(45.0),    # high
        _period_to_freq_per_sec(15.0),    # high
    ], dtype=np.float64)
    is_high = _classify_freq_tier(freqs, low_freq_period_days=60.0, time_unit_seconds=True)
    assert is_high.tolist() == [False, False, False, True, True]


def test_classify_freq_tier_handles_empty_input():
    is_high = _classify_freq_tier(np.zeros(0), low_freq_period_days=60.0, time_unit_seconds=True)
    assert is_high.shape == (0,)
    assert is_high.dtype == np.bool_


def test_classify_freq_tier_zero_freq_treated_as_low():
    freqs = np.array([0.0, _period_to_freq_per_sec(30.0)], dtype=np.float64)
    is_high = _classify_freq_tier(freqs, low_freq_period_days=60.0, time_unit_seconds=True)
    assert is_high.tolist() == [False, True]


from src.nufrost import _tiered_ridge_solve, design_matrix


def _make_synthetic_pixel(rng, n=120, low_amp=0.5, high_amp=0.05, noise=0.02):
    t = np.linspace(0.0, 3.0 * 365.25 * 86400.0, n)
    annual = 2 * np.pi / (365.25 * 86400.0)
    short = 2 * np.pi / (15.0 * 86400.0)
    y = low_amp * np.cos(annual * t) + high_amp * np.cos(short * t)
    y = y + rng.normal(scale=noise, size=n)
    return t, y


def test_tiered_ridge_shrinks_high_freq_more_than_low():
    rng = np.random.default_rng(0)
    t, y = _make_synthetic_pixel(rng)
    annual_f = 1.0 / (365.25 * 86400.0)
    short_f = 1.0 / (15.0 * 86400.0)
    freqs = np.array([annual_f, short_f], dtype=np.float64)
    X = design_matrix(t - t.min(), freqs, include_trend=False, include_dc=True)
    beta_equal = _tiered_ridge_solve(X, y, freqs,
                                     lambda_beta=1e-3, lambda_high=1e-3,
                                     low_freq_period_days=60.0,
                                     freq_weight=1.0,
                                     include_dc=True, include_trend=False)
    # Use a sufficiently large lambda_high so the additive penalty on
    # high-tier cos/sin columns dominates the corresponding XᵀX diagonal
    # (~n/2 ≈ 60 for n=120), producing the expected >50% shrinkage.
    beta_tier = _tiered_ridge_solve(X, y, freqs,
                                    lambda_beta=1e-3, lambda_high=100.0,
                                    low_freq_period_days=60.0,
                                    freq_weight=1.0,
                                    include_dc=True, include_trend=False)
    # Cosine amplitude for the short-period frequency lives in column 3
    # (DC, cos(annual), sin(annual), cos(short), sin(short))
    short_amp_equal = np.hypot(beta_equal[3], beta_equal[4])
    short_amp_tier = np.hypot(beta_tier[3], beta_tier[4])
    annual_amp_equal = np.hypot(beta_equal[1], beta_equal[2])
    annual_amp_tier = np.hypot(beta_tier[1], beta_tier[2])
    assert short_amp_tier < 0.5 * short_amp_equal
    # Annual amplitude must be largely preserved
    assert annual_amp_tier > 0.7 * annual_amp_equal


def test_tiered_ridge_close_to_legacy_when_lambda_equal_and_small():
    """When λ_high == λ_β at a non-trivial ridge, the new solver should
    agree with the legacy ridge on frequency coefficients (legacy and
    tiered apply the same W_freq weighting there). The DC coefficient
    differs by at most O(λ_β) because the new solver penalizes DC with
    1.0² where legacy uses 0.

    We use lam=1e-2 (large enough that the ridge actually bites on a
    signal of unit-amplitude scale) so that this test would catch a
    wrongly-perturbed freq penalty (e.g. swapping cos(annual)'s row of
    W_freq) — verified by perturbation."""
    rng = np.random.default_rng(1)
    t, y = _make_synthetic_pixel(rng)
    annual_f = 1.0 / (365.25 * 86400.0)
    short_f = 1.0 / (15.0 * 86400.0)
    freqs = np.array([annual_f, short_f], dtype=np.float64)
    X = design_matrix(t - t.min(), freqs, include_trend=False, include_dc=True)
    from src.nufrost import ridge_with_freq_weights
    lam = 1e-2
    beta_legacy, _ = ridge_with_freq_weights(
        X, y, freqs, lam=lam, include_dc=True, include_trend=False, freq_weight=1.0
    )
    beta_tier = _tiered_ridge_solve(
        X, y, freqs, lambda_beta=lam, lambda_high=lam,
        low_freq_period_days=60.0, freq_weight=1.0,
        include_dc=True, include_trend=False,
    )
    # DC differs by O(λ_β · |β_DC|); for this signal |β_DC| ~ O(1) so a
    # 0.05 tolerance is loose enough to pass yet tight enough to catch a
    # gross perturbation of the freq block.
    assert abs(beta_legacy[0] - beta_tier[0]) < 0.05
    # Frequency coefficients should match to within numerical noise.
    assert np.allclose(beta_legacy[1:], beta_tier[1:], atol=1e-3)
