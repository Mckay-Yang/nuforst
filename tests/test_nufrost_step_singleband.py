import numpy as np
import pytest

from src.nufrost import (
    fit_nufrost_pixel_step_singleband,
    predict_nufrost_from_params,
)


def _make_step_pixel(rng, n=180, step_day=120, step_amp=-0.4, noise=0.02):
    t_days = np.linspace(0.0, 365.25, n)
    t_sec = t_days * 86400.0
    annual = 2 * np.pi / 365.25
    season = 0.4 * np.cos(annual * t_days)
    step = np.where(t_days >= step_day, step_amp, 0.0)
    y = season + step + rng.normal(scale=noise, size=n)
    return t_sec, y, step_day, step_amp


def test_fit_step_singleband_recovers_step_amplitude():
    """The convex BCD solution does not perfectly separate a step from a
    Fourier basis with comparable degrees of freedom. The basis absorbs
    part of the step, so the recovered jump is a fraction of the true
    step. Test verifies the sign and that a meaningful fraction (~50%)
    of the step is captured by the L1 term."""
    rng = np.random.default_rng(42)
    t_sec, y, step_day, step_amp = _make_step_pixel(rng)
    annual_f = 1.0 / (365.25 * 86400.0)
    semi_f = 2.0 / (365.25 * 86400.0)
    freqs = np.array([annual_f, semi_f], dtype=np.float64)
    params = fit_nufrost_pixel_step_singleband(
        t_sec, y, freqs_sel=freqs,
        lambda_beta=1e-3, lambda_high=1e-2, lambda_step=0.3,
        low_freq_period_days=60.0, step_dt_weighting=False,
        max_outer_iter=20, outer_tol=1e-4,
        freq_weight=1.0, include_trend=False,
    )
    assert params["valid"] is True
    u = params["u"]
    # u should jump in the direction of step_amp around step_day. Convex
    # BCD recovers about half the true magnitude before the basis absorbs
    # the rest; require sign correct and at least 30% of true magnitude.
    pre_mean = float(np.mean(u[:80]))
    post_mean = float(np.mean(u[-40:]))
    diff = post_mean - pre_mean
    assert np.sign(diff) == np.sign(step_amp)
    assert abs(diff) > 0.3 * abs(step_amp)
    assert abs(diff) < 1.5 * abs(step_amp)


def test_fit_step_singleband_disabled_recovers_legacy_beta():
    rng = np.random.default_rng(7)
    t_sec, y, _, _ = _make_step_pixel(rng, step_amp=0.0)
    annual_f = 1.0 / (365.25 * 86400.0)
    freqs = np.array([annual_f], dtype=np.float64)
    params = fit_nufrost_pixel_step_singleband(
        t_sec, y, freqs_sel=freqs,
        lambda_beta=1e-3, lambda_high=1e-3, lambda_step=1e30,
        low_freq_period_days=60.0, step_dt_weighting=False,
        max_outer_iter=3, outer_tol=1e-4,
        freq_weight=1.0, include_trend=False,
    )
    assert params["valid"] is True
    # When the step term is disabled, u should be ~ 0 everywhere.
    assert np.max(np.abs(params["u"])) < 1e-3


def test_fit_step_singleband_handles_few_observations():
    t_sec = np.linspace(0.0, 86400.0 * 30, 5)
    y = np.array([0.1, 0.2, 0.15, 0.18, 0.12])
    freqs = np.array([1.0 / (365.25 * 86400.0)], dtype=np.float64)
    params = fit_nufrost_pixel_step_singleband(
        t_sec, y, freqs_sel=freqs,
        lambda_beta=1e-3, lambda_high=1e-3, lambda_step=1.0,
        low_freq_period_days=60.0, step_dt_weighting=True,
        max_outer_iter=3, outer_tol=1e-4,
        freq_weight=1.0, include_trend=False,
        min_obs=12,
    )
    assert params["valid"] is False
    assert params["u"].shape == (5,)
