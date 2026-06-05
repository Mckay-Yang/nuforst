import numpy as np
import pytest

from src.nufrost import fit_nufrost_pixel_multiband


def _make_multiband_pixel(rng, n=180, B=4, step_day=120,
                          step_amps=(-0.4, 0.3, -0.2, 0.5),
                          season_amp=0.4, noise=0.02):
    t_days = np.linspace(0.0, 365.25, n)
    t_sec = t_days * 86400.0
    annual = 2 * np.pi / 365.25
    season = season_amp * np.cos(annual * t_days)
    Y = np.zeros((n, B))
    for b in range(B):
        step = np.where(t_days >= step_day, step_amps[b], 0.0)
        Y[:, b] = season + step + rng.normal(scale=noise, size=n)
    return t_sec, Y


def test_fit_multiband_recovers_shared_step():
    rng = np.random.default_rng(11)
    t_sec, Y = _make_multiband_pixel(rng)
    annual_f = 1.0 / (365.25 * 86400.0)
    semi_f = 2.0 / (365.25 * 86400.0)
    freqs = np.array([annual_f, semi_f], dtype=np.float64)
    res = fit_nufrost_pixel_multiband(
        t_sec, Y, freqs_sel=freqs,
        lambda_beta=1e-3, lambda_high=1e-2, lambda_step=0.5,
        low_freq_period_days=60.0, step_dt_weighting=False,
        joint_outlier=False, joint_outlier_sigma=2.5,
        max_outer_iter=10, outer_tol=1e-4,
        admm_rho=1.0, admm_max_iter=200, admm_tol=1e-6,
        freq_weight=1.0, include_trend=False,
    )
    assert res["valid"]
    U = res["u"]                           # (n, B)
    assert U.shape == Y.shape
    pre = U[:80].mean(axis=0)
    post = U[-40:].mean(axis=0)
    diff = post - pre
    truth = np.array([-0.4, 0.3, -0.2, 0.5])
    for b in range(4):
        assert np.sign(diff[b]) == np.sign(truth[b]), f"band {b} sign wrong: {diff[b]} vs {truth[b]}"
        assert abs(diff[b]) > 0.3 * abs(truth[b]), f"band {b} too small: {diff[b]}"


def test_fit_multiband_disabled_step_returns_zero_u():
    rng = np.random.default_rng(13)
    t_sec, Y = _make_multiband_pixel(rng, step_amps=(0.0, 0.0, 0.0, 0.0))
    annual_f = 1.0 / (365.25 * 86400.0)
    freqs = np.array([annual_f], dtype=np.float64)
    res = fit_nufrost_pixel_multiband(
        t_sec, Y, freqs_sel=freqs,
        lambda_beta=1e-3, lambda_high=1e-3, lambda_step=1e30,
        low_freq_period_days=60.0, step_dt_weighting=False,
        joint_outlier=False, joint_outlier_sigma=2.5,
        max_outer_iter=3, outer_tol=1e-4,
        admm_rho=1.0, admm_max_iter=80, admm_tol=1e-4,
        freq_weight=1.0, include_trend=False,
    )
    assert res["valid"]
    assert np.max(np.abs(res["u"])) < 1e-3


def test_fit_multiband_joint_outlier_drops_correlated_clouds():
    """Inject a multi-band cloud event; verify the joint outlier mask
    excludes that timestep from the fit."""
    rng = np.random.default_rng(17)
    t_sec, Y = _make_multiband_pixel(rng)
    Y[50, :] += 5.0  # cloud-like spike across all bands
    annual_f = 1.0 / (365.25 * 86400.0)
    freqs = np.array([annual_f], dtype=np.float64)
    res = fit_nufrost_pixel_multiband(
        t_sec, Y, freqs_sel=freqs,
        lambda_beta=1e-3, lambda_high=1e-3, lambda_step=0.3,
        low_freq_period_days=60.0, step_dt_weighting=False,
        joint_outlier=True, joint_outlier_sigma=2.5,
        max_outer_iter=5, outer_tol=1e-4,
        admm_rho=1.0, admm_max_iter=200, admm_tol=1e-6,
        freq_weight=1.0, include_trend=False,
    )
    assert res["valid"]
    mask = res["mask"]
    assert mask.shape == (180,)
    assert not mask[50], "cloud index should be masked out"
    # Most clean indices kept
    assert mask.sum() > 160


def test_fit_multiband_handles_few_observations():
    t_sec = np.linspace(0.0, 86400.0 * 30, 5)
    Y = np.array([[0.1, 0.2], [0.2, 0.3], [0.15, 0.18],
                  [0.18, 0.25], [0.12, 0.2]])
    freqs = np.array([1.0 / (365.25 * 86400.0)], dtype=np.float64)
    res = fit_nufrost_pixel_multiband(
        t_sec, Y, freqs_sel=freqs,
        lambda_beta=1e-3, lambda_high=1e-3, lambda_step=1.0,
        low_freq_period_days=60.0, step_dt_weighting=True,
        joint_outlier=True, joint_outlier_sigma=2.5,
        max_outer_iter=3, outer_tol=1e-4,
        admm_rho=1.0, admm_max_iter=20, admm_tol=1e-3,
        freq_weight=1.0, include_trend=False,
        min_obs=12,
    )
    assert res["valid"] is False
    assert res["u"].shape == (5, 2)
