import numpy as np
import pytest

from src.nufrost import _group_fused_lasso_admm, _fused_lasso_1d


def test_group_fused_lasso_admm_recovers_shared_breakpoint():
    """4-band signal with one shared step at t=120; per-band amplitudes
    differ. Verify ADMM recovers the shared breakpoint location and the
    per-band magnitudes."""
    n = 200
    B = 4
    truth = np.zeros((n, B))
    step_at = 120
    amps = np.array([0.2, 0.5, -0.3, 0.4])
    truth[step_at:, :] = amps
    rng = np.random.default_rng(0)
    R = truth + 0.05 * rng.normal(size=(n, B))
    weights = np.ones(n - 1)
    U = _group_fused_lasso_admm(R, lambda_step=0.4, weights=weights,
                                 rho=1.0, max_iter=200, tol=1e-6)
    # Recovered jump magnitudes should be within 30% of truth (convex
    # global optimum trades data fidelity vs L1 cost).
    pre = U[:step_at - 5, :].mean(axis=0)
    post = U[step_at + 5:, :].mean(axis=0)
    recovered = post - pre
    for b in range(B):
        rel = abs(recovered[b] - amps[b]) / max(abs(amps[b]), 1e-3)
        assert rel < 0.5, f"band {b} relative error {rel:.3f}"
    # All bands must jump in the same direction the true amplitudes have.
    for b in range(B):
        assert np.sign(recovered[b]) == np.sign(amps[b])


def test_group_fused_lasso_admm_zero_lambda_returns_input():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(40, 3))
    weights = np.ones(39)
    U = _group_fused_lasso_admm(R, lambda_step=0.0, weights=weights,
                                 rho=1.0, max_iter=200, tol=1e-6)
    assert np.allclose(U, R)


def test_group_fused_lasso_admm_huge_lambda_returns_constant():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(60, 5))
    weights = np.ones(59)
    U = _group_fused_lasso_admm(R, lambda_step=1e6, weights=weights,
                                 rho=1.0, max_iter=200, tol=1e-6)
    # Each band collapses to its own column mean.
    expected = np.tile(R.mean(axis=0), (60, 1))
    assert np.allclose(U, expected, atol=1e-3)


def test_group_fused_lasso_admm_b1_matches_singleband():
    """For B=1, the group L2 reduces to absolute value, so the ADMM
    solver must produce the same result as `_fused_lasso_1d`."""
    rng = np.random.default_rng(3)
    n = 80
    r = rng.normal(size=n)
    weights = np.ones(n - 1)
    u_admm = _group_fused_lasso_admm(r.reshape(n, 1), lambda_step=0.3,
                                      weights=weights, rho=1.0,
                                      max_iter=400, tol=1e-7).ravel()
    u_ref = _fused_lasso_1d(r, lambda_step=0.3, weights=weights)
    assert np.allclose(u_admm, u_ref, atol=1e-3)


def test_group_fused_lasso_admm_handles_short_input():
    U0 = _group_fused_lasso_admm(np.zeros((0, 3)), lambda_step=1.0,
                                  weights=np.zeros(0), rho=1.0,
                                  max_iter=10, tol=1e-6)
    assert U0.shape == (0, 3)
    U1 = _group_fused_lasso_admm(np.array([[1.0, 2.0]]), lambda_step=1.0,
                                  weights=np.zeros(0), rho=1.0,
                                  max_iter=10, tol=1e-6)
    assert U1.shape == (1, 2)
    assert np.allclose(U1, [[1.0, 2.0]])
