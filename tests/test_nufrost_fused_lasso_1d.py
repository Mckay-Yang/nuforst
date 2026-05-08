import numpy as np
import pytest

from src.nufrost import _difference_weights, _fused_lasso_1d


def test_difference_weights_uniform_when_disabled():
    t = np.array([0.0, 86400.0, 5 * 86400.0, 6 * 86400.0])
    w = _difference_weights(t, enable_dt_weighting=False)
    assert w.shape == (3,)
    assert np.allclose(w, 1.0)


def test_difference_weights_inverse_sqrt_dt_when_enabled():
    t_days = np.array([0.0, 1.0, 5.0, 6.0])
    t_sec = t_days * 86400.0
    w = _difference_weights(t_sec, enable_dt_weighting=True)
    # Δt = [1d, 4d, 1d] -> w = [1, 1/2, 1]
    assert np.allclose(w, np.array([1.0, 0.5, 1.0]))


def test_difference_weights_floors_short_gaps_at_one_day():
    t_sec = np.array([0.0, 100.0, 200.0])  # both gaps below 1 day
    w = _difference_weights(t_sec, enable_dt_weighting=True)
    assert np.allclose(w, 1.0)


def test_fused_lasso_1d_recovers_clean_step():
    n = 200
    r = np.concatenate([np.zeros(100), np.ones(100)])
    weights = np.ones(n - 1, dtype=np.float64)
    u = _fused_lasso_1d(r, lambda_step=0.1, weights=weights)
    # Should yield two near-constant segments around 0 and 1
    assert np.std(u[:90]) < 1e-3
    assert np.std(u[110:]) < 1e-3
    assert abs(u[10] - 0.0) < 0.05
    assert abs(u[180] - 1.0) < 0.05


def test_fused_lasso_1d_rejects_single_pulse():
    n = 100
    r = np.zeros(n)
    r[50] = 5.0
    weights = np.ones(n - 1, dtype=np.float64)
    u = _fused_lasso_1d(r, lambda_step=2.5, weights=weights)
    # A single pulse costs the L1 twice; output should remain near zero everywhere.
    assert np.max(np.abs(u)) < 0.5


def test_fused_lasso_1d_constant_when_lambda_huge():
    rng = np.random.default_rng(0)
    r = rng.normal(size=80)
    weights = np.ones(79, dtype=np.float64)
    u = _fused_lasso_1d(r, lambda_step=1e6, weights=weights)
    # Effectively forces u to a constant (the mean of r).
    assert np.std(u) < 1e-3
    assert abs(u.mean() - r.mean()) < 1e-3


def test_fused_lasso_1d_zero_lambda_returns_input():
    rng = np.random.default_rng(1)
    r = rng.normal(size=50)
    weights = np.ones(49, dtype=np.float64)
    u = _fused_lasso_1d(r, lambda_step=0.0, weights=weights)
    assert np.allclose(u, r)


def test_fused_lasso_1d_handles_short_input():
    u1 = _fused_lasso_1d(np.array([3.0]), lambda_step=1.0, weights=np.zeros(0))
    assert u1.shape == (1,)
    assert u1[0] == 3.0
    u0 = _fused_lasso_1d(np.zeros(0), lambda_step=1.0, weights=np.zeros(0))
    assert u0.shape == (0,)


def _reference_fused_lasso_1d(r, lambda_step, weights, max_iter=200_000, tol=1e-12):
    """Plain (non-FISTA) projected dual ascent. Independent code path from
    the production solver. Slower but its primary purpose is correctness:
    the only shared mechanic with the production solver is the underlying
    KKT conditions of the same convex problem.
    """
    r = np.asarray(r, dtype=np.float64)
    n = r.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return r.copy()
    if lambda_step <= 0.0:
        return r.copy()
    w = np.asarray(weights, dtype=np.float64)
    lam = lambda_step * w
    z = np.zeros(n - 1, dtype=np.float64)
    step = 0.25
    for _ in range(max_iter):
        DT_z = np.zeros(n, dtype=np.float64)
        DT_z[:-1] -= z
        DT_z[1:] += z
        u = r - DT_z
        Du = np.diff(u)
        z_new = np.clip(z + step * Du, -lam, lam)
        if np.max(np.abs(z_new - z)) < tol:
            z = z_new
            break
        z = z_new
    DT_z = np.zeros(n, dtype=np.float64)
    DT_z[:-1] -= z
    DT_z[1:] += z
    return r - DT_z


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("uniform_weights", [True, False])
def test_fused_lasso_1d_matches_reference_solver(seed, uniform_weights):
    """Cross-validate against an independent slow-but-correct reference
    on randomized multi-step inputs. Catches solver bugs that planted
    fixed-input tests miss (e.g., the variable-lambda Condat regression).
    """
    rng = np.random.default_rng(seed)
    n = 60
    # piecewise-constant ground truth with two breaks
    levels = rng.normal(scale=2.0, size=3)
    seg_lens = [20, 20, 20]
    truth = np.concatenate([np.full(L, v) for L, v in zip(seg_lens, levels)])
    r = truth + 0.1 * rng.normal(size=n)
    if uniform_weights:
        w = np.ones(n - 1)
    else:
        # gap-like weighting: alternating short and long gaps
        w = np.where(np.arange(n - 1) % 5 == 0, 0.3, 1.0)
    lambda_step = 0.5

    u_fast = _fused_lasso_1d(r, lambda_step=lambda_step, weights=w)
    u_ref = _reference_fused_lasso_1d(r, lambda_step=lambda_step, weights=w)
    err = np.max(np.abs(u_fast - u_ref))
    assert err < 1e-3, (
        f"seed={seed} uniform={uniform_weights} max |u_fast - u_ref| = {err:.4g}"
    )
