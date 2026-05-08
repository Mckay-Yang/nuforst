import numpy as np
import pytest

from src.nufrost import _joint_outlier_mask


def test_joint_outlier_mask_keeps_clean_data():
    rng = np.random.default_rng(0)
    n = 100
    B = 4
    R = rng.normal(scale=1.0, size=(n, B))
    sigmas = np.ones(B)
    mask = _joint_outlier_mask(R, sigmas, sigma=2.5)
    assert mask.shape == (n,)
    assert mask.dtype == np.bool_
    # most points should survive
    assert mask.sum() >= 90


def test_joint_outlier_mask_removes_correlated_clouds():
    rng = np.random.default_rng(1)
    n = 100
    B = 4
    R = rng.normal(scale=1.0, size=(n, B))
    # Inject 5 cloud-like outliers at known indices, all bands hit together
    cloud_idx = [10, 25, 47, 72, 88]
    for idx in cloud_idx:
        R[idx, :] += 20.0
    sigmas = np.full(B, 1.0)
    mask = _joint_outlier_mask(R, sigmas, sigma=2.5)
    # All cloud indices should be masked out
    for idx in cloud_idx:
        assert not mask[idx], f"cloud at {idx} not masked"
    # Most clean points should survive
    clean = [i for i in range(n) if i not in cloud_idx]
    assert sum(mask[i] for i in clean) >= len(clean) - 5


def test_joint_outlier_mask_handles_zero_sigma_band():
    rng = np.random.default_rng(2)
    n = 80
    B = 3
    R = rng.normal(scale=1.0, size=(n, B))
    sigmas = np.array([1.0, 0.0, 1.0])  # Band 1 has zero scale (degenerate)
    mask = _joint_outlier_mask(R, sigmas, sigma=2.5)
    # Zero-sigma band should be ignored, mask still well-defined.
    assert mask.shape == (n,)
    assert mask.sum() > 0


def test_joint_outlier_mask_one_band_falls_back_to_marginal():
    rng = np.random.default_rng(3)
    n = 60
    B = 1
    R = rng.normal(scale=1.0, size=(n, B))
    R[5, 0] = 50.0  # extreme outlier
    sigmas = np.array([1.0])
    mask = _joint_outlier_mask(R, sigmas, sigma=2.5)
    assert not mask[5]
    assert mask.sum() >= 55
