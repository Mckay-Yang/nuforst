import numpy as np
import pytest

from src.nufrost import _difference_weights


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
