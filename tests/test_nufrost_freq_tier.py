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
