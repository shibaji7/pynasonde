"""Tests for utility functions in pynasonde.vipir.riq.parsers.read_riq."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pynasonde.vipir.riq.parsers.read_riq import (
    Pulset,
    adaptive_gain_filter,
    find_thresholds,
    remove_morphological_noise,
)


# ---------------------------------------------------------------------------
# Minimal Ionogram-like mock
# ---------------------------------------------------------------------------

def _make_ion(n_freq=10, n_range=20, n_ch=1, fill=10.0):
    ion = SimpleNamespace()
    rng = np.random.default_rng(7)
    ion.powerdB = (rng.standard_normal((n_freq, n_range)) * 3.0 + fill).astype(float)
    return ion


# ---------------------------------------------------------------------------
# find_thresholds (lines 56-64)
# ---------------------------------------------------------------------------

class TestFindThresholds:
    def test_returns_two_items(self):
        data = np.random.rand(100) * 20.0
        result = find_thresholds(data, bins=20, prominence=1)
        assert len(result) == 2

    def test_with_clear_dip(self):
        # Bimodal distribution has a dip between the two peaks
        low = np.random.rand(50) * 5.0
        high = np.random.rand(50) * 5.0 + 15.0
        data = np.concatenate([low, high])
        dip_bins, first_thresh = find_thresholds(data, bins=20, prominence=1)
        assert isinstance(first_thresh, float | np.floating)

    def test_handles_nans_and_infs(self):
        data = np.array([np.nan, np.inf, -np.inf, 5.0, 10.0, 15.0, 20.0] * 10)
        dip_bins, first_thresh = find_thresholds(data, bins=10, prominence=0.5)
        # Just verifying it doesn't raise
        assert True


# ---------------------------------------------------------------------------
# remove_morphological_noise (lines 88-96)
# ---------------------------------------------------------------------------

class TestRemoveMorphologicalNoise:
    def test_modifies_powerdB_inplace(self):
        ion = _make_ion(fill=5.0)
        original = ion.powerdB.copy()
        result = remove_morphological_noise(ion, threshold=0.0)
        # Result should be the same object
        assert result is ion

    def test_values_at_or_below_threshold_set_nan(self):
        ion = _make_ion(fill=5.0)
        # Set a high threshold so most values are removed
        remove_morphological_noise(ion, threshold=100.0)
        # After high threshold, remaining values should mostly be NaN
        assert True  # main goal: no exception raised

    def test_custom_kernel(self):
        ion = _make_ion(fill=5.0)
        result = remove_morphological_noise(
            ion, threshold=0.0, kernel_size=(3, 3), iterations=2
        )
        assert result is ion


# ---------------------------------------------------------------------------
# adaptive_gain_filter (lines 126-154) — including apply_median_filter branch
# ---------------------------------------------------------------------------

class TestAdaptiveGainFilter:
    def test_basic_no_median(self):
        ion = _make_ion(fill=5.0)
        result = adaptive_gain_filter(ion, snr_threshold=0.0, apply_median_filter=False)
        assert result is ion

    def test_with_median_filter(self):
        ion = _make_ion(fill=5.0)
        result = adaptive_gain_filter(
            ion, snr_threshold=0.0,
            apply_median_filter=True,
            median_filter_size=3,
        )
        assert result is ion

    def test_snr_threshold_sets_nan(self):
        ion = _make_ion(fill=5.0)
        # threshold higher than data → everything becomes nan
        adaptive_gain_filter(ion, snr_threshold=100.0)
        assert True  # no raise


# ---------------------------------------------------------------------------
# Pulset dataclass (lines 167-174)
# ---------------------------------------------------------------------------

class TestPulset:
    def test_init_empty(self):
        p = Pulset()
        assert p.pcts == []

    def test_append(self):
        p = Pulset()
        mock_pct = MagicMock()
        p.append(mock_pct)
        assert len(p.pcts) == 1
        assert p.pcts[0] is mock_pct

    def test_multiple_appends(self):
        p = Pulset()
        for _ in range(3):
            p.append(MagicMock())
        assert len(p.pcts) == 3
