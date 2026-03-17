"""Unit tests for pynasonde.vipir.ngi.scale: NoiseProfile, AutoScaler."""

import numpy as np
import pandas as pd
import pytest

from pynasonde.vipir.ngi.scale import AutoScaler, NoiseProfile, parabola
from pynasonde.vipir.ngi.source import Dataset

# ---------------------------------------------------------------------------
# Helper: build a minimal synthetic Dataset that AutoScaler.extract() accepts
# ---------------------------------------------------------------------------


def _make_dataset(n_freq=10, n_range=20):
    ds = Dataset()
    ds.Frequency = np.linspace(1000.0, 10000.0, n_freq)  # kHz
    ds.Range = np.linspace(50.0, 500.0, n_range)  # km
    ds.O_mode_power = np.random.rand(n_freq, n_range) * 50.0
    ds.O_mode_noise = np.ones(n_freq) * 5.0
    ds.X_mode_power = np.random.rand(n_freq, n_range) * 40.0
    ds.X_mode_noise = np.ones(n_freq) * 4.0
    return ds


# ---------------------------------------------------------------------------
# parabola()
# ---------------------------------------------------------------------------


class TestParabola:
    def test_zero_coefficients(self):
        assert parabola(0.0, 0, 0, 0) == 0.0

    def test_linear_when_a_zero(self):
        # a=0, b=2, c=1 → 2*x + 1
        np.testing.assert_allclose(
            parabola(np.array([0.0, 1.0, 2.0]), 0, 2, 1), [1.0, 3.0, 5.0]
        )

    def test_returns_array_for_array_input(self):
        x = np.linspace(0, 1, 5)
        result = parabola(x, 1.0, 0.0, 0.0)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# NoiseProfile
# ---------------------------------------------------------------------------


class TestNoiseProfile:
    def test_defaults(self):
        np = NoiseProfile()
        assert np.type == "exp"
        assert np.profile == pytest.approx(1.5)

    def test_custom_constant(self):
        np_ = NoiseProfile(constant=2.0)
        assert np_.profile == pytest.approx(2.0)

    def test_get_exp_profile(self):
        prof = NoiseProfile()
        x = np.linspace(0, 10, 50)
        result = prof.get_exp_profile(x, a0=5.0, b0=0.5, x0=10.0)
        # At x=0: a0*exp(0) = 5.0
        assert result[0] == pytest.approx(5.0)
        # Should be monotonically decreasing (b0, x0 positive)
        assert result[-1] < result[0]
        assert prof.profile is result  # in-place update


# ---------------------------------------------------------------------------
# AutoScaler – construct, mdeian_filter, image_segmentation, fit_parabola
# ---------------------------------------------------------------------------


class TestAutoScalerExtract:
    def test_init_sets_image2d(self):
        ds = _make_dataset()
        scaler = AutoScaler(ds, mode="O")
        assert scaler.image2d.shape == (10, 20)

    def test_apply_filter_zeros_out_of_range(self):
        ds = _make_dataset()
        # Restrict height to [200, 400] and freq to [3, 8] MHz
        filt = {"frequency": [3.0, 8.0], "height": [200.0, 400.0]}
        scaler = AutoScaler(ds, mode="O", filter=filt, apply_filter=True)
        # Height axis: Range < 200 or Range > 400 should be zeroed
        low_range_idx = np.where(ds.Range < 200.0)[0]
        if len(low_range_idx):
            assert np.all(scaler.image2d[:, low_range_idx] == 0.0)

    def test_no_filter(self):
        ds = _make_dataset()
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        # image2d should not be all zeros (original data preserved)
        assert scaler.image2d.sum() > 0

    def test_x_mode(self):
        ds = _make_dataset()
        scaler = AutoScaler(ds, mode="X")
        assert scaler.image2d.shape == (10, 20)

    def test_frequency_and_height_stored(self):
        ds = _make_dataset()
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        # frequency stored in MHz (kHz / 1e3)
        assert scaler.frequency.shape == (10,)
        assert scaler.height.shape == (20,)


class TestAutoScalerMedianFilter:
    def test_produces_filtered_image(self):
        ds = _make_dataset(n_freq=8, n_range=8)
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        scaler.mdeian_filter(tau=2)
        assert scaler.filtered_2D_image.shape == scaler.image2d.shape

    def test_zeros_sparse_pixels(self):
        ds = _make_dataset(n_freq=8, n_range=8)
        # Make image mostly zero so tau requirement fails
        ds.O_mode_power = np.zeros((8, 8))
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        scaler.mdeian_filter(tau=4)
        assert scaler.filtered_2D_image.sum() == 0.0


class TestAutoScalerImageSegmentation:
    def test_kmeans_runs(self):
        ds = _make_dataset(n_freq=6, n_range=6)
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        scaler.mdeian_filter(tau=1)
        # Replace filtered image with a 3-column-divisible shape for cv2
        scaler.filtered_2D_image = np.random.rand(6, 6).astype(np.float32) * 50.0
        scaler.image_segmentation(segmentation_method="k-means")
        assert hasattr(scaler, "segmented_image")

    def test_none_data_uses_filtered_image(self):
        ds = _make_dataset(n_freq=6, n_range=6)
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        scaler.mdeian_filter(tau=1)
        scaler.filtered_2D_image = np.zeros((6, 6), dtype=np.float32)
        # Should not raise
        scaler.image_segmentation()


class TestAutoScalerFitParabola:
    def test_short_trace_warns_not_fit(self):
        ds = _make_dataset()
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        scaler.traces = {}
        scaler.trace_params = {}
        # <= 10 rows: no fit should happen
        tr = pd.DataFrame(
            {
                "frequency": np.linspace(2, 6, 5),
                "height": np.linspace(100, 300, 5),
            }
        )
        scaler.fit_parabola(tr, label=0)
        assert 0 not in scaler.traces

    def test_long_trace_fits(self):
        ds = _make_dataset()
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        scaler.traces = {}
        scaler.trace_params = {}
        freqs = np.linspace(2, 8, 20)
        heights = -((freqs - 5.0) ** 2) + 300.0  # parabola shape
        tr = pd.DataFrame({"frequency": freqs, "height": heights})
        scaler.fit_parabola(tr, label=1)
        assert 1 in scaler.traces
        assert 1 in scaler.trace_params

    def test_exception_in_fit_does_not_raise(self):
        ds = _make_dataset()
        scaler = AutoScaler(ds, mode="O", apply_filter=False)
        scaler.traces = {}
        scaler.trace_params = {}
        # NaN data → curve_fit will fail
        tr = pd.DataFrame(
            {
                "frequency": [np.nan] * 15,
                "height": [np.nan] * 15,
            }
        )
        scaler.fit_parabola(tr, label=2)  # should not raise
        assert 2 not in scaler.traces
