"""Unit tests for pynasonde.vipir.riq.parsers.filter — IonogramFilter.

All tests use synthetic DataFrames; no real RIQ file is required.

Coverage
--------
IonogramFilter
    __init__          — default / custom parameter storage
    filter()          — single sounding, multi-sounding, empty input
    summary()         — string contains stage names
    stats             — populated after filter()
    _to_dataframe     — EchoExtractor / DataFrame / list inputs
Stage 1 (RFI)         — high height-IQR frequency blanked
Stage 2 (EP)          — echoes above ep_max_deg rejected; NaN EP passes
Stage 3 (Multi-hop)   — 2F echo at 2×h_1F + weaker amplitude flagged
Stage 4 (DBSCAN)      — isolated noise points rejected
Stage 5 (RANSAC)      — outlier far from smooth trace rejected
Stage 6 (Temporal)    — echoes not repeated across soundings removed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pynasonde.vipir.riq.parsers.filter import IonogramFilter


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 20,
    freq_khz: float = 5_000.0,
    height_km: float = 300.0,
    height_spread: float = 10.0,
    velocity_mps: float = 50.0,
    amplitude_db: float = 30.0,
    residual_deg: float | None = 15.0,
    sounding_index: int = 0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return a synthetic echo DataFrame with all required columns."""
    if rng is None:
        rng = np.random.default_rng(42)
    rows = {
        "frequency_khz": np.full(n, freq_khz),
        "height_km": rng.normal(height_km, height_spread, n),
        "velocity_mps": rng.normal(velocity_mps, 5.0, n),
        "amplitude_db": rng.normal(amplitude_db, 2.0, n),
        "residual_deg": (
            np.full(n, residual_deg)
            if residual_deg is not None
            else np.full(n, np.nan)
        ),
        "sounding_index": np.full(n, sounding_index, dtype=int),
    }
    return pd.DataFrame(rows)


def _make_clean_trace(
    n_freqs: int = 20,
    freq_start_khz: float = 2_000.0,
    freq_step_khz: float = 500.0,
    height_base_km: float = 200.0,
    height_slope: float = 10.0,
    echoes_per_freq: int = 4,
    noise_km: float = 3.0,
    sounding_index: int = 0,
) -> pd.DataFrame:
    """Simulate a clean h*(f) ionospheric trace spread over multiple frequencies."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_freqs):
        f = freq_start_khz + i * freq_step_khz
        h = height_base_km + i * height_slope
        for _ in range(echoes_per_freq):
            rows.append({
                "frequency_khz": f,
                "height_km": h + rng.normal(0, noise_km),
                "velocity_mps": rng.normal(50.0, 5.0),
                "amplitude_db": rng.normal(30.0, 2.0),
                "residual_deg": rng.uniform(5.0, 20.0),
                "sounding_index": sounding_index,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: minimal EchoExtractor-like stub
# ---------------------------------------------------------------------------

class _FakeExtractor:
    """Stand-in for EchoExtractor that exposes .to_dataframe()."""
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


# ===========================================================================
# IonogramFilter — __init__
# ===========================================================================

class TestInit:
    def test_defaults(self):
        f = IonogramFilter()
        assert f.rfi_enabled is True
        assert f.rfi_height_iqr_km == pytest.approx(300.0)
        assert f.rfi_min_echoes == 3
        assert f.ep_filter_enabled is True
        assert f.ep_max_deg == pytest.approx(90.0)
        assert f.multihop_enabled is True
        assert f.multihop_orders == (2, 3)
        assert f.multihop_height_tol_km == pytest.approx(50.0)
        assert f.multihop_snr_margin_db == pytest.approx(6.0)
        assert f.dbscan_enabled is True
        assert f.dbscan_eps == pytest.approx(1.0)
        assert f.dbscan_min_samples == 5
        assert f.ransac_enabled is True
        assert f.ransac_residual_km == pytest.approx(100.0)
        assert f.ransac_min_samples == 10
        assert f.ransac_n_iter == 200
        assert f.ransac_poly_degree == 3
        assert f.ransac_min_inlier_fraction == pytest.approx(0.3)
        assert f.temporal_enabled is True
        assert f.temporal_min_soundings == 3
        assert f.temporal_freq_bin_khz == pytest.approx(50.0)
        assert f.temporal_height_bin_km == pytest.approx(50.0)

    def test_custom_params(self):
        f = IonogramFilter(
            rfi_height_iqr_km=150.0,
            ep_max_deg=45.0,
            dbscan_eps=2.0,
            ransac_residual_km=50.0,
            temporal_min_soundings=2,
        )
        assert f.rfi_height_iqr_km == pytest.approx(150.0)
        assert f.ep_max_deg == pytest.approx(45.0)
        assert f.dbscan_eps == pytest.approx(2.0)
        assert f.ransac_residual_km == pytest.approx(50.0)
        assert f.temporal_min_soundings == 2

    def test_removed_params_not_present(self):
        """Reverted parameters must not appear on the instance."""
        f = IonogramFilter()
        assert not hasattr(f, "rfi_velocity_iqr_mps")
        assert not hasattr(f, "rfi_freq_span_fraction")
        assert not hasattr(f, "rfi_height_bin_km")
        assert not hasattr(f, "ransac_no_extrapolation")

    def test_stats_empty_before_filter(self):
        f = IonogramFilter()
        assert f.stats == {}

    def test_summary_before_filter(self):
        f = IonogramFilter()
        assert "No filter run yet" in f.summary()


# ===========================================================================
# _to_dataframe helper
# ===========================================================================

class TestToDataframe:
    def test_dataframe_passthrough(self):
        from pynasonde.vipir.riq.parsers.filter import _to_dataframe
        df = _make_df()
        result = _to_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_extractor_stub(self):
        from pynasonde.vipir.riq.parsers.filter import _to_dataframe
        df = _make_df()
        stub = _FakeExtractor(df)
        result = _to_dataframe(stub)
        assert isinstance(result, pd.DataFrame)

    def test_unknown_type_raises(self):
        from pynasonde.vipir.riq.parsers.filter import _to_dataframe
        with pytest.raises(TypeError):
            _to_dataframe(42)


# ===========================================================================
# filter() — basic contract
# ===========================================================================

class TestFilterContract:
    def _all_disabled(self) -> IonogramFilter:
        return IonogramFilter(
            rfi_enabled=False,
            ep_filter_enabled=False,
            multihop_enabled=False,
            dbscan_enabled=False,
            ransac_enabled=False,
            temporal_enabled=False,
        )

    def test_returns_dataframe(self):
        filt = self._all_disabled()
        df = _make_df(n=10)
        result = filt.filter(df)
        assert isinstance(result, pd.DataFrame)

    def test_sounding_index_column_present(self):
        filt = self._all_disabled()
        result = filt.filter(_make_df())
        assert "sounding_index" in result.columns

    def test_filter_mask_column_true(self):
        filt = self._all_disabled()
        result = filt.filter(_make_df())
        assert "filter_mask" in result.columns
        assert result["filter_mask"].all()

    def test_empty_input_returns_empty(self):
        filt = self._all_disabled()
        df = pd.DataFrame(columns=["frequency_khz", "height_km",
                                    "velocity_mps", "amplitude_db"])
        result = filt.filter(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_all_disabled_keeps_all_echoes(self):
        filt = self._all_disabled()
        df = _make_df(n=30)
        result = filt.filter(df)
        assert len(result) == 30

    def test_accepts_extractor_stub(self):
        filt = self._all_disabled()
        stub = _FakeExtractor(_make_df(n=15))
        result = filt.filter(stub)
        assert len(result) == 15

    def test_accepts_list_of_stubs(self):
        filt = self._all_disabled()
        stubs = [_FakeExtractor(_make_df(n=10)) for _ in range(3)]
        result = filt.filter(stubs)
        assert len(result) == 30

    def test_sounding_index_assigned_for_list(self):
        filt = self._all_disabled()
        stubs = [_FakeExtractor(_make_df(n=10)) for _ in range(3)]
        result = filt.filter(stubs)
        assert set(result["sounding_index"]) == {0, 1, 2}

    def test_stats_populated_after_filter(self):
        filt = IonogramFilter(
            rfi_enabled=True, ep_filter_enabled=False,
            multihop_enabled=False, dbscan_enabled=False,
            ransac_enabled=False, temporal_enabled=False,
        )
        filt.filter(_make_df(n=20))
        assert "rfi" in filt.stats
        assert "summary" in filt.stats

    def test_summary_contains_stage_labels(self):
        filt = IonogramFilter(
            rfi_enabled=True, ep_filter_enabled=True,
            multihop_enabled=False, dbscan_enabled=False,
            ransac_enabled=False, temporal_enabled=False,
        )
        filt.filter(_make_df(n=20))
        s = filt.summary()
        assert "RFI" in s
        assert "EP" in s


# ===========================================================================
# Stage 1 — RFI blanking
# ===========================================================================

class TestStageRFI:
    def _filt(self, **kw) -> IonogramFilter:
        defaults = dict(
            rfi_enabled=True, ep_filter_enabled=False,
            multihop_enabled=False, dbscan_enabled=False,
            ransac_enabled=False, temporal_enabled=False,
        )
        defaults.update(kw)
        return IonogramFilter(**defaults)

    def test_high_height_iqr_blanked(self):
        """A frequency with echoes spread 0–900 km has IQR >> 300 km → blanked."""
        rng = np.random.default_rng(1)
        # Noisy frequency: heights spread across full range
        noisy = pd.DataFrame({
            "frequency_khz": np.full(20, 5_000.0),
            "height_km": np.linspace(60, 940, 20),  # guaranteed IQR = 440 km > 300 km
            "velocity_mps": np.zeros(20),
            "amplitude_db": np.full(20, 30.0),
            "residual_deg": np.full(20, 10.0),
            "sounding_index": np.zeros(20, dtype=int),
        })
        # Clean frequency: echoes tightly clustered
        clean = pd.DataFrame({
            "frequency_khz": np.full(20, 6_000.0),
            "height_km": rng.normal(300, 5, 20),
            "velocity_mps": np.zeros(20),
            "amplitude_db": np.full(20, 30.0),
            "residual_deg": np.full(20, 10.0),
            "sounding_index": np.zeros(20, dtype=int),
        })
        df = pd.concat([noisy, clean], ignore_index=True)
        result = self._filt(rfi_height_iqr_km=300.0).filter(df)
        # The noisy frequency should be removed
        assert 5_000.0 not in result["frequency_khz"].values
        # The clean frequency should survive
        assert 6_000.0 in result["frequency_khz"].values

    def test_clean_frequency_survives(self):
        """A frequency with tightly clustered echoes must not be blanked."""
        df = _make_df(n=20, height_spread=5.0)  # IQR << 300 km
        result = self._filt(rfi_height_iqr_km=300.0).filter(df)
        assert len(result) == 20

    def test_below_min_echoes_not_tested(self):
        """Frequencies with fewer than rfi_min_echoes are never blanked."""
        rng = np.random.default_rng(2)
        tiny = pd.DataFrame({
            "frequency_khz": [5_000.0, 5_000.0],   # only 2 echoes < min_echoes=3
            "height_km": [100.0, 900.0],             # huge spread, but too few
            "velocity_mps": [0.0, 0.0],
            "amplitude_db": [30.0, 30.0],
            "residual_deg": [10.0, 10.0],
            "sounding_index": [0, 0],
        })
        result = self._filt(rfi_min_echoes=3).filter(tiny)
        assert len(result) == 2   # nothing blanked

    def test_stats_rfi_key(self):
        df = _make_df(n=20)
        filt = self._filt()
        filt.filter(df)
        assert "rejected" in filt.stats["rfi"]


# ===========================================================================
# Stage 2 — EP filter
# ===========================================================================

class TestStageEP:
    def _filt(self, ep_max_deg=90.0) -> IonogramFilter:
        return IonogramFilter(
            rfi_enabled=False, ep_filter_enabled=True,
            multihop_enabled=False, dbscan_enabled=False,
            ransac_enabled=False, temporal_enabled=False,
            ep_max_deg=ep_max_deg,
        )

    def test_echoes_below_threshold_kept(self):
        df = _make_df(n=20, residual_deg=30.0)
        result = self._filt(ep_max_deg=90.0).filter(df)
        assert len(result) == 20

    def test_echoes_above_threshold_rejected(self):
        df = _make_df(n=20, residual_deg=120.0)
        result = self._filt(ep_max_deg=90.0).filter(df)
        assert len(result) == 0

    def test_nan_ep_always_passes(self):
        """Echoes with NaN EP (e.g. PL407 n_rx=2) must never be rejected."""
        df = _make_df(n=20, residual_deg=None)
        result = self._filt(ep_max_deg=45.0).filter(df)
        assert len(result) == 20

    def test_mixed_ep_partial_rejection(self):
        good = _make_df(n=10, residual_deg=20.0)
        bad  = _make_df(n=10, residual_deg=100.0)
        df = pd.concat([good, bad], ignore_index=True)
        result = self._filt(ep_max_deg=90.0).filter(df)
        assert len(result) == 10

    def test_stats_ep_key(self):
        df = _make_df(n=20, residual_deg=100.0)
        filt = self._filt(ep_max_deg=90.0)
        filt.filter(df)
        assert "ep" in filt.stats


# ===========================================================================
# Stage 3 — Multi-hop
# ===========================================================================

class TestStageMultihop:
    def _filt(self, **kw) -> IonogramFilter:
        defaults = dict(
            rfi_enabled=False, ep_filter_enabled=False,
            multihop_enabled=True, dbscan_enabled=False,
            ransac_enabled=False, temporal_enabled=False,
            multihop_orders=(2,), multihop_height_tol_km=30.0,
            multihop_snr_margin_db=6.0,
        )
        defaults.update(kw)
        return IonogramFilter(**defaults)

    def _make_with_2f(self, h_1f=200.0, amp_1f=40.0, amp_2f=30.0, n_1f=10) -> pd.DataFrame:
        """Build a DataFrame with a clear 1F cluster and one 2F echo."""
        rng = np.random.default_rng(10)
        # 1F echoes clustered near h_1f
        rows_1f = pd.DataFrame({
            "frequency_khz": np.full(n_1f, 5_000.0),
            "height_km": rng.normal(h_1f, 5.0, n_1f),
            "velocity_mps": np.zeros(n_1f),
            "amplitude_db": np.full(n_1f, amp_1f),
            "residual_deg": np.full(n_1f, 10.0),
            "sounding_index": np.zeros(n_1f, dtype=int),
        })
        # 2F echo at exactly 2×h_1f, weaker
        rows_2f = pd.DataFrame({
            "frequency_khz": [5_000.0],
            "height_km": [2.0 * h_1f],
            "velocity_mps": [0.0],
            "amplitude_db": [amp_2f],
            "residual_deg": [10.0],
            "sounding_index": [0],
        })
        return pd.concat([rows_1f, rows_2f], ignore_index=True)

    def test_2f_echo_flagged(self):
        df = self._make_with_2f(h_1f=200.0, amp_1f=40.0, amp_2f=30.0)
        result = self._filt().filter(df)
        # 2F echo at ~400 km should be removed
        assert result["height_km"].max() < 380.0

    def test_1f_echoes_retained(self):
        df = self._make_with_2f(h_1f=200.0, amp_1f=40.0, amp_2f=30.0)
        result = self._filt().filter(df)
        assert len(result) >= 10   # all 1F echoes survive

    def test_strong_2f_not_flagged(self):
        """A 2F echo that is NOT weaker by the margin must be kept."""
        df = self._make_with_2f(h_1f=200.0, amp_1f=40.0, amp_2f=38.0)
        # 2F is only 2 dB weaker < snr_margin_db=6 → keep it
        result = self._filt(multihop_snr_margin_db=6.0).filter(df)
        assert result["height_km"].max() > 380.0

    def test_stats_multihop_key(self):
        df = self._make_with_2f()
        filt = self._filt()
        filt.filter(df)
        assert "multihop" in filt.stats


# ===========================================================================
# Stage 4 — DBSCAN
# ===========================================================================

class TestStageDBSCAN:
    def _filt(self, **kw) -> IonogramFilter:
        defaults = dict(
            rfi_enabled=False, ep_filter_enabled=False,
            multihop_enabled=False, dbscan_enabled=True,
            ransac_enabled=False, temporal_enabled=False,
            dbscan_eps=1.0, dbscan_min_samples=5,
            dbscan_features=("frequency_khz", "height_km",
                             "velocity_mps", "amplitude_db"),
        )
        defaults.update(kw)
        return IonogramFilter(**defaults)

    def test_dense_cluster_survives(self):
        """A tight cluster of echoes at a single frequency should all be kept."""
        df = _make_clean_trace(n_freqs=10, echoes_per_freq=8)
        result = self._filt(dbscan_min_samples=4).filter(df)
        # Most echoes should survive
        assert len(result) > len(df) * 0.5

    def test_isolated_noise_rejected(self):
        """Echoes far from any cluster (noise label = -1) should be removed."""
        rng = np.random.default_rng(99)
        # Tight cluster
        cluster = pd.DataFrame({
            "frequency_khz": np.full(30, 5_000.0),
            "height_km": rng.normal(300, 3, 30),
            "velocity_mps": rng.normal(50, 2, 30),
            "amplitude_db": rng.normal(30, 1, 30),
            "residual_deg": np.full(30, 10.0),
            "sounding_index": np.zeros(30, dtype=int),
        })
        # Single isolated noise point far away
        noise = pd.DataFrame({
            "frequency_khz": [10_000.0],
            "height_km": [900.0],
            "velocity_mps": [999.0],
            "amplitude_db": [5.0],
            "residual_deg": [80.0],
            "sounding_index": [0],
        })
        df = pd.concat([cluster, noise], ignore_index=True)
        result = self._filt(dbscan_min_samples=3).filter(df)
        # The isolated noise point should be gone
        assert 10_000.0 not in result["frequency_khz"].values

    def test_nan_features_handled(self):
        """NaN features (e.g. residual_deg for PL407) must not crash DBSCAN."""
        df = _make_df(n=20, residual_deg=None)
        filt = self._filt(dbscan_features=("frequency_khz", "height_km",
                                            "velocity_mps", "amplitude_db",
                                            "residual_deg"))
        result = filt.filter(df)
        assert isinstance(result, pd.DataFrame)

    def test_too_few_echoes_skips_stage(self):
        """If fewer than min_samples echoes survive, DBSCAN is skipped."""
        df = _make_df(n=2)
        result = self._filt(dbscan_min_samples=10).filter(df)
        assert len(result) == 2   # nothing rejected — stage skipped

    def test_stats_dbscan_key(self):
        df = _make_clean_trace()
        filt = self._filt()
        filt.filter(df)
        assert "dbscan" in filt.stats


# ===========================================================================
# Stage 5 — RANSAC
# ===========================================================================

class TestStageRANSAC:
    def _filt(self, **kw) -> IonogramFilter:
        defaults = dict(
            rfi_enabled=False, ep_filter_enabled=False,
            multihop_enabled=False, dbscan_enabled=False,
            ransac_enabled=True, temporal_enabled=False,
            ransac_residual_km=30.0,
            ransac_min_samples=5,
            ransac_n_iter=100,
            ransac_poly_degree=2,
            ransac_min_inlier_fraction=0.3,
        )
        defaults.update(kw)
        return IonogramFilter(**defaults)

    def _make_trace_with_outlier(
        self,
        n_freqs: int = 15,
        outlier_height: float = 900.0,
    ) -> pd.DataFrame:
        """Clean trace + one extreme outlier."""
        trace = _make_clean_trace(
            n_freqs=n_freqs, echoes_per_freq=5,
            height_base_km=200.0, height_slope=8.0, noise_km=2.0,
        )
        outlier = pd.DataFrame({
            "frequency_khz": [trace["frequency_khz"].mean()],
            "height_km": [outlier_height],
            "velocity_mps": [50.0],
            "amplitude_db": [30.0],
            "residual_deg": [10.0],
            "sounding_index": [0],
        })
        return pd.concat([trace, outlier], ignore_index=True)

    def test_outlier_rejected(self):
        df = self._make_trace_with_outlier(outlier_height=900.0)
        result = self._filt().filter(df)
        # The extreme outlier at 900 km should not survive
        assert result["height_km"].max() < 500.0

    def test_inliers_retained(self):
        df = self._make_trace_with_outlier(outlier_height=900.0)
        n_before = len(df) - 1   # all but the outlier
        result = self._filt().filter(df)
        assert len(result) >= int(n_before * 0.7)   # most inliers kept

    def test_skipped_when_too_few_echoes(self):
        """RANSAC is silently skipped when fewer than min_samples echoes remain."""
        df = _make_df(n=3)
        result = self._filt(ransac_min_samples=10).filter(df)
        assert len(result) == 3

    def test_per_sounding_independence(self):
        """RANSAC must fit each sounding's trace independently."""
        df0 = _make_clean_trace(sounding_index=0, n_freqs=10, echoes_per_freq=5,
                                 height_base_km=200.0)
        df1 = _make_clean_trace(sounding_index=1, n_freqs=10, echoes_per_freq=5,
                                 height_base_km=350.0)   # different layer height
        # Pass as list so filter() assigns sounding_index=0/1 per DataFrame
        result = self._filt(ransac_residual_km=50.0).filter([df0, df1])
        # Both soundings should have survivors
        assert (result["sounding_index"] == 0).sum() > 0
        assert (result["sounding_index"] == 1).sum() > 0

    def test_stats_ransac_key(self):
        df = self._make_trace_with_outlier()
        filt = self._filt()
        filt.filter(df)
        assert "ransac" in filt.stats


# ===========================================================================
# Stage 6 — Temporal coherence
# ===========================================================================

class TestStageTemporal:
    def _filt(self, min_soundings=2, **kw) -> IonogramFilter:
        defaults = dict(
            rfi_enabled=False, ep_filter_enabled=False,
            multihop_enabled=False, dbscan_enabled=False,
            ransac_enabled=False, temporal_enabled=True,
            temporal_min_soundings=min_soundings,
            temporal_freq_bin_khz=100.0,
            temporal_height_bin_km=50.0,
        )
        defaults.update(kw)
        return IonogramFilter(**defaults)

    def _same_cell(self, n=20, freq_khz=5_000.0, height_km=300.0) -> pd.DataFrame:
        """Create echoes that land in the same (f_bin, h_bin) cell."""
        rng = np.random.default_rng(5)
        rows = []
        for si in range(3):
            rows.append(pd.DataFrame({
                "frequency_khz": np.full(n, freq_khz),
                "height_km": rng.normal(height_km, 5, n),
                "velocity_mps": np.zeros(n),
                "amplitude_db": np.full(n, 30.0),
                "residual_deg": np.full(n, 10.0),
                "sounding_index": np.full(n, si, dtype=int),
            }))
        return pd.concat(rows, ignore_index=True)

    def test_coherent_echoes_retained(self):
        """Echoes in the same cell across 3 soundings → all kept (min=2)."""
        df = self._same_cell()
        result = self._filt(min_soundings=2).filter([
            _FakeExtractor(df[df["sounding_index"] == i]) for i in range(3)
        ])
        assert len(result) > 0

    def test_unique_cell_removed(self):
        """An echo appearing in only 1 of 3 soundings is removed when min=2."""
        # Three soundings share a cell at 5000 kHz / 300 km
        shared = self._same_cell()
        # Unique echo: only in sounding 0, at a completely different cell
        unique = pd.DataFrame({
            "frequency_khz": [15_000.0],   # far from any shared cell
            "height_km": [850.0],
            "velocity_mps": [0.0],
            "amplitude_db": [30.0],
            "residual_deg": [10.0],
            "sounding_index": [0],
        })
        dfs = [
            pd.concat([shared[shared["sounding_index"] == 0], unique], ignore_index=True),
            shared[shared["sounding_index"] == 1].copy(),
            shared[shared["sounding_index"] == 2].copy(),
        ]
        result = self._filt(min_soundings=2).filter(
            [_FakeExtractor(d) for d in dfs]
        )
        assert 15_000.0 not in result["frequency_khz"].values

    def test_temporal_skipped_single_sounding(self):
        """Stage 6 must be silently skipped when only one sounding is provided."""
        df = _make_df(n=30)
        filt = self._filt(min_soundings=3)
        result = filt.filter(df)
        # All echoes should survive (nothing to compare against)
        assert len(result) == 30
        assert "temporal" not in filt.stats

    def test_stats_temporal_key(self):
        df = self._same_cell()
        filt = self._filt(min_soundings=2)
        filt.filter([_FakeExtractor(df[df["sounding_index"] == i]) for i in range(3)])
        assert "temporal" in filt.stats


# ===========================================================================
# summary() and stats
# ===========================================================================

class TestSummaryAndStats:
    def test_summary_contains_total_counts(self):
        filt = IonogramFilter(
            rfi_enabled=True, ep_filter_enabled=True,
            multihop_enabled=False, dbscan_enabled=False,
            ransac_enabled=False, temporal_enabled=False,
        )
        df = _make_df(n=20)
        filt.filter(df)
        s = filt.summary()
        assert "Input echoes" in s or "total" in s.lower()

    def test_summary_shows_each_active_stage(self):
        filt = IonogramFilter(
            rfi_enabled=True, ep_filter_enabled=True,
            multihop_enabled=True, dbscan_enabled=True,
            ransac_enabled=True, temporal_enabled=False,
            dbscan_features=("frequency_khz", "height_km",
                             "velocity_mps", "amplitude_db"),
        )
        df = _make_clean_trace(n_freqs=15, echoes_per_freq=6)
        filt.filter(df)
        s = filt.summary()
        for label in ("RFI", "EP", "Multi-hop", "DBSCAN", "RANSAC"):
            assert label in s

    def test_stats_summary_fields(self):
        filt = IonogramFilter(
            rfi_enabled=False, ep_filter_enabled=False,
            multihop_enabled=False, dbscan_enabled=False,
            ransac_enabled=False, temporal_enabled=False,
        )
        df = _make_df(n=20)
        filt.filter(df)
        s = filt.stats["summary"]
        assert "total_input" in s
        assert "total_kept" in s
        assert s["total_input"] == 20
        assert s["total_kept"] == 20
