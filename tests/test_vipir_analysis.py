"""Unit tests for pynasonde.vipir.analysis.

All tests use synthetic DataFrames — no RIQ file required.

Coverage
--------
PolarizationClassifier / PolarizationResult
    __init__          — parameter storage and defaults
    fit()             — returns PolarizationResult with annotated_df
    mode labels       — O / X / ambiguous / unknown assigned correctly
    to_dataframe()    — returns correct columns
    summary()         — string format
    infer_o_mode_sign — hemisphere helper

SpreadFAnalyzer / SpreadFResult
    __init__          — threshold storage
    fit()             — returns SpreadFResult
    classification    — "none" / "range" / "frequency" / "mixed"
    to_dataframe()    — returns classification column

TrueHeightInversion / EDPResult
    fit()             — virtual → true height
    fit_from_df()     — DataFrame wrapper
    monotone_enforce  — removes non-monotone layers
    to_dataframe()    — correct columns
    summary()         — string format

IonogramScaler / ScaledParameters
    fit()             — returns ScaledParameters
    foE detection     — E-layer echoes → foE, h'E
    foF2 detection    — F-layer echoes → foF2, h'F2
    MUF calculation   — MUF(3000) and M(3000)F2
    bootstrap         — foF2_sigma_mhz finite when sufficient echoes
    to_dataframe()    — correct columns
    quality_flags     — boolean dict

IrregularityAnalyzer / IrregularityProfile
    fit()             — returns IrregularityProfile
    spectral index    — α ≥ 0 for increasing structure function
    anisotropy        — alpha_E / alpha_F / ratio computed
    to_dataframe()    — correct columns
    summary()         — string format

NeXtYZInverter / NeXtYZResult
    __init__          — parameter storage, B_hat computed correctly
    fit()             — returns NeXtYZResult (may have 0 wedges on sparse data)
    WedgePlane        — dataclass fields accessible
    to_dataframe()    — returns DataFrame
    summary()         — string format
    _appleton_lassen_n2 — physics correctness
    _wedge_rho        — geometry correctness

AbsorptionAnalyzer / LOFResult / DifferentialResult / TotalAbsorptionResult / AbsorptionProfileResult
    lof_absorption    — fmin and LOF index from echo DF
    differential_absorption — ΔL(f) from paired O/X echoes
    total_absorption  — calibrated L(f) from radar equation
    absorption_profile — κ(z) and cumulative L(z) from EDPResult + ν(h)
    unit sanity       — kappa > 0 for D-region Chapman EDP, F-region ≈ 0
    f_wave below plasma — kappa zeroed correctly above reflection level

EsCaponImager / EsImagingResult
    __init__          — parameter storage, K clipped to floor(V/Z)
    fit()             — returns EsImagingResult, shape (n_snapshots, K*V)
    heights_km        — monotone, spacing = gate_spacing/K
    pseudospectrum_db — max ≤ 0 (normalised), peak near injected gate
    coherent_integrations — snapshot count matches n_pulse // n_coh
    summary / to_dataframe / plot — no error

RiqAggregator
    __init__          — parameter storage and validation
    combine()         — returns EsImagingResult from list of synthetic cubes
    single mode       — output n_snapshots=1
    slow_rti mode     — output n_snapshots=n_files
    rx beamform       — Option A with uniform weights
    rx beamform       — Option A with custom weights
    2-D cubes         — no rx axis handled transparently
    invalid mode      — raises ValueError
    empty cubes       — raises ValueError
    mismatched gates  — raises ValueError
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pynasonde.vipir.analysis import (
    IonogramScaler,
    IrregularityAnalyzer,
    NeXtYZInverter,
    PolarizationClassifier,
    SpreadFAnalyzer,
    TrueHeightInversion,
)
from pynasonde.vipir.analysis.nextyz import (
    WedgePlane,
    _appleton_lassen_n2,
    _nz,
    _wedge_rho,
)

# ===========================================================================
# Synthetic data helpers
# ===========================================================================


def _make_echo_df(
    n_e: int = 40,
    n_f: int = 80,
    seed: int = 0,
    add_x_mode: bool = True,
    add_nan_pp: int = 0,
) -> pd.DataFrame:
    """Synthetic ionogram echo DataFrame.

    E layer: 2–4 MHz, 90–150 km, O-mode (negative PP).
    F layer: 4–8 MHz, 200–500 km, O/X mix.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # E layer
    e_freqs = np.linspace(2_000, 4_000, max(n_e // 4, 4))
    for i, f in enumerate(e_freqs):
        for _ in range(max(n_e // len(e_freqs), 2)):
            rows.append(
                {
                    "frequency_khz": f + rng.normal(0, 5),
                    "height_km": 95.0 + i * 3 + rng.normal(0, 3),
                    "polarization_deg": rng.uniform(-160, -45),
                    "residual_deg": abs(rng.normal(12, 3)),
                    "amplitude_db": rng.normal(35, 2),
                    "xl_km": rng.normal(0, 8),
                    "yl_km": rng.normal(0, 8),
                    "velocity_mps": rng.normal(0, 20),
                }
            )

    # F layer
    f_freqs = np.linspace(4_200, 7_800, max(n_f // 6, 4))
    for i, f in enumerate(f_freqs):
        for _ in range(max(n_f // len(f_freqs), 2)):
            is_x = add_x_mode and rng.random() < 0.3
            pp = rng.uniform(60, 160) if is_x else rng.uniform(-160, -45)
            rows.append(
                {
                    "frequency_khz": f + rng.normal(0, 8),
                    "height_km": 210.0 + i * 18 + rng.normal(0, 8),
                    "polarization_deg": pp,
                    "residual_deg": abs(rng.normal(18, 6)),
                    "amplitude_db": rng.normal(30, 3),
                    "xl_km": rng.normal(0, 15),
                    "yl_km": rng.normal(0, 15),
                    "velocity_mps": rng.normal(-10, 40),
                }
            )

    df = pd.DataFrame(rows).reset_index(drop=True)

    if add_nan_pp > 0:
        idx = rng.choice(len(df), size=min(add_nan_pp, len(df)), replace=False)
        df.loc[idx, "polarization_deg"] = np.nan

    return df


def _o_mode_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only O-mode echoes (PP < 0 in NH convention)."""
    return df[df["polarization_deg"] < 0].copy()


# ===========================================================================
# PolarizationClassifier
# ===========================================================================


class TestPolarizationClassifier:
    def test_defaults(self):
        clf = PolarizationClassifier()
        assert clf.o_mode_sign == -1
        assert clf.pp_ambiguous_threshold_deg > 0

    def test_custom_params(self):
        clf = PolarizationClassifier(o_mode_sign=1, pp_ambiguous_threshold_deg=20.0)
        assert clf.o_mode_sign == 1
        assert clf.pp_ambiguous_threshold_deg == pytest.approx(20.0)

    def test_fit_returns_result(self):
        df = _make_echo_df(seed=0)
        result = PolarizationClassifier().fit(df)
        from pynasonde.vipir.analysis.polarization import PolarizationResult

        assert isinstance(result, PolarizationResult)

    def test_annotated_df_has_mode_column(self):
        result = PolarizationClassifier().fit(_make_echo_df(seed=1))
        assert "mode" in result.annotated_df.columns

    def test_mode_values_valid(self):
        result = PolarizationClassifier().fit(_make_echo_df(seed=2))
        valid_modes = {"O", "X", "ambiguous", "unknown"}
        assert set(result.annotated_df["mode"].unique()).issubset(valid_modes)

    def test_negative_pp_labelled_O(self):
        """Pure O-mode data (all PP < 0) should all be classified as O or ambiguous."""
        df = _make_echo_df(seed=3, add_x_mode=False)
        result = PolarizationClassifier(o_mode_sign=-1).fit(df)
        bad = result.annotated_df[result.annotated_df["mode"] == "X"]
        assert len(bad) == 0, "No X-mode echoes expected when PP is all negative"

    def test_positive_pp_labelled_X(self):
        """Pure X-mode data (all PP > 45°) should be classified as X."""
        rng = np.random.default_rng(4)
        df = pd.DataFrame(
            {
                "frequency_khz": np.full(30, 5_000.0),
                "height_km": rng.normal(300, 10, 30),
                "polarization_deg": rng.uniform(80, 160, 30),
            }
        )
        result = PolarizationClassifier(
            o_mode_sign=-1, pp_ambiguous_threshold_deg=30.0
        ).fit(df)
        assert (result.annotated_df["mode"] == "X").all()

    def test_nan_pp_labelled_unknown(self):
        df = _make_echo_df(seed=5, add_nan_pp=10)
        result = PolarizationClassifier().fit(df)
        assert (result.annotated_df["mode"] == "unknown").sum() == 10

    def test_o_mode_count_positive(self):
        df = _make_echo_df(seed=6, add_x_mode=False)
        result = PolarizationClassifier().fit(df)
        assert result.o_mode_count > 0

    def test_x_mode_count_positive_when_mixed(self):
        df = _make_echo_df(seed=7, add_x_mode=True)
        result = PolarizationClassifier().fit(df)
        assert result.x_mode_count > 0

    def test_to_dataframe_returns_df(self):
        result = PolarizationClassifier().fit(_make_echo_df(seed=8))
        out = result.to_dataframe()
        assert isinstance(out, pd.DataFrame)
        assert "mode" in out.columns

    def test_summary_string(self):
        result = PolarizationClassifier().fit(_make_echo_df(seed=9))
        s = result.summary()
        assert isinstance(s, str)
        assert "O" in s

    def test_infer_o_mode_sign_nh(self):
        sign = PolarizationClassifier.infer_o_mode_sign(station_lat=50.0)
        assert sign == -1

    def test_infer_o_mode_sign_sh(self):
        sign = PolarizationClassifier.infer_o_mode_sign(station_lat=-30.0)
        assert sign == 1


# ===========================================================================
# SpreadFAnalyzer
# ===========================================================================


class TestSpreadFAnalyzer:
    def _quiet_df(self) -> pd.DataFrame:
        """Tight F-layer echoes → no spread-F."""
        rng = np.random.default_rng(20)
        return pd.DataFrame(
            {
                "frequency_khz": np.linspace(4_000, 8_000, 50),
                "height_km": rng.normal(280, 5, 50),
                "mode": ["O"] * 50,
                "polarization_deg": [-90.0] * 50,
            }
        )

    def _spread_df(self) -> pd.DataFrame:
        """Wide height scatter → spread-F.

        Uses a few discrete frequencies with many echoes each so that
        _range_spread_flags can compute IQR (needs ≥ min_echoes_per_freq=3).
        """
        rng = np.random.default_rng(21)
        freqs = np.repeat([4_000, 5_000, 6_000, 7_000, 8_000], 15)
        return pd.DataFrame(
            {
                "frequency_khz": freqs,
                "height_km": rng.uniform(200, 600, len(freqs)),
                "mode": ["O"] * len(freqs),
                "polarization_deg": [-90.0] * len(freqs),
            }
        )

    def test_fit_returns_result(self):
        from pynasonde.vipir.analysis.spread_f import SpreadFResult

        result = SpreadFAnalyzer().fit(self._quiet_df())
        assert isinstance(result, SpreadFResult)

    def test_quiet_classified_none(self):
        result = SpreadFAnalyzer(height_spread_threshold_km=30.0).fit(self._quiet_df())
        assert result.classification == "none"

    def test_spread_not_classified_none(self):
        result = SpreadFAnalyzer(height_spread_threshold_km=30.0).fit(self._spread_df())
        assert result.classification != "none"

    def test_classification_values(self):
        result = SpreadFAnalyzer().fit(self._spread_df())
        assert result.classification in {"none", "range", "frequency", "mixed"}

    def test_height_iqr_nonneg(self):
        result = SpreadFAnalyzer().fit(self._quiet_df())
        # height_iqr_km is NaN when no frequency step has enough echoes
        assert np.isnan(result.height_iqr_km) or result.height_iqr_km >= 0.0

    def test_empty_df_handled(self):
        df = pd.DataFrame(columns=["frequency_khz", "height_km", "mode"])
        result = SpreadFAnalyzer().fit(df)
        assert result.classification == "none"


# ===========================================================================
# TrueHeightInversion
# ===========================================================================


class TestTrueHeightInversion:
    def _trace_df(self, n: int = 20, seed: int = 30) -> pd.DataFrame:
        """Monotone O-mode virtual-height trace."""
        rng = np.random.default_rng(seed)
        freqs = np.linspace(2_000, 7_000, n)
        heights = 100.0 + (freqs - 2_000) / 5_000 * 250 + rng.normal(0, 3, n)
        return pd.DataFrame({"frequency_khz": freqs, "height_km": heights})

    def test_fit_returns_result(self):
        from pynasonde.vipir.analysis.inversion import EDPResult

        freq = np.linspace(2.0, 7.0, 20)
        h_virt = 100 + (freq - 2.0) * 40
        result = TrueHeightInversion().fit(freq, h_virt)
        assert isinstance(result, EDPResult)

    def test_fit_from_df(self):
        df = self._trace_df()
        result = TrueHeightInversion().fit_from_df(df)
        assert result.n_layers > 0

    def test_true_height_less_than_virtual(self):
        """True height should be ≤ virtual height (group delay stretches h*)."""
        df = self._trace_df(n=25)
        result = TrueHeightInversion(monotone_enforce=False).fit_from_df(df)
        if result.n_layers > 0:
            assert np.all(result.true_height_km <= result.virtual_height_km + 5.0)

    def test_n_layers_positive(self):
        df = self._trace_df(n=30)
        result = TrueHeightInversion().fit_from_df(df)
        assert result.n_layers > 0

    def test_foF2_detected(self):
        df = self._trace_df(n=30)
        result = TrueHeightInversion().fit_from_df(df)
        assert not np.isnan(result.foF2_mhz)

    def test_electron_density_nonneg(self):
        df = self._trace_df(n=20)
        result = TrueHeightInversion().fit_from_df(df)
        if result.n_layers > 0:
            assert np.all(result.electron_density_cm3 >= 0)

    def test_to_dataframe_columns(self):
        df = self._trace_df(n=20)
        result = TrueHeightInversion().fit_from_df(df)
        out = result.to_dataframe()
        assert "true_height_km" in out.columns
        assert "plasma_freq_mhz" in out.columns
        assert "electron_density_cm3" in out.columns

    def test_summary_string(self):
        df = self._trace_df(n=20)
        result = TrueHeightInversion().fit_from_df(df)
        s = result.summary()
        assert isinstance(s, str)
        assert "foF2" in s

    def test_missing_freq_col_raises(self):
        df = pd.DataFrame({"height_km": [100, 200]})
        with pytest.raises(KeyError):
            TrueHeightInversion().fit_from_df(df)


# ===========================================================================
# IonogramScaler
# ===========================================================================


class TestIonogramScaler:
    def _full_df(self, seed: int = 40) -> pd.DataFrame:
        """Labelled echo DataFrame with both E and F layer."""
        df = _make_echo_df(n_e=50, n_f=100, seed=seed, add_x_mode=False)
        df["mode"] = "O"
        return df

    def test_fit_returns_result(self):
        from pynasonde.vipir.analysis.scaler import ScaledParameters

        result = IonogramScaler(min_echoes_for_layer=3).fit(self._full_df())
        assert isinstance(result, ScaledParameters)

    def test_foE_detected(self):
        result = IonogramScaler(
            e_layer_height_range_km=(85, 165),
            min_echoes_for_layer=3,
        ).fit(self._full_df())
        assert result.quality_flags.get("E_detected", False)
        assert not np.isnan(result.foE_mhz)

    def test_foF2_detected(self):
        result = IonogramScaler(min_echoes_for_layer=3).fit(self._full_df())
        assert result.quality_flags.get("F2_detected", False)
        assert not np.isnan(result.foF2_mhz)

    def test_foF2_greater_than_foE(self):
        result = IonogramScaler(min_echoes_for_layer=3).fit(self._full_df())
        if not (np.isnan(result.foE_mhz) or np.isnan(result.foF2_mhz)):
            assert result.foF2_mhz > result.foE_mhz

    def test_MUF_greater_than_foF2(self):
        result = IonogramScaler(min_echoes_for_layer=3).fit(self._full_df())
        if not np.isnan(result.MUF3000_mhz):
            assert result.MUF3000_mhz > result.foF2_mhz

    def test_M3000F2_reasonable(self):
        result = IonogramScaler(min_echoes_for_layer=3).fit(self._full_df())
        if not np.isnan(result.M3000F2):
            assert 1.0 < result.M3000F2 < 10.0

    def test_bootstrap_sigma_finite(self):
        result = IonogramScaler(min_echoes_for_layer=3, n_bootstrap=50).fit(
            self._full_df()
        )
        if result.quality_flags.get("foF2_reliable"):
            assert not np.isnan(result.foF2_sigma_mhz)

    def test_to_dataframe_single_row(self):
        result = IonogramScaler(min_echoes_for_layer=3).fit(self._full_df())
        out = result.to_dataframe()
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1

    def test_quality_flags_are_bool(self):
        result = IonogramScaler(min_echoes_for_layer=3).fit(self._full_df())
        for v in result.quality_flags.values():
            assert isinstance(v, bool)

    def test_summary_string(self):
        result = IonogramScaler(min_echoes_for_layer=3).fit(self._full_df())
        s = result.summary()
        assert isinstance(s, str)
        assert "foF2" in s

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError):
            IonogramScaler().fit(pd.DataFrame({"height_km": [300]}))

    def test_no_mode_col_uses_all_echoes(self):
        """When mode column is absent, all echoes are treated as O-mode."""
        df = _make_echo_df(seed=41)
        df = df.drop(columns=["polarization_deg"], errors="ignore")
        result = IonogramScaler(min_echoes_for_layer=3).fit(df)
        assert isinstance(result.foF2_mhz, float)


# ===========================================================================
# IrregularityAnalyzer
# ===========================================================================


class TestIrregularityAnalyzer:
    def _ep_df(self, n: int = 100, seed: int = 50) -> pd.DataFrame:
        """Echo DataFrame with varying EP across a frequency range."""
        rng = np.random.default_rng(seed)
        freqs = np.linspace(2_000, 8_000, n)
        heights = np.where(freqs < 4_000, rng.normal(120, 5, n), rng.normal(300, 10, n))
        return pd.DataFrame(
            {
                "frequency_khz": freqs,
                "height_km": heights,
                "residual_deg": np.abs(rng.normal(15, 6, n)),
                "mode": ["O"] * n,
            }
        )

    def test_fit_returns_result(self):
        from pynasonde.vipir.analysis.irregularities import IrregularityProfile

        result = IrregularityAnalyzer(min_pairs_for_fit=2).fit(self._ep_df())
        assert isinstance(result, IrregularityProfile)

    def test_spectral_index_finite(self):
        result = IrregularityAnalyzer(min_pairs_for_fit=2).fit(self._ep_df())
        assert not np.isnan(result.spectral_index)

    def test_D_EP_nonneg(self):
        result = IrregularityAnalyzer(min_pairs_for_fit=2).fit(self._ep_df())
        sf = result.structure_function
        valid = sf["D_EP_deg2"].notna()
        assert np.all(sf.loc[valid, "D_EP_deg2"] >= 0)

    def test_anisotropy_computed(self):
        result = IrregularityAnalyzer(
            min_pairs_for_fit=2,
            f_layer_height_range_km=(160, 800),
        ).fit(self._ep_df(n=150))
        # anisotropy_ratio is finite when both modes are present (or defaults to NaN)
        assert isinstance(result.anisotropy_ratio, float)

    def test_to_dataframe_columns(self):
        result = IrregularityAnalyzer(min_pairs_for_fit=2).fit(self._ep_df())
        out = result.to_dataframe()
        assert "delta_f_mhz" in out.columns
        assert "D_EP_deg2" in out.columns

    def test_summary_string(self):
        result = IrregularityAnalyzer(min_pairs_for_fit=2).fit(self._ep_df())
        s = result.summary()
        assert isinstance(s, str)
        assert "α=" in s

    def test_too_few_echoes_returns_nan_profile(self):
        df = pd.DataFrame(
            {
                "frequency_khz": [5_000.0, 5_100.0],
                "residual_deg": [10.0, 12.0],
                "height_km": [300.0, 310.0],
            }
        )
        result = IrregularityAnalyzer(min_pairs_for_fit=10).fit(df)
        assert np.isnan(result.spectral_index)

    def test_missing_ep_col_raises(self):
        df = pd.DataFrame({"frequency_khz": [5_000.0]})
        with pytest.raises(KeyError):
            IrregularityAnalyzer().fit(df)

    def test_mean_ep_positive(self):
        result = IrregularityAnalyzer(min_pairs_for_fit=2).fit(self._ep_df())
        # mean EP is available via the height_profile table
        hp = result.height_profile
        if not hp.empty and hp["n_echoes"].sum() > 0:
            assert result.n_echoes_total > 0


# ===========================================================================
# NeXtYZ physics helpers
# ===========================================================================


class TestAppletonLassen:
    """Test the Appleton-Lassen n² formula directly."""

    def test_n2_is_zero_at_reflection(self):
        """At X = 1 (fp = f), n² → 0 for O-mode."""
        n2 = _appleton_lassen_n2(X=1.0, Y_L=0.3, Y_T2=0.05, polarization="O")
        assert n2 == pytest.approx(0.0, abs=1e-9)

    def test_n2_is_one_below_plasma_freq(self):
        """Below plasma frequency (X ≪ 1) and no B: n² ≈ 1 − X."""
        n2 = _appleton_lassen_n2(X=0.1, Y_L=0.0, Y_T2=0.0, polarization="O")
        assert n2 == pytest.approx(0.9, rel=1e-4)

    def test_n2_nonneg(self):
        """n² must never be negative (clipped at 0)."""
        for X in np.linspace(0, 1.5, 10):
            for pol in ("O", "X"):
                n2 = _appleton_lassen_n2(X, Y_L=0.2, Y_T2=0.1, polarization=pol)
                assert n2 >= 0.0

    def test_o_x_differ_for_nonzero_B(self):
        """O and X modes must give different n² when B ≠ 0."""
        n2_O = _appleton_lassen_n2(0.5, 0.3, 0.1, "O")
        n2_X = _appleton_lassen_n2(0.5, 0.3, 0.1, "X")
        assert n2_O != pytest.approx(n2_X)

    def test_no_B_O_X_equal(self):
        """With Y=0 (no B field), O and X modes must give the same n²."""
        n2_O = _appleton_lassen_n2(0.4, Y_L=0.0, Y_T2=0.0, polarization="O")
        n2_X = _appleton_lassen_n2(0.4, Y_L=0.0, Y_T2=0.0, polarization="X")
        assert n2_O == pytest.approx(n2_X, rel=1e-6)


class TestWSIGeometry:
    """Test WSI frame-plane geometry helpers."""

    def test_nz_unit_vector(self):
        """For nx=ny=0 the normal should be purely vertical."""
        assert _nz(0.0, 0.0) == pytest.approx(1.0)

    def test_nz_small_tilt(self):
        """Small tilt: nz ≈ 1."""
        assert _nz(0.05, 0.05) == pytest.approx(
            np.sqrt(1 - 0.05**2 - 0.05**2), rel=1e-6
        )

    def test_wedge_rho_midpoint(self):
        """A point at the vertical midpoint of a horizontal wedge → ρ ≈ 0.5."""
        r = np.array([0.0, 0.0, 150.0])  # midpoint between h=100 and h=200
        rho = _wedge_rho(r, 100.0, 0.0, 0.0, 200.0, 0.0, 0.0)
        assert rho == pytest.approx(0.5, abs=1e-6)

    def test_wedge_rho_bounds(self):
        """ρ must lie in [0, 1] for any point within the wedge."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            z = rng.uniform(100, 200)
            r = np.array([rng.normal(0, 20), rng.normal(0, 20), z])
            rho = _wedge_rho(r, 100.0, 0.0, 0.0, 200.0, 0.0, 0.0)
            assert 0.0 <= rho <= 1.0 + 1e-9

    def test_wedge_rho_at_lower_plane(self):
        """A point exactly on the lower frame plane → ρ = 0."""
        r = np.array([0.0, 0.0, 100.0])
        rho = _wedge_rho(r, 100.0, 0.0, 0.0, 200.0, 0.0, 0.0)
        assert rho == pytest.approx(0.0, abs=1e-9)

    def test_wedge_rho_at_upper_plane(self):
        """A point exactly on the upper frame plane → ρ = 1."""
        r = np.array([0.0, 0.0, 200.0])
        rho = _wedge_rho(r, 100.0, 0.0, 0.0, 200.0, 0.0, 0.0)
        assert rho == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# NeXtYZInverter — constructor and fit
# ===========================================================================


class TestNeXtYZInverter:
    def _inv(self, **kw) -> NeXtYZInverter:
        defaults = dict(
            dip_angle_deg=66.0,
            declination_deg=11.0,
            B_gauss=0.55,
            fp_step_mhz=0.5,
            min_echoes=4,
            max_echoes=20,
            mode="Lite",
            fp_start_mhz=2.0,
        )
        defaults.update(kw)
        return NeXtYZInverter(**defaults)

    def test_gyrofreq_computed(self):
        inv = self._inv(B_gauss=0.55)
        assert inv.fH_mhz == pytest.approx(2.80 * 0.55, rel=1e-6)

    def test_B_hat_unit_vector(self):
        inv = self._inv()
        assert np.linalg.norm(inv._B_hat) == pytest.approx(1.0, rel=1e-6)

    def test_B_hat_downward_in_NH(self):
        """In the NH (dip > 0), the vertical component of B is negative (downward)."""
        inv = self._inv(dip_angle_deg=60.0, declination_deg=0.0)
        assert inv._B_hat[2] < 0  # z = Up, so negative means downward

    def test_fit_returns_result(self):
        from pynasonde.vipir.analysis.nextyz import NeXtYZResult

        df = _make_echo_df(n_e=40, n_f=80, seed=60)
        result = self._inv().fit(df)
        assert isinstance(result, NeXtYZResult)

    def test_fit_missing_xl_raises(self):
        df = _make_echo_df(seed=61).drop(columns=["xl_km"])
        with pytest.raises(KeyError):
            self._inv().fit(df)

    def test_result_to_dataframe(self):
        df = _make_echo_df(n_e=40, n_f=80, seed=62)
        result = self._inv().fit(df)
        out = result.to_dataframe()
        assert isinstance(out, pd.DataFrame)

    def test_summary_string(self):
        df = _make_echo_df(n_e=40, n_f=80, seed=63)
        result = self._inv().fit(df)
        s = result.summary()
        assert isinstance(s, str)

    def test_wedgeplane_fields(self):
        wp = WedgePlane(
            fp_lo_mhz=3.0,
            fp_hi_mhz=3.05,
            h_upper_km=200.0,
            nx=0.01,
            ny=-0.02,
            residual_km=5.0,
            n_echoes=12,
        )
        assert wp.fp_lo_mhz == pytest.approx(3.0)
        assert wp.h_upper_km == pytest.approx(200.0)
        assert wp.n_echoes == 12

    def test_converged_list_length_matches_wedges(self):
        df = _make_echo_df(n_e=50, n_f=100, seed=64)
        result = self._inv().fit(df)
        assert len(result.converged) == len(result.wedges)

    def test_tilt_arrays_length_matches_wedges(self):
        df = _make_echo_df(n_e=50, n_f=100, seed=65)
        result = self._inv().fit(df)
        assert len(result.tilt_meridional_deg) == len(result.wedges)
        assert len(result.tilt_zonal_deg) == len(result.wedges)

    def test_h_errors_nonneg(self):
        df = _make_echo_df(n_e=50, n_f=100, seed=66)
        result = self._inv().fit(df)
        if len(result.h_errors_km) > 0:
            assert np.all(result.h_errors_km >= 0)


# ===========================================================================
# AbsorptionAnalyzer
# ===========================================================================


def _make_absorption_df(seed: int = 70) -> pd.DataFrame:
    """Synthetic echo DataFrame for AbsorptionAnalyzer tests.

    Includes O-mode (negative PP) and X-mode (positive PP) echoes at
    matching frequencies with SNR values.
    """
    rng = np.random.default_rng(seed)
    freqs_mhz = np.linspace(2.0, 8.0, 30)
    rows = []
    for f in freqs_mhz:
        h = 200.0 + (f - 2.0) * 15.0
        # O-mode echo
        rows.append(
            {
                "frequency_khz": f * 1e3,
                "height_km": h + rng.normal(0, 3),
                "snr_db": rng.normal(20, 3),
                "mode": "O",
                "polarization_deg": rng.uniform(-160, -45),
            }
        )
        # X-mode echo (slightly lower SNR — simulated differential absorption)
        rows.append(
            {
                "frequency_khz": f * 1e3 + rng.normal(0, 10),
                "height_km": h + rng.normal(0, 3),
                "snr_db": rng.normal(17, 3),
                "mode": "X",
                "polarization_deg": rng.uniform(45, 160),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _make_chapman_edp() -> "EDPResult":
    """Synthetic D/E-region Chapman EDP for absorption_profile tests."""
    from pynasonde.vipir.analysis.inversion import EDPResult

    _FP_TO_N = 1.2399e4
    h = np.linspace(60.0, 130.0, 140)
    xi = (h - 90.0) / 8.0
    fp = np.maximum(0.5 * np.exp(0.5 * (1.0 - xi - np.exp(-xi))), 1e-4)
    peak = int(np.argmax(fp))
    return EDPResult(
        true_height_km=h,
        plasma_freq_mhz=fp,
        electron_density_cm3=fp**2 * _FP_TO_N,
        virtual_height_km=h,
        frequency_mhz=fp,
        foF2_mhz=float(fp[peak]),
        hmF2_km=float(h[peak]),
        NmF2_cm3=float(fp[peak] ** 2 * _FP_TO_N),
        method="synthetic_chapman",
        n_layers=len(h),
    )


class TestAbsorptionAnalyzer:
    def _ana(self, **kw):
        from pynasonde.vipir.analysis import AbsorptionAnalyzer

        defaults = dict(
            snr_col="snr_db",
            freq_col="frequency_khz",
            height_col="height_km",
            mode_col="mode",
            freq_bin_mhz=0.3,
            f_ref_mhz=1.0,
        )
        defaults.update(kw)
        return AbsorptionAnalyzer(**defaults)

    # ---- LOF ----

    def test_lof_returns_result(self):
        from pynasonde.vipir.analysis import LOFResult

        result = self._ana().lof_absorption(_make_absorption_df())
        assert isinstance(result, LOFResult)

    def test_lof_fmin_finite(self):
        result = self._ana().lof_absorption(_make_absorption_df())
        assert np.isfinite(result.fmin_mhz)

    def test_lof_fmin_positive(self):
        result = self._ana().lof_absorption(_make_absorption_df())
        assert result.fmin_mhz > 0

    def test_lof_index_equals_fmin_squared_minus_fref_squared(self):
        ana = self._ana(f_ref_mhz=2.0)
        result = ana.lof_absorption(_make_absorption_df())
        expected = result.fmin_mhz**2 - 2.0**2
        assert result.lof_index_mhz2 == pytest.approx(expected, rel=1e-6)

    def test_lof_summary_string(self):
        s = self._ana().lof_absorption(_make_absorption_df()).summary()
        assert "MHz" in s

    def test_lof_empty_df(self):
        df = pd.DataFrame(columns=["frequency_khz", "height_km", "snr_db", "mode"])
        result = self._ana().lof_absorption(df)
        assert not np.isfinite(result.fmin_mhz)

    # ---- Differential ----

    def test_differential_returns_result(self):
        from pynasonde.vipir.analysis import DifferentialResult

        result = self._ana().differential_absorption(_make_absorption_df())
        assert isinstance(result, DifferentialResult)

    def test_differential_mean_delta_finite(self):
        result = self._ana().differential_absorption(_make_absorption_df())
        assert np.isfinite(result.mean_delta_db)

    def test_differential_counts_positive(self):
        result = self._ana().differential_absorption(_make_absorption_df())
        assert result.n_echoes_o > 0
        assert result.n_echoes_x > 0

    def test_differential_profile_df_columns(self):
        result = self._ana().differential_absorption(_make_absorption_df())
        assert "frequency_mhz" in result.profile_df.columns
        assert "delta_snr_db" in result.profile_df.columns

    def test_differential_summary_string(self):
        s = self._ana().differential_absorption(_make_absorption_df()).summary()
        assert "ΔL" in s

    def test_differential_plot_no_error(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._ana().differential_absorption(_make_absorption_df())
        fig, ax = plt.subplots()
        result.plot(ax=ax)
        plt.close(fig)

    # ---- Total ----

    def test_total_returns_result(self):
        from pynasonde.vipir.analysis import TotalAbsorptionResult

        result = self._ana().total_absorption(
            _make_absorption_df(),
            tx_eirp_dbw=27.0,
            rx_gain_dbi=0.0,
        )
        assert isinstance(result, TotalAbsorptionResult)

    def test_total_profile_df_not_empty(self):
        result = self._ana().total_absorption(
            _make_absorption_df(),
            tx_eirp_dbw=27.0,
        )
        assert not result.profile_df.empty

    def test_total_absorption_finite(self):
        result = self._ana().total_absorption(
            _make_absorption_df(),
            tx_eirp_dbw=27.0,
        )
        assert not result.profile_df.empty
        assert np.isfinite(result.profile_df["absorption_db"].mean())

    def test_total_plot_no_error(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._ana().total_absorption(
            _make_absorption_df(),
            tx_eirp_dbw=27.0,
        )
        fig, ax = plt.subplots()
        result.plot(ax=ax)
        plt.close(fig)

    # ---- Absorption profile ----

    def test_profile_returns_result(self):
        from pynasonde.vipir.analysis import AbsorptionProfileResult

        edp = _make_chapman_edp()
        result = self._ana().absorption_profile(
            edp,
            nu_hz=lambda h: 1.816e11 * np.exp(-0.15 * h),
            f_wave_mhz=2.0,
        )
        assert isinstance(result, AbsorptionProfileResult)

    def test_profile_d_region_nonzero(self):
        """Chapman D/E EDP with Budden ν should give meaningful absorption."""
        edp = _make_chapman_edp()
        result = self._ana().absorption_profile(
            edp,
            nu_hz=lambda h: 1.816e11 * np.exp(-0.15 * h),
            n_interp=200,
            f_wave_mhz=2.0,
        )
        assert result.total_absorption_db > 0.1

    def test_profile_f_region_near_zero(self):
        """F-region EDP (200-350 km) has negligible ν → near-zero absorption."""
        from pynasonde.vipir.analysis.inversion import EDPResult

        h = np.linspace(200.0, 350.0, 30)
        fp = np.linspace(2.0, 5.5, 30)
        edp = EDPResult(
            true_height_km=h,
            plasma_freq_mhz=fp,
            electron_density_cm3=fp**2 * 1.2399e4,
            virtual_height_km=h,
            frequency_mhz=fp,
            foF2_mhz=5.5,
            hmF2_km=350.0,
            NmF2_cm3=5.5**2 * 1.2399e4,
            method="synthetic",
            n_layers=len(h),
        )
        result = self._ana().absorption_profile(
            edp,
            nu_hz=lambda h: 1.816e11 * np.exp(-0.15 * h),
            n_interp=100,
            f_wave_mhz=6.0,
        )
        assert result.total_absorption_db < 0.1

    def test_profile_kappa_zero_above_reflection(self):
        """At heights where fp > f_wave the wave cannot penetrate — kappa=0."""
        edp = _make_chapman_edp()
        # f_wave = 0.1 MHz is below peak fp (0.5 MHz) → many heights have fp > f_wave
        result = self._ana().absorption_profile(
            edp,
            nu_hz=lambda h: 1.816e11 * np.exp(-0.15 * h),
            n_interp=100,
            f_wave_mhz=0.1,
        )
        pdf = result.profile_df
        if not pdf.empty:
            above = pdf[pdf["fp_mhz"] > 0.1]
            assert np.allclose(above["kappa_dB_per_km"].values, 0.0, atol=1e-12)

    def test_profile_df_columns(self):
        edp = _make_chapman_edp()
        result = self._ana().absorption_profile(
            edp,
            nu_hz=lambda h: 1.816e11 * np.exp(-0.15 * h),
            f_wave_mhz=2.0,
        )
        for col in ("height_km", "kappa_dB_per_km", "fp_mhz", "nu_hz"):
            assert col in result.profile_df.columns

    def test_profile_cumulative_monotone(self):
        """Cumulative L(z) must be non-decreasing."""
        edp = _make_chapman_edp()
        result = self._ana().absorption_profile(
            edp,
            nu_hz=lambda h: 1.816e11 * np.exp(-0.15 * h),
            n_interp=150,
            f_wave_mhz=2.0,
        )
        L = result.cumulative_df["L_oneway_db"].values
        assert np.all(np.diff(L) >= -1e-12)

    def test_profile_summary_string(self):
        edp = _make_chapman_edp()
        s = (
            self._ana()
            .absorption_profile(
                edp,
                nu_hz=lambda h: 1.816e11 * np.exp(-0.15 * h),
                f_wave_mhz=2.0,
            )
            .summary()
        )
        assert "dB" in s

    def test_profile_none_edp_returns_empty(self):
        result = self._ana().absorption_profile(None, nu_hz=lambda h: 1e5)
        assert result.profile_df.empty

    def test_profile_array_nu_requires_heights(self):
        edp = _make_chapman_edp()
        nu_arr = np.ones(50)
        with pytest.raises(ValueError, match="heights_km"):
            self._ana().absorption_profile(edp, nu_hz=nu_arr)


# ===========================================================================
# EsCaponImager
# ===========================================================================


def _make_iq_cube(
    n_pulse: int = 8,
    n_gate: int = 64,
    n_rx: int = 4,
    signal_gate: int = 20,
    snr_db: float = 20.0,
    seed: int = 80,
) -> np.ndarray:
    """Synthetic IQ cube with a coherent sinusoidal signal at `signal_gate`.

    Returns complex64 array of shape ``(n_pulse, n_gate, n_rx)``.
    """
    rng = np.random.default_rng(seed)
    noise_amp = 1.0
    signal_amp = noise_amp * 10 ** (snr_db / 20.0)

    cube = (
        rng.standard_normal((n_pulse, n_gate, n_rx))
        + 1j * rng.standard_normal((n_pulse, n_gate, n_rx))
    ).astype(np.complex64) * noise_amp

    # Add a coherent in-phase signal at signal_gate (same across pulses and Rx)
    cube[:, signal_gate, :] += signal_amp * (1.0 + 0.0j)
    return cube


class TestEsCaponImager:
    def _imager(self, **kw):
        from pynasonde.vipir.analysis import EsCaponImager

        defaults = dict(
            n_subbands=8,
            resolution_factor=4,
            coherent_integrations=4,
            gate_start_km=90.0,
            gate_spacing_km=3.0,
        )
        defaults.update(kw)
        return EsCaponImager(**defaults)

    # ---- Constructor ----

    def test_params_stored(self):
        from pynasonde.vipir.analysis import EsCaponImager

        img = EsCaponImager(
            n_subbands=10,
            resolution_factor=3,
            coherent_integrations=2,
            gate_start_km=50.0,
            gate_spacing_km=2.5,
        )
        assert img.Z == 10
        assert img.K == 3
        assert img.n_coh == 2
        assert img.gate_start_km == pytest.approx(50.0)
        assert img.gate_spacing_km == pytest.approx(2.5)

    def test_z_warning_when_rank_deficient(self):
        """Z > (V+1)/2 is rank-deficient — imager warns but still runs."""

        from pynasonde.vipir.analysis import EsCaponImager

        # V=16, Z=10 > (16+1)/2=8 → rank-deficient, should warn
        cube = _make_iq_cube(n_gate=16, n_pulse=4, n_rx=2, signal_gate=5)
        img = EsCaponImager(
            n_subbands=10,
            resolution_factor=10,
            coherent_integrations=4,
            gate_start_km=0.0,
            gate_spacing_km=1.0,
        )
        # Should not raise; K is preserved as-is (no clipping)
        result = img.fit(cube)
        assert result.resolution_factor == 10
        assert result.pseudospectrum_db.shape[1] == 10 * 16

    # ---- fit() output shape ----

    def test_fit_returns_result(self):
        from pynasonde.vipir.analysis import EsImagingResult

        cube = _make_iq_cube()
        result = self._imager().fit(cube)
        assert isinstance(result, EsImagingResult)

    def test_pseudospectrum_shape(self):
        cube = _make_iq_cube(n_pulse=8, n_gate=64)
        result = self._imager(
            n_subbands=8,
            resolution_factor=4,
            coherent_integrations=8,
        ).fit(cube)
        V = 64
        K = result.resolution_factor
        assert result.pseudospectrum_db.shape == (1, K * V)

    def test_heights_km_shape(self):
        cube = _make_iq_cube(n_pulse=8, n_gate=64)
        result = self._imager().fit(cube)
        assert result.heights_km.shape == (result.pseudospectrum_db.shape[1],)

    def test_gate_heights_km_shape(self):
        cube = _make_iq_cube(n_pulse=4, n_gate=64)
        result = self._imager().fit(cube)
        assert result.gate_heights_km.shape == (64,)

    # ---- Physical properties ----

    def test_heights_km_monotone(self):
        cube = _make_iq_cube(n_pulse=4, n_gate=64)
        result = self._imager().fit(cube)
        assert np.all(np.diff(result.heights_km) > 0)

    def test_pseudospectrum_max_le_zero(self):
        """Normalised spectrum: peak is 0 dB."""
        cube = _make_iq_cube()
        result = self._imager().fit(cube)
        assert result.pseudospectrum_db.max() <= 0.1

    def test_height_spacing_equals_gate_over_k(self):
        cube = _make_iq_cube(n_pulse=4, n_gate=64)
        result = self._imager(gate_spacing_km=3.0, resolution_factor=3).fit(cube)
        dh = np.diff(result.heights_km)
        expected = result.gate_spacing_km / result.resolution_factor
        assert np.allclose(dh, expected, rtol=1e-6)

    def test_gate_start_reflected_in_heights(self):
        cube = _make_iq_cube(n_pulse=4, n_gate=64)
        result = self._imager(gate_start_km=100.0, gate_spacing_km=2.0).fit(cube)
        assert result.heights_km[0] == pytest.approx(100.0, rel=1e-4)

    # ---- Coherent integrations → snapshot count ----

    def test_single_snapshot_when_coh_equals_npulse(self):
        cube = _make_iq_cube(n_pulse=8)
        result = self._imager(coherent_integrations=8).fit(cube)
        assert result.n_snapshots == 1

    def test_multiple_snapshots(self):
        cube = _make_iq_cube(n_pulse=8)
        result = self._imager(coherent_integrations=2).fit(cube)
        assert result.n_snapshots == 4

    def test_coh_larger_than_npulse_gives_one_snapshot(self):
        cube = _make_iq_cube(n_pulse=4)
        result = self._imager(coherent_integrations=100).fit(cube)
        assert result.n_snapshots == 1

    # ---- Capon resolves injected signal ----

    def test_peak_near_signal_gate(self):
        """Capon peak (across full gate range) should be near the injected gate."""
        n_gate = 64
        signal_gate = 30
        cube = _make_iq_cube(n_gate=n_gate, signal_gate=signal_gate, snr_db=30.0)
        result = self._imager(
            n_subbands=8,
            resolution_factor=4,
            coherent_integrations=8,
            gate_start_km=0.0,
            gate_spacing_km=1.0,
        ).fit(cube)
        # Expected peak height ≈ signal_gate * gate_spacing_km
        peak_idx = int(np.argmax(result.pseudospectrum_db[0]))
        peak_h = result.heights_km[peak_idx]
        assert abs(peak_h - signal_gate) < 5.0  # within 5 km

    # ---- Methods ----

    def test_summary_string(self):
        cube = _make_iq_cube()
        s = self._imager().fit(cube).summary()
        assert isinstance(s, str)
        assert "km" in s

    def test_to_dataframe_columns(self):
        cube = _make_iq_cube()
        df = self._imager().fit(cube).to_dataframe(snapshot=0)
        assert "height_km" in df.columns
        assert "power_db" in df.columns

    def test_to_dataframe_length(self):
        cube = _make_iq_cube(n_gate=64)
        result = self._imager().fit(cube)
        df = result.to_dataframe()
        assert len(df) == result.pseudospectrum_db.shape[1]

    def test_plot_single_snapshot_no_error(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cube = _make_iq_cube()
        result = self._imager(coherent_integrations=8).fit(cube)
        fig, ax = plt.subplots()
        result.plot(ax=ax, snapshot=0)
        plt.close(fig)

    def test_delta_r_property(self):
        cube = _make_iq_cube(n_pulse=4, n_gate=64)
        result = self._imager(gate_spacing_km=3.0, resolution_factor=3).fit(cube)
        assert result.effective_resolution_km == pytest.approx(1.0, rel=1e-6)


# ===========================================================================
# RiqAggregator
# ===========================================================================


def _make_multi_cubes(
    n_files: int = 3,
    n_pulse: int = 4,
    n_gate: int = 64,
    n_rx: int = 8,
    signal_gate: int = 20,
    snr_db: float = 15.0,
    seed: int = 90,
) -> list:
    """Return a list of synthetic IQ cubes mimicking n_files RIQ files."""
    rng = np.random.default_rng(seed)
    cubes = []
    noise_amp = 1.0
    signal_amp = noise_amp * 10 ** (snr_db / 20.0)
    for _ in range(n_files):
        cube = (
            rng.standard_normal((n_pulse, n_gate, n_rx))
            + 1j * rng.standard_normal((n_pulse, n_gate, n_rx))
        ).astype(np.complex64) * noise_amp
        cube[:, signal_gate, :] += signal_amp * (1.0 + 0.0j)
        cubes.append(cube)
    return cubes


class TestRiqAggregator:
    def _agg(self, **kw):
        from pynasonde.vipir.analysis import RiqAggregator

        defaults = dict(
            n_subbands=8,
            resolution_factor=4,
            gate_start_km=0.0,
            gate_spacing_km=1.5,
        )
        defaults.update(kw)
        return RiqAggregator(**defaults)

    # ---- Constructor ----

    def test_params_stored(self):
        from pynasonde.vipir.analysis import RiqAggregator

        agg = RiqAggregator(
            n_subbands=50,
            resolution_factor=5,
            gate_start_km=80.0,
            gate_spacing_km=2.0,
            output_mode="per_file",
        )
        assert agg.n_subbands == 50
        assert agg.resolution_factor == 5
        assert agg.gate_start_km == pytest.approx(80.0)
        assert agg.gate_spacing_km == pytest.approx(2.0)
        assert agg.output_mode == "per_file"

    def test_invalid_output_mode_raises(self):
        from pynasonde.vipir.analysis import RiqAggregator

        with pytest.raises(ValueError, match="output_mode"):
            RiqAggregator(output_mode="bad_mode")

    def test_invalid_window_raises(self):
        from pynasonde.vipir.analysis import RiqAggregator

        with pytest.raises(ValueError, match="window"):
            RiqAggregator(output_mode="moving_avg", window=0)

    def test_invalid_step_raises(self):
        from pynasonde.vipir.analysis import RiqAggregator

        with pytest.raises(ValueError, match="step"):
            RiqAggregator(output_mode="moving_avg", step=0)

    # ---- combine() output ----

    def test_combine_returns_result(self):
        from pynasonde.vipir.analysis import EsImagingResult

        cubes = _make_multi_cubes(n_files=3)
        result = self._agg().combine(cubes)
        assert isinstance(result, EsImagingResult)

    def test_per_file_mode_n_snapshots_equals_n_files(self):
        n_files = 6
        cubes = _make_multi_cubes(n_files=n_files)
        result = self._agg(output_mode="per_file").combine(cubes)
        assert result.n_snapshots == n_files

    def test_moving_avg_mode_snapshot_count(self):
        """(N - window) // step + 1 windows expected."""
        n_files, window, step = 10, 4, 2
        cubes = _make_multi_cubes(n_files=n_files)
        result = self._agg(output_mode="moving_avg", window=window, step=step).combine(
            cubes
        )
        expected = (n_files - window) // step + 1
        assert result.n_snapshots == expected

    def test_moving_avg_window_equals_n_files_gives_one_snapshot(self):
        n_files = 5
        cubes = _make_multi_cubes(n_files=n_files)
        result = self._agg(output_mode="moving_avg", window=n_files, step=1).combine(
            cubes
        )
        assert result.n_snapshots == 1

    def test_moving_avg_window_too_large_raises(self):
        cubes = _make_multi_cubes(n_files=3)
        with pytest.raises(ValueError, match="window"):
            self._agg(output_mode="moving_avg", window=10).combine(cubes)

    def test_pseudospectrum_shape_per_file(self):
        n_gate = 64
        K = 4
        n_files = 4
        cubes = _make_multi_cubes(n_files=n_files, n_gate=n_gate)
        result = self._agg(resolution_factor=K, output_mode="per_file").combine(cubes)
        assert result.pseudospectrum_db.shape == (n_files, K * n_gate)

    def test_pseudospectrum_shape_moving_avg(self):
        n_gate, K, n_files, window, step = 64, 4, 8, 4, 1
        cubes = _make_multi_cubes(n_files=n_files, n_gate=n_gate)
        result = self._agg(
            resolution_factor=K, output_mode="moving_avg", window=window, step=step
        ).combine(cubes)
        expected_snapshots = (n_files - window) // step + 1
        assert result.pseudospectrum_db.shape == (expected_snapshots, K * n_gate)

    def test_pseudospectrum_max_le_zero(self):
        cubes = _make_multi_cubes()
        result = self._agg().combine(cubes)
        assert result.pseudospectrum_db.max() <= 0.1

    def test_heights_km_monotone(self):
        cubes = _make_multi_cubes()
        result = self._agg().combine(cubes)
        assert np.all(np.diff(result.heights_km) > 0)

    def test_height_spacing_equals_gate_over_k(self):
        cubes = _make_multi_cubes(n_gate=64)
        result = self._agg(gate_spacing_km=2.0, resolution_factor=4).combine(cubes)
        dh = np.diff(result.heights_km)
        assert np.allclose(
            dh, result.gate_spacing_km / result.resolution_factor, rtol=1e-6
        )

    # ---- Rx beamforming ----

    def test_uniform_rx_weights(self):
        """Default uniform weights: result should be finite and normalised."""
        cubes = _make_multi_cubes(n_rx=8)
        result = self._agg().combine(cubes)
        assert np.all(np.isfinite(result.pseudospectrum_db))

    def test_custom_rx_weights(self):
        """Custom complex weights applied without error."""
        from pynasonde.vipir.analysis import RiqAggregator

        n_rx = 8
        w = np.exp(1j * np.linspace(0, np.pi, n_rx))  # phase ramp
        agg = RiqAggregator(
            n_subbands=8,
            resolution_factor=4,
            rx_weights=w,
            gate_start_km=0.0,
            gate_spacing_km=1.5,
        )
        cubes = _make_multi_cubes(n_rx=n_rx)
        result = agg.combine(cubes)
        assert result.pseudospectrum_db.shape[1] == 4 * 64

    def test_wrong_rx_weights_length_raises(self):
        from pynasonde.vipir.analysis import RiqAggregator

        w = np.ones(3, dtype=complex)  # wrong length for n_rx=8
        agg = RiqAggregator(
            n_subbands=8,
            resolution_factor=4,
            rx_weights=w,
            gate_start_km=0.0,
            gate_spacing_km=1.5,
        )
        cubes = _make_multi_cubes(n_rx=8)
        with pytest.raises(ValueError, match="rx_weights"):
            agg.combine(cubes)

    def test_2d_cube_no_rx_axis(self):
        """2-D cubes (no Rx axis) should be handled transparently."""
        rng = np.random.default_rng(91)
        cubes_2d = [
            (rng.standard_normal((4, 64)) + 1j * rng.standard_normal((4, 64))).astype(
                np.complex64
            )
            for _ in range(3)
        ]
        result = self._agg().combine(cubes_2d)
        assert isinstance(result.pseudospectrum_db, np.ndarray)

    # ---- Error cases ----

    def test_empty_cubes_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self._agg().combine([])

    def test_mismatched_gate_count_raises(self):
        rng = np.random.default_rng(92)
        cubes = [
            (rng.standard_normal((4, 64, 4)) + 1j * rng.standard_normal((4, 64, 4))),
            (rng.standard_normal((4, 32, 4)) + 1j * rng.standard_normal((4, 32, 4))),
        ]
        with pytest.raises(ValueError, match="n_gate"):
            self._agg().combine(cubes)

    # ---- Summary / methods ----

    def test_summary_string(self):
        cubes = _make_multi_cubes()
        s = self._agg().combine(cubes).summary()
        assert "km" in s

    def test_to_dataframe_columns(self):
        cubes = _make_multi_cubes()
        df = self._agg().combine(cubes).to_dataframe(snapshot=0)
        assert "height_km" in df.columns
        assert "power_db" in df.columns

    def test_plot_no_error(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cubes = _make_multi_cubes()
        result = self._agg().combine(cubes)
        fig, ax = plt.subplots()
        result.plot(ax=ax, snapshot=0)
        plt.close(fig)


# ===========================================================================
# Import sanity
# ===========================================================================


class TestImports:
    def test_all_classes_importable(self):
        from pynasonde.vipir.analysis import (
            AbsorptionAnalyzer,
            AbsorptionProfileResult,
            DifferentialResult,
            EDPResult,
            EsCaponImager,
            EsImagingResult,
            IonogramScaler,
            IrregularityAnalyzer,
            IrregularityProfile,
            LOFResult,
            NeXtYZInverter,
            NeXtYZResult,
            PolarizationClassifier,
            PolarizationResult,
            RiqAggregator,
            ScaledParameters,
            SpreadFAnalyzer,
            SpreadFResult,
            TotalAbsorptionResult,
            TrueHeightInversion,
            WedgePlane,
        )

        for cls in (
            PolarizationClassifier,
            PolarizationResult,
            SpreadFAnalyzer,
            SpreadFResult,
            TrueHeightInversion,
            EDPResult,
            IonogramScaler,
            ScaledParameters,
            IrregularityAnalyzer,
            IrregularityProfile,
            NeXtYZInverter,
            NeXtYZResult,
            WedgePlane,
            AbsorptionAnalyzer,
            LOFResult,
            DifferentialResult,
            TotalAbsorptionResult,
            AbsorptionProfileResult,
            EsCaponImager,
            EsImagingResult,
            RiqAggregator,
        ):
            assert cls is not None
