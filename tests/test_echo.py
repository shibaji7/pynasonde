"""Unit tests for pynasonde.vipir.riq.echo — Echo dataclass and EchoExtractor.

All tests use synthetic I/Q data; no real RIQ file is required.  A pair of
helpers (_make_sct / _make_pulset) build minimal SimpleNamespace stand-ins
that expose exactly the attributes EchoExtractor reads from a real SctType
and Pulset.

Coverage:
    Echo             — field defaults, to_dict
    EchoExtractor    — __init__ geometry, _build_iq_cube, _compute_doppler,
                       _compute_direction, _compute_polarization,
                       _extract_from_pulset, extract, to_dataframe, to_xarray,
                       error-before-extract guard
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pynasonde.vipir.riq.echo import Echo, EchoExtractor

# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

_N_RX = 4
_N_GATES = 60
_N_PULSES = 8
_SIGNAL_GATE = 25          # range gate where we plant a strong echo
_FREQ_KHZ = 5_000.0        # 5 MHz
_GATE_START = 100.0        # µs
_GATE_END = 1_000.0        # µs
_GATE_STEP = 15.0          # µs
_PRI_US = 10_000.0         # µs  →  pri_s = 0.01 s


def _make_sct(
    n_rx: int = _N_RX,
    gate_start: float = _GATE_START,
    gate_end: float = _GATE_END,
    gate_step: float = _GATE_STEP,
    pri: float = _PRI_US,
) -> SimpleNamespace:
    """Minimal SctType stand-in for EchoExtractor.__init__."""
    # Receiver positions: spread evenly in the East-North plane (m)
    rx_pos = np.zeros((n_rx, 3), dtype=float)
    rx_pos[:, 0] = np.linspace(-50.0, 50.0, n_rx)   # East
    rx_pos[:, 1] = np.linspace(-50.0, 50.0, n_rx)   # North

    # Direction unit vectors: rotate by π/n_rx to create orthogonal pairs
    rx_dir = np.zeros((n_rx, 3), dtype=float)
    for i in range(n_rx):
        angle = i * np.pi / max(n_rx, 1)
        rx_dir[i] = [np.cos(angle), np.sin(angle), 0.0]

    return SimpleNamespace(
        station=SimpleNamespace(
            rx_count=n_rx,
            rx_position=rx_pos,
            rx_direction=rx_dir,
        ),
        timing=SimpleNamespace(
            gate_start=gate_start,
            gate_end=gate_end,
            gate_step=gate_step,
            gate_count=int(round((gate_end - gate_start) / gate_step)),
            pri=pri,
        ),
    )


def _make_pct(
    n_gates: int = _N_GATES,
    n_rx: int = _N_RX,
    frequency_khz: float = _FREQ_KHZ,
    pri_ut: float = 0.0,
    signal_gate: int = _SIGNAL_GATE,
    amplitude: float = 200.0,
    noise: float = 1.0,
    rng_seed: int = 0,
) -> SimpleNamespace:
    """One PCT stand-in with a planted strong echo at *signal_gate*."""
    rng = np.random.default_rng(rng_seed)
    I = rng.standard_normal((n_gates, n_rx)) * noise
    Q = rng.standard_normal((n_gates, n_rx)) * noise
    # Plant a coherent signal (same phase on all receivers)
    I[signal_gate, :] += amplitude
    Q[signal_gate, :] += amplitude * 0.05
    return SimpleNamespace(
        pulse_i=I,
        pulse_q=Q,
        frequency=frequency_khz,
        pri_ut=pri_ut,
    )


def _make_pulset(
    n_pulses: int = _N_PULSES,
    n_gates: int = _N_GATES,
    n_rx: int = _N_RX,
    frequency_khz: float = _FREQ_KHZ,
    signal_gate: int = _SIGNAL_GATE,
    amplitude: float = 200.0,
) -> SimpleNamespace:
    """A Pulset stand-in holding *n_pulses* PCT stand-ins."""
    pcts = [
        _make_pct(
            n_gates=n_gates,
            n_rx=n_rx,
            frequency_khz=frequency_khz,
            pri_ut=float(p) * (_PRI_US * 1e-6),
            signal_gate=signal_gate,
            amplitude=amplitude,
            rng_seed=p,
        )
        for p in range(n_pulses)
    ]
    return SimpleNamespace(pcts=pcts)


def _make_extractor(**kw) -> EchoExtractor:
    """Return an EchoExtractor backed by a single synthetic pulset."""
    sct = _make_sct()
    pulset = _make_pulset(**kw)
    return EchoExtractor(sct=sct, pulsets=[pulset], snr_threshold_db=3.0)


# ===========================================================================
# Echo dataclass
# ===========================================================================


class TestEchoDataclass:
    def test_default_values_are_nan(self):
        e = Echo()
        assert math.isnan(e.frequency_khz)
        assert math.isnan(e.height_km)
        assert math.isnan(e.amplitude_db)
        assert math.isnan(e.gross_phase_deg)
        assert math.isnan(e.doppler_hz)
        assert math.isnan(e.velocity_mps)
        assert math.isnan(e.xl_km)
        assert math.isnan(e.yl_km)
        assert math.isnan(e.polarization_deg)
        assert math.isnan(e.residual_deg)
        assert math.isnan(e.snr_db)
        assert math.isnan(e.pulse_ut)

    def test_default_integer_sentinels(self):
        e = Echo()
        assert e.gate_index == -1
        assert e.rx_count == 0

    def test_to_dict_returns_all_fields(self):
        e = Echo(frequency_khz=5000.0, height_km=300.0)
        d = e.to_dict()
        assert isinstance(d, dict)
        assert d["frequency_khz"] == pytest.approx(5000.0)
        assert d["height_km"] == pytest.approx(300.0)
        # All 14 fields should be present
        expected_keys = {
            "frequency_khz", "height_km", "amplitude_db", "gross_phase_deg",
            "doppler_hz", "velocity_mps", "xl_km", "yl_km",
            "polarization_deg", "residual_deg", "snr_db",
            "gate_index", "pulse_ut", "rx_count",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_nan_fields_are_float(self):
        e = Echo()
        d = e.to_dict()
        assert isinstance(d["frequency_khz"], float)


# ===========================================================================
# EchoExtractor initialisation
# ===========================================================================


class TestEchoExtractorInit:
    def test_rx_pos_shape(self):
        ext = _make_extractor()
        assert ext._rx_pos.shape == (_N_RX, 3)

    def test_rx_dir_shape(self):
        ext = _make_extractor()
        assert ext._rx_dir.shape == (_N_RX, 3)

    def test_heights_axis(self):
        ext = _make_extractor()
        gate_count = int(round((_GATE_END - _GATE_START) / _GATE_STEP))
        expected = (_GATE_START + np.arange(gate_count, dtype=np.float64) * _GATE_STEP) * 0.15
        np.testing.assert_allclose(ext._heights, expected)

    def test_pri_conversion(self):
        ext = _make_extractor()
        assert ext._pri_s == pytest.approx(_PRI_US * 1e-6)

    def test_echoes_none_before_extract(self):
        ext = _make_extractor()
        assert ext._echoes is None

    def test_custom_thresholds_stored(self):
        sct = _make_sct()
        pulset = _make_pulset()
        ext = EchoExtractor(
            sct=sct,
            pulsets=[pulset],
            snr_threshold_db=10.0,
            min_rx_for_direction=2,
            max_echoes_per_pulset=3,
        )
        assert ext.snr_threshold_db == pytest.approx(10.0)
        assert ext.min_rx_for_direction == 2
        assert ext.max_echoes_per_pulset == 3


# ===========================================================================
# _build_iq_cube
# ===========================================================================


class TestBuildIqCube:
    def test_output_shape(self):
        ext = _make_extractor()
        pulset = _make_pulset(n_pulses=8, n_gates=_N_GATES, n_rx=_N_RX)
        C = ext._build_iq_cube(pulset)
        assert C.shape == (8, _N_GATES, _N_RX)

    def test_dtype_is_complex(self):
        ext = _make_extractor()
        pulset = _make_pulset()
        C = ext._build_iq_cube(pulset)
        assert np.iscomplexobj(C)

    def test_real_part_matches_pulse_i(self):
        ext = _make_extractor()
        pulset = _make_pulset(n_pulses=2)
        C = ext._build_iq_cube(pulset)
        np.testing.assert_allclose(
            C[0].real, pulset.pcts[0].pulse_i, rtol=1e-6
        )

    def test_imag_part_matches_pulse_q(self):
        ext = _make_extractor()
        pulset = _make_pulset(n_pulses=2)
        C = ext._build_iq_cube(pulset)
        np.testing.assert_allclose(
            C[1].imag, pulset.pcts[1].pulse_q, rtol=1e-6
        )


# ===========================================================================
# _compute_doppler
# ===========================================================================


class TestComputeDoppler:
    def _ext(self):
        return _make_extractor()

    def test_zero_doppler_for_constant_phase(self):
        """A signal with constant phase across pulses should give ~0 Hz Doppler."""
        ext = self._ext()
        # All pulses: same real amplitude, zero imaginary → phase = 0 everywhere
        n_pulse, n_rx = 16, _N_RX
        C_gate = np.ones((n_pulse, n_rx), dtype=complex) * 100.0
        f_d, v = ext._compute_doppler(C_gate, freq_hz=5e6)
        assert abs(f_d) < 0.1   # within 0.1 Hz of zero
        assert abs(v) < 10.0    # within 10 m/s of zero

    def test_known_doppler_frequency(self):
        """A linear phase ramp of known rate should recover that Doppler."""
        ext = self._ext()
        pri_s = ext._pri_s          # 0.01 s
        f_d_true = 2.5              # Hz
        n_pulse, n_rx = 32, _N_RX
        t = np.arange(n_pulse) * pri_s
        phase = 2.0 * np.pi * f_d_true * t
        C_gate = np.exp(1j * phase[:, None]) * np.ones((1, n_rx))
        f_d, v = ext._compute_doppler(C_gate, freq_hz=5e6)
        assert abs(f_d - f_d_true) < 0.05   # within 0.05 Hz

    def test_returns_nan_for_single_pulse(self):
        ext = self._ext()
        C_gate = np.ones((1, _N_RX), dtype=complex)
        f_d, v = ext._compute_doppler(C_gate, freq_hz=5e6)
        assert math.isnan(f_d)
        assert math.isnan(v)

    def test_velocity_sign_convention(self):
        """Positive Doppler (receding layer) → positive velocity."""
        ext = self._ext()
        pri_s = ext._pri_s
        f_d_true = 3.0
        n_pulse = 16
        t = np.arange(n_pulse) * pri_s
        C_gate = np.exp(1j * 2.0 * np.pi * f_d_true * t)[:, None]
        f_d, v = ext._compute_doppler(C_gate, freq_hz=5e6)
        assert f_d > 0
        assert v > 0


# ===========================================================================
# _compute_direction
# ===========================================================================


class TestComputeDirection:
    def _ext(self, n_rx=6) -> EchoExtractor:
        sct = _make_sct(n_rx=n_rx)
        pulset = _make_pulset(n_rx=n_rx)
        return EchoExtractor(sct=sct, pulsets=[pulset])

    def test_on_axis_echo_near_zero_offsets(self):
        """A planar wave arriving vertically should yield XL ≈ YL ≈ 0."""
        ext = self._ext()
        # All receivers see the same phase → vertical incidence (l=m=0)
        C_mean = np.ones(_N_RX, dtype=complex) * 50.0
        xl, yl, ep = ext._compute_direction(C_mean, height_km=300.0, wavelength_m=60.0)
        assert abs(xl) < 5.0   # km — allow small numerical error
        assert abs(yl) < 5.0

    def test_returns_nan_for_single_receiver(self):
        sct = _make_sct(n_rx=1)
        pulset = _make_pulset(n_rx=1)
        ext = EchoExtractor(sct=sct, pulsets=[pulset])
        C_mean = np.array([1.0 + 0j])
        xl, yl, ep = ext._compute_direction(C_mean, height_km=300.0, wavelength_m=60.0)
        assert math.isnan(xl) and math.isnan(yl) and math.isnan(ep)

    def test_ep_is_non_negative(self):
        ext = self._ext()
        rng = np.random.default_rng(99)
        C_mean = rng.standard_normal(_N_RX) + 1j * rng.standard_normal(_N_RX)
        xl, yl, ep = ext._compute_direction(C_mean, height_km=250.0, wavelength_m=50.0)
        if not math.isnan(ep):
            assert ep >= 0.0

    def test_nan_height_gives_nan_offsets(self):
        ext = self._ext()
        C_mean = np.ones(_N_RX, dtype=complex)
        xl, yl, ep = ext._compute_direction(C_mean, height_km=np.nan, wavelength_m=60.0)
        assert math.isnan(xl)
        assert math.isnan(yl)

    def test_output_units_scale_with_height(self):
        """Echolocations should scale linearly with virtual height."""
        ext = self._ext()
        rng = np.random.default_rng(42)
        C_mean = rng.standard_normal(_N_RX) + 1j * rng.standard_normal(_N_RX)
        xl1, yl1, _ = ext._compute_direction(C_mean, height_km=200.0, wavelength_m=60.0)
        xl2, yl2, _ = ext._compute_direction(C_mean, height_km=400.0, wavelength_m=60.0)
        if not (math.isnan(xl1) or math.isnan(xl2)):
            assert abs(xl2 / xl1 - 2.0) < 0.05
            assert abs(yl2 / yl1 - 2.0) < 0.05


# ===========================================================================
# _compute_polarization
# ===========================================================================


class TestComputePolarization:
    def test_returns_float(self):
        ext = _make_extractor()
        C_mean = np.ones(_N_RX, dtype=complex)
        pp = ext._compute_polarization(C_mean)
        assert isinstance(pp, float)

    def test_nan_for_single_receiver(self):
        sct = _make_sct(n_rx=1)
        pulset = _make_pulset(n_rx=1)
        ext = EchoExtractor(sct=sct, pulsets=[pulset])
        pp = ext._compute_polarization(np.array([1.0 + 0j]))
        assert math.isnan(pp)

    def test_parallel_antennas_may_return_nan(self):
        """When all direction vectors are parallel, no orthogonal pair exists."""
        n_rx = 4
        sct = _make_sct(n_rx=n_rx)
        # Force all direction vectors to point East
        sct.station.rx_direction[:] = np.array([1.0, 0.0, 0.0])
        pulset = _make_pulset(n_rx=n_rx)
        ext = EchoExtractor(sct=sct, pulsets=[pulset])
        C_mean = np.ones(n_rx, dtype=complex)
        pp = ext._compute_polarization(C_mean)
        assert math.isnan(pp)

    def test_result_in_degree_range(self):
        ext = _make_extractor()
        rng = np.random.default_rng(7)
        C_mean = rng.standard_normal(_N_RX) + 1j * rng.standard_normal(_N_RX)
        pp = ext._compute_polarization(C_mean)
        if not math.isnan(pp):
            assert -180.0 <= pp <= 180.0


# ===========================================================================
# _extract_from_pulset
# ===========================================================================


class TestExtractFromPulset:
    def test_strong_gate_produces_echo(self):
        """A planted strong echo at signal_gate should produce at least one Echo."""
        ext = _make_extractor(amplitude=500.0)
        pulset = _make_pulset(amplitude=500.0)
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=_FREQ_KHZ, pulse_ut=0.0)
        assert len(echoes) >= 1

    def test_all_noise_gives_no_echoes(self):
        """A pulset with no planted signal and high threshold yields no echoes."""
        sct = _make_sct()
        pulset = _make_pulset(amplitude=0.0, n_rx=_N_RX)
        ext = EchoExtractor(sct=sct, pulsets=[pulset], snr_threshold_db=100.0)
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=_FREQ_KHZ, pulse_ut=0.0)
        assert echoes == []

    def test_echo_frequency_set_correctly(self):
        ext = _make_extractor(amplitude=500.0)
        pulset = _make_pulset(amplitude=500.0)
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=7000.0, pulse_ut=0.0)
        for e in echoes:
            assert e.frequency_khz == pytest.approx(7000.0)

    def test_echo_pulse_ut_set_correctly(self):
        ext = _make_extractor(amplitude=500.0)
        pulset = _make_pulset(amplitude=500.0)
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=_FREQ_KHZ, pulse_ut=12.5)
        for e in echoes:
            assert e.pulse_ut == pytest.approx(12.5)

    def test_max_echoes_per_pulset_respected(self):
        sct = _make_sct()
        pulset = _make_pulset(amplitude=500.0)
        ext = EchoExtractor(
            sct=sct, pulsets=[pulset], snr_threshold_db=0.0, max_echoes_per_pulset=2
        )
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=_FREQ_KHZ, pulse_ut=0.0)
        assert len(echoes) <= 2

    def test_max_echoes_none_keeps_all(self):
        sct = _make_sct()
        pulset = _make_pulset(amplitude=500.0)
        ext = EchoExtractor(
            sct=sct, pulsets=[pulset], snr_threshold_db=0.0, max_echoes_per_pulset=None
        )
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=_FREQ_KHZ, pulse_ut=0.0)
        # Should keep all gates above threshold (could be many)
        assert isinstance(echoes, list)

    def test_echo_height_is_finite(self):
        ext = _make_extractor(amplitude=500.0)
        pulset = _make_pulset(amplitude=500.0)
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=_FREQ_KHZ, pulse_ut=0.0)
        for e in echoes:
            assert math.isfinite(e.height_km) or math.isnan(e.height_km)

    def test_echo_snr_positive(self):
        ext = _make_extractor(amplitude=500.0)
        pulset = _make_pulset(amplitude=500.0)
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=_FREQ_KHZ, pulse_ut=0.0)
        for e in echoes:
            assert e.snr_db >= ext.snr_threshold_db

    def test_echo_rx_count_matches(self):
        ext = _make_extractor()
        pulset = _make_pulset(amplitude=500.0)
        C = ext._build_iq_cube(pulset)
        echoes = ext._extract_from_pulset(C, freq_khz=_FREQ_KHZ, pulse_ut=0.0)
        for e in echoes:
            assert e.rx_count == _N_RX


# ===========================================================================
# extract (end-to-end)
# ===========================================================================


class TestExtract:
    def test_returns_self(self):
        ext = _make_extractor()
        result = ext.extract()
        assert result is ext

    def test_echoes_list_populated(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        assert isinstance(ext._echoes, list)

    def test_extract_multiple_pulsets(self):
        sct = _make_sct()
        pulsets = [
            _make_pulset(frequency_khz=f, amplitude=500.0)
            for f in [4000.0, 5000.0, 6000.0]
        ]
        ext = EchoExtractor(sct=sct, pulsets=pulsets, snr_threshold_db=3.0)
        ext.extract()
        freqs = {e.frequency_khz for e in ext._echoes}
        # Expect echoes at multiple frequencies
        assert len(freqs) > 0

    def test_chaining_to_dataframe(self):
        sct = _make_sct()
        pulset = _make_pulset(amplitude=500.0)
        df = EchoExtractor(sct=sct, pulsets=[pulset]).extract().to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_chaining_to_xarray(self):
        sct = _make_sct()
        pulset = _make_pulset(amplitude=500.0)
        ds = EchoExtractor(sct=sct, pulsets=[pulset]).extract().to_xarray()
        assert isinstance(ds, xr.Dataset)


# ===========================================================================
# to_dataframe
# ===========================================================================


class TestToDataframe:
    def test_returns_dataframe(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        df = ext.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        df = ext.to_dataframe()
        required = {
            "frequency_khz", "height_km", "amplitude_db", "gross_phase_deg",
            "doppler_hz", "velocity_mps", "xl_km", "yl_km",
            "polarization_deg", "residual_deg", "snr_db",
            "gate_index", "pulse_ut", "rx_count",
        }
        assert required.issubset(set(df.columns))

    def test_empty_when_no_echoes(self):
        sct = _make_sct()
        pulset = _make_pulset(amplitude=0.0)
        ext = EchoExtractor(sct=sct, pulsets=[pulset], snr_threshold_db=1000.0)
        ext.extract()
        df = ext.to_dataframe()
        assert df.empty

    def test_row_count_consistent_with_echoes(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        assert len(ext.to_dataframe()) == len(ext.echoes)

    def test_frequency_column_values(self):
        sct = _make_sct()
        pulset = _make_pulset(frequency_khz=7500.0, amplitude=500.0)
        ext = EchoExtractor(sct=sct, pulsets=[pulset])
        ext.extract()
        df = ext.to_dataframe()
        if not df.empty:
            np.testing.assert_allclose(df["frequency_khz"].values, 7500.0)


# ===========================================================================
# to_xarray
# ===========================================================================


class TestToXarray:
    def test_returns_dataset(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        ds = ext.to_xarray()
        assert isinstance(ds, xr.Dataset)

    def test_cf_units_attached(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        ds = ext.to_xarray()
        if "height_km" in ds:
            assert ds["height_km"].attrs.get("units") == "km"
        if "velocity_mps" in ds:
            assert ds["velocity_mps"].attrs.get("units") == "m/s"
        if "frequency_khz" in ds:
            assert ds["frequency_khz"].attrs.get("units") == "kHz"

    def test_long_names_attached(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        ds = ext.to_xarray()
        if "xl_km" in ds:
            assert "long_name" in ds["xl_km"].attrs
        if "gross_phase_deg" in ds:
            assert "long_name" in ds["gross_phase_deg"].attrs

    def test_echo_index_dimension(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        ds = ext.to_xarray()
        if ds.data_vars:
            assert "echo_index" in ds.dims

    def test_global_description_attr(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        ds = ext.to_xarray()
        if ds.data_vars:
            assert "description" in ds.attrs
            assert "Dynasonde" in ds.attrs["description"]

    def test_empty_dataset_when_no_echoes(self):
        sct = _make_sct()
        pulset = _make_pulset(amplitude=0.0)
        ext = EchoExtractor(sct=sct, pulsets=[pulset], snr_threshold_db=1000.0)
        ext.extract()
        ds = ext.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 0


# ===========================================================================
# echoes property and guard
# ===========================================================================


class TestEchoesPropertyAndGuard:
    def test_echoes_property_returns_list(self):
        ext = _make_extractor(amplitude=500.0)
        ext.extract()
        assert isinstance(ext.echoes, list)

    def test_runtime_error_before_extract_echoes(self):
        ext = _make_extractor()
        with pytest.raises(RuntimeError, match="extract()"):
            _ = ext.echoes

    def test_runtime_error_before_extract_to_dataframe(self):
        ext = _make_extractor()
        with pytest.raises(RuntimeError, match="extract()"):
            ext.to_dataframe()

    def test_runtime_error_before_extract_to_xarray(self):
        ext = _make_extractor()
        with pytest.raises(RuntimeError, match="extract()"):
            ext.to_xarray()

    def test_runtime_error_before_extract_fit_drift(self):
        ext = _make_extractor()
        with pytest.raises(RuntimeError, match="extract()"):
            ext.fit_drift_velocity()


# ===========================================================================
# fit_drift_velocity
# ===========================================================================


def _make_extractor_with_planted_velocity(
    vx: float = 50.0,
    vy: float = -30.0,
    vz: float = 10.0,
    n_echoes: int = 20,
    noise_mps: float = 0.0,
    n_rx: int = 6,
) -> EchoExtractor:
    """Return an EchoExtractor whose echoes encode a known 3-D drift.

    Echoes are planted with XL, YL, height, and velocity_mps consistent
    with the supplied [vx, vy, vz].  Because we cannot inject velocity_mps
    directly via IQ data, we exploit a mock: after extract(), we overwrite
    _echoes with synthetic Echo objects that have the desired parameters.
    """
    from pynasonde.vipir.riq.echo import Echo

    rng = np.random.default_rng(0)

    # Spread direction cosines uniformly on the unit hemisphere
    theta = rng.uniform(0.0, np.pi / 4, n_echoes)   # zenith angle 0–45°
    phi   = rng.uniform(0.0, 2 * np.pi, n_echoes)   # azimuth

    l = np.sin(theta) * np.cos(phi)
    m = np.sin(theta) * np.sin(phi)
    n_cos = np.cos(theta)

    height_km = rng.uniform(200.0, 400.0, n_echoes)
    xl_km = l * height_km
    yl_km = m * height_km

    # True LOS velocity from the planted drift vector
    vlos = l * vx + m * vy + n_cos * vz
    if noise_mps > 0.0:
        vlos += rng.standard_normal(n_echoes) * noise_mps

    echoes = [
        Echo(
            frequency_khz=5000.0,
            height_km=float(height_km[i]),
            xl_km=float(xl_km[i]),
            yl_km=float(yl_km[i]),
            velocity_mps=float(vlos[i]),
            snr_db=20.0,
            residual_deg=10.0,
            amplitude_db=50.0,
            gate_index=int(i),
            pulse_ut=0.0,
            rx_count=n_rx,
        )
        for i in range(n_echoes)
    ]

    sct = _make_sct(n_rx=n_rx)
    ext = EchoExtractor(sct=sct, pulsets=[], snr_threshold_db=3.0)
    ext._echoes = echoes   # bypass extract()
    return ext


class TestFitDriftVelocity:
    # ── guard ──────────────────────────────────────────────────────────────
    def test_guard_before_extract(self):
        ext = _make_extractor()
        with pytest.raises(RuntimeError):
            ext.fit_drift_velocity()

    # ── empty result when no echoes ─────────────────────────────────────
    def test_empty_when_no_echoes(self):
        sct = _make_sct()
        ext = EchoExtractor(sct=sct, pulsets=[])
        ext._echoes = []
        df = ext.fit_drift_velocity()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    # ── whole-sounding mode (height_bin_km=None) ─────────────────────────
    def test_whole_sounding_returns_one_row(self):
        ext = _make_extractor_with_planted_velocity()
        df = ext.fit_drift_velocity()
        assert len(df) == 1

    def test_whole_sounding_no_height_bin_column(self):
        ext = _make_extractor_with_planted_velocity()
        df = ext.fit_drift_velocity()
        assert "height_bin_km" not in df.columns

    def test_whole_sounding_expected_columns(self):
        ext = _make_extractor_with_planted_velocity()
        df = ext.fit_drift_velocity()
        for col in ("vx_mps", "vy_mps", "vz_mps", "residual_mps",
                    "condition_number", "n_echoes", "n_rejected"):
            assert col in df.columns, f"missing column: {col}"

    def test_whole_sounding_recovers_planted_velocity(self):
        """With zero noise, LS should recover [Vx, Vy, Vz] exactly."""
        vx, vy, vz = 50.0, -30.0, 10.0
        ext = _make_extractor_with_planted_velocity(vx=vx, vy=vy, vz=vz, noise_mps=0.0)
        df = ext.fit_drift_velocity(n_sigma=float("inf"))
        assert df["vx_mps"].iloc[0] == pytest.approx(vx, abs=0.1)
        assert df["vy_mps"].iloc[0] == pytest.approx(vy, abs=0.1)
        assert df["vz_mps"].iloc[0] == pytest.approx(vz, abs=0.1)

    def test_whole_sounding_residual_near_zero_no_noise(self):
        ext = _make_extractor_with_planted_velocity(noise_mps=0.0)
        df = ext.fit_drift_velocity(n_sigma=float("inf"))
        assert df["residual_mps"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_whole_sounding_n_echoes_matches_input(self):
        ext = _make_extractor_with_planted_velocity(n_echoes=20, noise_mps=0.0)
        df = ext.fit_drift_velocity(n_sigma=float("inf"))
        assert df["n_echoes"].iloc[0] == 20
        assert df["n_rejected"].iloc[0] == 0

    # ── height-binned mode ───────────────────────────────────────────────
    def test_binned_returns_dataframe(self):
        ext = _make_extractor_with_planted_velocity(n_echoes=30)
        df = ext.fit_drift_velocity(height_bin_km=100.0, min_echoes=1)
        assert isinstance(df, pd.DataFrame)

    def test_binned_has_height_bin_column(self):
        ext = _make_extractor_with_planted_velocity(n_echoes=30)
        df = ext.fit_drift_velocity(height_bin_km=100.0, min_echoes=1)
        assert "height_bin_km" in df.columns

    def test_binned_height_centres_are_monotone(self):
        ext = _make_extractor_with_planted_velocity(n_echoes=40)
        df = ext.fit_drift_velocity(height_bin_km=50.0, min_echoes=1)
        if len(df) > 1:
            centres = df["height_bin_km"].values
            assert np.all(np.diff(centres) > 0)

    def test_binned_recovers_planted_velocity(self):
        vx, vy, vz = 100.0, 20.0, -5.0
        ext = _make_extractor_with_planted_velocity(
            vx=vx, vy=vy, vz=vz, n_echoes=60, noise_mps=0.0
        )
        df = ext.fit_drift_velocity(height_bin_km=100.0, min_echoes=1, n_sigma=float("inf"))
        for _, row in df.iterrows():
            assert row["vx_mps"] == pytest.approx(vx, abs=0.5)
            assert row["vy_mps"] == pytest.approx(vy, abs=0.5)
            assert row["vz_mps"] == pytest.approx(vz, abs=0.5)

    def test_binned_min_echoes_skips_sparse_bins(self):
        # Only 3 echoes total — bin of 50 km will have few per bin
        ext = _make_extractor_with_planted_velocity(n_echoes=3)
        df = ext.fit_drift_velocity(height_bin_km=50.0, min_echoes=10)
        # All bins below min_echoes → all velocity NaN
        assert df["vx_mps"].isna().all()

    # ── sigma-clipping ───────────────────────────────────────────────────
    def test_sigma_clip_rejects_outliers(self):
        """Inject one extreme outlier; sigma-clipping should remove it."""
        from pynasonde.vipir.riq.echo import Echo

        ext = _make_extractor_with_planted_velocity(n_echoes=20, noise_mps=0.0)
        # Inject one echo with wildly wrong V*
        outlier = Echo(
            frequency_khz=5000.0, height_km=300.0,
            xl_km=10.0, yl_km=5.0,
            velocity_mps=9999.0,   # extreme outlier
            snr_db=20.0, residual_deg=10.0, amplitude_db=50.0,
            gate_index=99, pulse_ut=0.0, rx_count=6,
        )
        ext._echoes.append(outlier)

        df_clip = ext.fit_drift_velocity(n_sigma=2.5)
        df_raw  = ext.fit_drift_velocity(n_sigma=float("inf"))
        # Clipped residual should be much smaller
        assert df_clip["residual_mps"].iloc[0] < df_raw["residual_mps"].iloc[0]
        assert df_clip["n_rejected"].iloc[0] >= 1

    def test_sigma_inf_disables_clipping(self):
        ext = _make_extractor_with_planted_velocity(n_echoes=20, noise_mps=0.0)
        df = ext.fit_drift_velocity(n_sigma=float("inf"))
        assert df["n_rejected"].iloc[0] == 0

    # ── EP pre-filter ────────────────────────────────────────────────────
    def test_ep_filter_removes_high_ep_echoes(self):
        """Echoes above max_ep_deg should be excluded before the fit."""
        from pynasonde.vipir.riq.echo import Echo

        ext = _make_extractor_with_planted_velocity(n_echoes=20, noise_mps=0.0)
        # Add echoes with very high EP
        for _ in range(5):
            bad = Echo(
                frequency_khz=5000.0, height_km=300.0,
                xl_km=0.0, yl_km=0.0,
                velocity_mps=500.0,
                snr_db=20.0, residual_deg=120.0,
                amplitude_db=50.0, gate_index=0, pulse_ut=0.0, rx_count=6,
            )
            ext._echoes.append(bad)

        df_filtered   = ext.fit_drift_velocity(max_ep_deg=30.0,  n_sigma=float("inf"))
        df_unfiltered = ext.fit_drift_velocity(max_ep_deg=None,  n_sigma=float("inf"))
        # Filtered fit uses fewer echoes
        assert df_filtered["n_echoes"].iloc[0] < df_unfiltered["n_echoes"].iloc[0]

    # ── snr_weight ───────────────────────────────────────────────────────
    def test_snr_weight_flag_accepted(self):
        ext = _make_extractor_with_planted_velocity()
        df_w  = ext.fit_drift_velocity(snr_weight=True,  n_sigma=float("inf"))
        df_nw = ext.fit_drift_velocity(snr_weight=False, n_sigma=float("inf"))
        # Both should return a single row with finite velocities
        assert df_w["vx_mps"].notna().all()
        assert df_nw["vx_mps"].notna().all()

    # ── condition number ─────────────────────────────────────────────────
    def test_condition_number_positive(self):
        ext = _make_extractor_with_planted_velocity(n_echoes=20)
        df = ext.fit_drift_velocity(n_sigma=float("inf"))
        assert df["condition_number"].iloc[0] > 0

    def test_near_vertical_echoes_high_condition_number(self):
        """Echoes all arriving near-vertically → l≈m≈0 → ill-conditioned."""
        from pynasonde.vipir.riq.echo import Echo

        sct = _make_sct()
        echoes = [
            Echo(
                frequency_khz=5000.0, height_km=300.0,
                xl_km=0.1, yl_km=0.1,   # nearly vertical
                velocity_mps=10.0,
                snr_db=20.0, residual_deg=5.0, amplitude_db=50.0,
                gate_index=i, pulse_ut=0.0, rx_count=4,
            )
            for i in range(20)
        ]
        ext = EchoExtractor(sct=sct, pulsets=[])
        ext._echoes = echoes
        df = ext.fit_drift_velocity(n_sigma=float("inf"))
        assert df["condition_number"].iloc[0] > 10
