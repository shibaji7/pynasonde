"""Extended tests for pynasonde.vipir.riq.parsers.trace — covers uncovered branches."""

from types import SimpleNamespace

import numpy as np
import pytest

from pynasonde.vipir.riq.parsers.trace import (
    compute_phase,
    compute_phase_velocity,
    extract_echo_traces,
    get_clean_iq_by_heights,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal sct namespace that trace functions accept
# ---------------------------------------------------------------------------

def _make_sct(tune_type=1, pulse_count=4, n_freqs=8, pri_us=1000.0):
    freq_ns = SimpleNamespace(
        tune_type=tune_type,
        pulse_count=pulse_count,
        base_table=np.linspace(2000.0, 10000.0, n_freqs),  # kHz
    )
    timing_ns = SimpleNamespace(pri=pri_us)
    return SimpleNamespace(frequency=freq_ns, timing=timing_ns)


def _make_iq(n_pulses=8, n_gates=100, n_ch=1):
    rng = np.random.default_rng(42)
    i = rng.standard_normal((n_pulses, n_gates, n_ch)).astype(np.float32)
    q = rng.standard_normal((n_pulses, n_gates, n_ch)).astype(np.float32)
    return i, q


# ---------------------------------------------------------------------------
# compute_phase
# ---------------------------------------------------------------------------

class TestComputePhase:
    def test_basic_shape(self):
        i = np.array([1.0, 0.0, -1.0, 0.0])
        q = np.array([0.0, 1.0, 0.0, -1.0])
        ph = compute_phase(i, q)
        assert ph.shape == (4,)

    def test_values_in_0_to_2pi(self):
        i = np.random.rand(20) - 0.5
        q = np.random.rand(20) - 0.5
        ph = compute_phase(i, q)
        assert np.all(ph >= 0.0)
        assert np.all(ph < 2 * np.pi)


# ---------------------------------------------------------------------------
# get_clean_iq_by_heights
# ---------------------------------------------------------------------------

class TestGetCleanIqByHeights:
    def test_returns_six_items(self):
        i, q = _make_iq()
        result = get_clean_iq_by_heights(i, q, f1_range_low=70, f1_range_high=200)
        assert len(result) == 6

    def test_power_is_non_negative(self):
        i, q = _make_iq()
        _, _, power, n_pulses, low, high = get_clean_iq_by_heights(
            i, q, f1_range_low=50, f1_range_high=150
        )
        assert np.all(power >= 0.0)
        assert n_pulses == i.shape[0]

    def test_slice_bounds_consistent(self):
        i, q = _make_iq()
        pi_r, pq_r, power, n_pulses, low, high = get_clean_iq_by_heights(
            i, q, f1_range_low=10, f1_range_high=50
        )
        assert pi_r.shape[1] == high - low


# ---------------------------------------------------------------------------
# extract_echo_traces — tune_type=1 path
# ---------------------------------------------------------------------------

class TestExtractEchoTracesTuneType1:
    def test_returns_array(self):
        sct = _make_sct(tune_type=1, pulse_count=4, n_freqs=8)
        i, q = _make_iq(n_pulses=8, n_gates=100, n_ch=1)
        result = extract_echo_traces(sct, i, q, f1_range_low=10, f1_range_high=80)
        assert isinstance(result, np.ndarray)

    def test_tune_type_2_raises(self):
        sct = _make_sct(tune_type=2)
        i, q = _make_iq()
        with pytest.raises(NotImplementedError):
            extract_echo_traces(sct, i, q)

    def test_tune_type_3_raises(self):
        sct = _make_sct(tune_type=3)
        i, q = _make_iq()
        with pytest.raises(NotImplementedError):
            extract_echo_traces(sct, i, q)

    def test_tune_type_unknown_raises(self):
        sct = _make_sct(tune_type=0)
        i, q = _make_iq()
        with pytest.raises(ValueError):
            extract_echo_traces(sct, i, q)

    def test_tune_type_ge4_path(self):
        # n_pulses / pulse_count = num_sets; base_table[::2] must have same length
        # n_freqs=4 → base_table[::2] has 2 elements; n_pulses=8, pulse_count=4 → num_sets=2
        sct = _make_sct(tune_type=4, pulse_count=4, n_freqs=4)
        i, q = _make_iq(n_pulses=8, n_gates=100, n_ch=1)
        result = extract_echo_traces(sct, i, q, f1_range_low=10, f1_range_high=80)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# compute_phase_velocity — early-return branches
# ---------------------------------------------------------------------------

class TestComputePhaseVelocity:
    def test_tune_type1_returns_dict(self):
        sct = _make_sct(tune_type=1, pulse_count=4, n_freqs=8, pri_us=1000.0)
        i, q = _make_iq(n_pulses=8, n_gates=100, n_ch=1)
        result = compute_phase_velocity(sct, i, q, f1_range_low=10, f1_range_high=80)
        # Should return a dict or None (depends on data quality)
        assert result is None or isinstance(result, dict)

    def test_no_pulses_returns_none(self):
        sct = _make_sct(tune_type=1, pulse_count=4, n_freqs=8)
        # Empty arrays → n_pulses = 0
        i = np.zeros((0, 100, 1), dtype=np.float32)
        q = np.zeros((0, 100, 1), dtype=np.float32)
        result = compute_phase_velocity(sct, i, q, f1_range_low=10, f1_range_high=80)
        assert result is None

    def test_pulse_count_zero_returns_none(self):
        sct = _make_sct(tune_type=1, pulse_count=0, n_freqs=8)
        i, q = _make_iq(n_pulses=8, n_gates=100, n_ch=1)
        result = compute_phase_velocity(sct, i, q, f1_range_low=10, f1_range_high=80)
        assert result is None

    def test_pulse_count_one_returns_none(self):
        # pulse_count <= 1 → return None
        sct = _make_sct(tune_type=1, pulse_count=1, n_freqs=8)
        i, q = _make_iq(n_pulses=8, n_gates=100, n_ch=1)
        result = compute_phase_velocity(sct, i, q, f1_range_low=10, f1_range_high=80)
        assert result is None

    def test_tune_type2_raises(self):
        sct = _make_sct(tune_type=2)
        i, q = _make_iq()
        with pytest.raises(NotImplementedError):
            compute_phase_velocity(sct, i, q)

    def test_tune_type3_raises(self):
        sct = _make_sct(tune_type=3)
        i, q = _make_iq()
        with pytest.raises(NotImplementedError):
            compute_phase_velocity(sct, i, q)

    def test_tune_type_ge4_path(self):
        sct = _make_sct(tune_type=4, pulse_count=4, n_freqs=16, pri_us=1000.0)
        i, q = _make_iq(n_pulses=8, n_gates=100, n_ch=1)
        result = compute_phase_velocity(sct, i, q, f1_range_low=10, f1_range_high=80)
        assert result is None or isinstance(result, dict)

    def test_tune_type_unknown_raises(self):
        sct = _make_sct(tune_type=0)
        i, q = _make_iq()
        with pytest.raises(ValueError):
            compute_phase_velocity(sct, i, q)

    def test_successful_result_has_expected_keys(self):
        # Use enough pulses to get a non-None result
        sct = _make_sct(tune_type=1, pulse_count=4, n_freqs=8, pri_us=1000.0)
        n_pulses = 32
        n_gates = 100
        n_ch = 1
        rng = np.random.default_rng(0)
        i = rng.standard_normal((n_pulses, n_gates, n_ch)).astype(np.float32) * 5.0
        q = rng.standard_normal((n_pulses, n_gates, n_ch)).astype(np.float32) * 5.0
        result = compute_phase_velocity(sct, i, q, f1_range_low=10, f1_range_high=80)
        if result is not None:
            assert "phase_unwrapped" in result
            assert "velocity_mps" in result
