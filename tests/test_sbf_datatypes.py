"""Tests for pynasonde.digisonde.datatypes.sbfdatatypes.

Covers SbfHeader unit-conversion __post_init__, SbfFrequencyGroup.setup()
offset codes and array conversions, SbfDataUnit.setup() height vector, and
the SbfDataFile container.
"""

import datetime as dt

import numpy as np
import pytest

from pynasonde.digisonde.datatypes.sbfdatatypes import (
    SbfDataFile,
    SbfDataUnit,
    SbfFreuencyGroup,
    SbfHeader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sbf_header(**overrides):
    defaults = dict(
        record_type=5,
        header_length=80,
        version_maker=1,
        year=2023,
        doy=287,
        month=10,
        dom=14,
        hour=0,
        minute=9,
        second=15,
        stn_code_rx="KR835",
        stn_code_tx="KR835",
        schedule=1,
        program=1,
        start_frequency=10000,      # ×100 Hz → 1 000 000 Hz
        coarse_frequency_step=100,  # ×1 kHz  → 100 000 Hz
        stop_frequency=45000,       # ×100 Hz → 4 500 000 Hz
        fine_frequency_step=10,     # ×1 kHz  → 10 000 Hz
        num_small_steps_in_scan=3,
        phase_code=1,
        option_code=0,
        number_of_samples=5,
        pulse_repetition_rate=100,
        range_start=80,
        range_increment=5,
        number_of_heights=128,
        delay=0,
        base_gain=3,
        frequency_search=0,
        operating_mode=0,
        data_format=5,
        printer_output=0,
        threshold=10,
        constant_gain=0,
        cit_length=500,
        journal="0000",
        bottom_height_window=80,
        top_height_window=500,
        number_of_heights_stored=128,
        spare=b"\x00\x00",
        number_of_frequency_groups=10,
    )
    defaults.update(overrides)
    return SbfHeader(**defaults)


def make_sbf_group(n=8, **overrides):
    defaults = dict(
        pol="O",
        group_size=2,
        frequency_reading=300.0,
        offset=2,
        additional_gain=3.0,
        seconds=9,
        mpa=2.0,
        amplitude=np.ones(n, dtype=np.float64) * 5.0,
        dop_num=np.arange(n, dtype=np.float64),
        phase=np.ones(n, dtype=np.float64) * 2.0,
        azimuth=np.array([0, 1, 2, 3, 4, 5, 0, 1][:n], dtype=np.float64),
        height=np.zeros(n, dtype=np.float64),
    )
    defaults.update(overrides)
    return SbfFreuencyGroup(**defaults)


# ---------------------------------------------------------------------------
# SbfHeader
# ---------------------------------------------------------------------------

class TestSbfHeader:
    def test_start_frequency_to_hz(self):
        h = make_sbf_header(start_frequency=10000)
        assert h.start_frequency == pytest.approx(1_000_000.0)

    def test_stop_frequency_to_hz(self):
        h = make_sbf_header(stop_frequency=45000)
        assert h.stop_frequency == pytest.approx(4_500_000.0)

    def test_coarse_step_to_hz(self):
        h = make_sbf_header(coarse_frequency_step=100)
        assert h.coarse_frequency_step == pytest.approx(100_000.0)

    def test_fine_step_to_hz(self):
        h = make_sbf_header(fine_frequency_step=10)
        assert h.fine_frequency_step == pytest.approx(10_000.0)

    def test_range_increment_2_to_2_5(self):
        h = make_sbf_header(range_increment=2)
        assert h.range_increment == pytest.approx(2.5)

    def test_range_increment_5(self):
        h = make_sbf_header(range_increment=5)
        assert h.range_increment == pytest.approx(5.0)

    def test_range_increment_10(self):
        h = make_sbf_header(range_increment=10)
        assert h.range_increment == pytest.approx(10.0)

    def test_threshold_zero(self):
        h = make_sbf_header(threshold=10)
        assert h.threshold == pytest.approx(0.0)

    def test_threshold_nonzero(self):
        h = make_sbf_header(threshold=15)
        assert h.threshold == pytest.approx(15.0)   # 3*(15-10)=15

    def test_threshold_zero_raw_gives_nan(self):
        h = make_sbf_header(threshold=0)
        assert np.isnan(h.threshold)

    def test_date_synthesized(self):
        h = make_sbf_header(year=2023, month=10, dom=14, hour=0, minute=9, second=15)
        assert h.date == dt.datetime(2023, 10, 14, 0, 9, 15)

    def test_data_format_is_5(self):
        h = make_sbf_header(data_format=5)
        assert h.data_format == 5   # SBF = 5

    def test_phase_codes(self):
        h = make_sbf_header(phase_code=2)
        assert h.phase_code == 2

    def test_operating_mode_drift(self):
        h = make_sbf_header(operating_mode=1)
        assert h.operating_mode == 1


# ---------------------------------------------------------------------------
# SbfFreuencyGroup
# ---------------------------------------------------------------------------

class TestSbfFrequencyGroup:
    def test_azimuth_to_degrees(self):
        g = make_sbf_group(n=6,
                           azimuth=np.array([0, 1, 2, 3, 4, 5], dtype=float))
        g.setup()
        np.testing.assert_array_equal(g.azimuth, [0, 60, 120, 180, 240, 300])

    def test_phase_to_degrees(self):
        g = make_sbf_group(n=4)
        g.setup()
        np.testing.assert_array_almost_equal(g.phase, [22.5, 22.5, 22.5, 22.5])

    def test_amplitude_to_db(self):
        g = make_sbf_group(n=4, mpa=0.0)
        g.setup()
        np.testing.assert_array_almost_equal(g.amplitude, [15.0, 15.0, 15.0, 15.0])

    def test_offset_0_minus_20k(self):
        g = make_sbf_group(offset=0)
        g.setup()
        assert g.offset == -20e3

    def test_offset_1_minus_10k(self):
        g = make_sbf_group(offset=1)
        g.setup()
        assert g.offset == -10e3

    def test_offset_2_zero(self):
        g = make_sbf_group(offset=2)
        g.setup()
        assert g.offset == 0

    def test_offset_3_plus_10k(self):
        g = make_sbf_group(offset=3)
        g.setup()
        assert g.offset == 10e3

    def test_offset_4_plus_20k(self):
        g = make_sbf_group(offset=4)
        g.setup()
        assert g.offset == 20e3

    def test_offset_5_search_failed(self):
        g = make_sbf_group(offset=5)
        g.setup()
        assert g.offset == "Search Failed"

    def test_offset_E_forced(self):
        g = make_sbf_group(offset="E")
        g.setup()
        assert g.offset == "Forced"

    def test_offset_F_no_tx(self):
        g = make_sbf_group(offset="F")
        g.setup()
        assert g.offset == "No Tx"

    def test_additional_gain_converted(self):
        g = make_sbf_group(additional_gain=5.0)
        g.setup()
        assert g.additional_gain == pytest.approx(15.0)

    def test_frequency_reading_to_hz(self):
        g = make_sbf_group(frequency_reading=300.0)
        g.setup()
        assert g.frequency_reading == pytest.approx(300.0 * 10e3)

    def test_height_initialized_to_zeros(self):
        g = make_sbf_group(n=4)
        g.setup()
        assert len(g.height) == 4
        np.testing.assert_array_equal(g.height, [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# SbfDataUnit.setup()
# ---------------------------------------------------------------------------

class TestSbfDataUnit:
    def test_setup_populates_height_vectors(self):
        header = make_sbf_header(range_start=80, range_increment=5)
        groups = [make_sbf_group(n=4), make_sbf_group(n=4)]
        unit = SbfDataUnit(header=header, frequency_groups=groups)
        unit.setup()
        for g in unit.frequency_groups:
            expected = np.array([80, 85, 90, 95], dtype=float)
            np.testing.assert_array_almost_equal(g.height, expected)

    def test_setup_2_5km_increment(self):
        header = make_sbf_header(range_start=80, range_increment=2)  # 2→2.5
        groups = [make_sbf_group(n=4)]
        unit = SbfDataUnit(header=header, frequency_groups=groups)
        unit.setup()
        expected = np.array([80, 82.5, 85, 87.5], dtype=float)
        np.testing.assert_array_almost_equal(unit.frequency_groups[0].height, expected)


# ---------------------------------------------------------------------------
# SbfDataFile
# ---------------------------------------------------------------------------

class TestSbfDataFile:
    def test_container_holds_units(self):
        header = make_sbf_header()
        unit = SbfDataUnit(header=header, frequency_groups=[])
        f = SbfDataFile(sbf_data_units=[unit])
        assert len(f.sbf_data_units) == 1

    def test_empty_container(self):
        f = SbfDataFile(sbf_data_units=[])
        assert f.sbf_data_units == []
