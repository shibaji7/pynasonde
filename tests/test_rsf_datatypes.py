"""Tests for pynasonde.digisonde.datatypes.rsfdatatypes.

Exercises RsfHeader unit-conversion __post_init__, RsfFrequencyGroup.setup()
(azimuth, phase, amplitude, offset, height, azm_directions), RsfDataUnit.setup()
(height vector population), and the RsfDataFile container.
"""

import datetime as dt

import numpy as np
import pytest

from pynasonde.digisonde.datatypes.rsfdatatypes import (
    RsfDataFile,
    RsfDataUnit,
    RsfFreuencyGroup,
    RsfHeader,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_header(**overrides):
    """Create an RsfHeader with sensible defaults."""
    defaults = dict(
        record_type=4,
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
        start_frequency=10000,  # ×100 Hz → 1 000 000 Hz = 1 MHz
        coarse_frequency_step=100,  # ×1 kHz  → 100 000 Hz
        stop_frequency=45000,  # ×100 Hz → 4 500 000 Hz = 4.5 MHz
        fine_frequency_step=10,  # ×1 kHz  → 10 000 Hz
        num_small_steps_in_scan=3,
        phase_code=1,
        option_code=0,
        number_of_samples=5,
        pulse_repetition_rate=100,
        range_start=80,
        range_increment=5,  # → 5 km
        number_of_heights=128,
        delay=0,
        base_gain=3,
        frequency_search=0,
        operating_mode=0,
        data_format=4,
        printer_output=0,
        threshold=10,  # → 3*(10-10) = 0 dB
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
    return RsfHeader(**defaults)


def make_frequency_group(n=8, **overrides):
    """Create an RsfFreuencyGroup with synthetic array data."""
    defaults = dict(
        pol="O",
        group_size=2,
        frequency_reading=300.0,  # → 300 × 10000 = 3 000 000 Hz
        offset=2,  # → 0 Hz (no offset)
        additional_gain=3.0,  # → 9 dB
        seconds=9,
        mpa=2.0,  # → 2 (raw); amplitude values < 2 set to 0
        amplitude=np.ones(n, dtype=np.float64) * 5.0,
        dop_num=np.arange(n, dtype=np.float64),
        phase=np.ones(n, dtype=np.float64) * 2.0,
        azimuth=np.array([0, 1, 2, 3, 4, 5, 0, 1][:n], dtype=np.float64),
        height=np.zeros(n, dtype=np.float64),
    )
    defaults.update(overrides)
    return RsfFreuencyGroup(**defaults)


# ---------------------------------------------------------------------------
# RsfHeader
# ---------------------------------------------------------------------------


class TestRsfHeader:
    def test_frequency_conversion(self):
        h = make_header(start_frequency=10000)
        assert h.start_frequency == pytest.approx(1_000_000.0)  # 10000 × 100 Hz

    def test_coarse_step_conversion(self):
        h = make_header(coarse_frequency_step=100)
        assert h.coarse_frequency_step == pytest.approx(100_000.0)  # 100 kHz

    def test_stop_frequency_conversion(self):
        h = make_header(stop_frequency=45000)
        assert h.stop_frequency == pytest.approx(4_500_000.0)

    def test_fine_step_conversion(self):
        h = make_header(fine_frequency_step=10)
        assert h.fine_frequency_step == pytest.approx(10_000.0)

    def test_range_increment_2_becomes_2_5(self):
        h = make_header(range_increment=2)
        assert h.range_increment == pytest.approx(2.5)

    def test_range_increment_5(self):
        h = make_header(range_increment=5)
        assert h.range_increment == pytest.approx(5.0)

    def test_range_increment_10(self):
        h = make_header(range_increment=10)
        assert h.range_increment == pytest.approx(10.0)

    def test_threshold_zero(self):
        h = make_header(threshold=10)
        assert h.threshold == pytest.approx(0.0)  # 3*(10-10)=0

    def test_threshold_nonzero(self):
        h = make_header(threshold=13)
        assert h.threshold == pytest.approx(9.0)  # 3*(13-10)=9

    def test_threshold_zero_raw(self):
        h = make_header(threshold=0)
        assert np.isnan(h.threshold)  # 0 → NaN special case

    def test_date_synthesized(self):
        h = make_header(year=2023, month=10, dom=14, hour=0, minute=9, second=15)
        assert h.date == dt.datetime(2023, 10, 14, 0, 9, 15)

    def test_data_format_code(self):
        h = make_header(data_format=4)
        assert h.data_format == 4  # RSF = 4

    def test_phase_code_field(self):
        h = make_header(phase_code=2)
        assert h.phase_code == 2

    def test_operating_mode_field(self):
        h = make_header(operating_mode=1)
        assert h.operating_mode == 1


# ---------------------------------------------------------------------------
# RsfFreuencyGroup
# ---------------------------------------------------------------------------


class TestRsfFrequencyGroup:
    def test_azimuth_converted_to_degrees(self):
        fg = make_frequency_group(n=6)
        fg.setup()
        # raw azimuth [0,1,2,3,4,5] × 60 → [0,60,120,180,240,300]
        expected = np.array([0, 60, 120, 180, 240, 300], dtype=float)
        np.testing.assert_array_equal(fg.azimuth, expected)

    def test_phase_converted_to_degrees(self):
        fg = make_frequency_group(n=4)
        fg.setup()
        # raw phase 2.0 × 11.25 = 22.5 deg
        np.testing.assert_array_almost_equal(fg.phase, [22.5, 22.5, 22.5, 22.5])

    def test_amplitude_converted_to_db(self):
        fg = make_frequency_group(n=4, mpa=0.0)
        fg.setup()
        # raw amplitude 5.0 × 3 = 15.0 dB
        np.testing.assert_array_almost_equal(fg.amplitude, [15.0, 15.0, 15.0, 15.0])

    def test_amplitude_below_mpa_zeroed(self):
        fg = make_frequency_group(
            n=4, mpa=10.0, amplitude=np.array([3.0, 15.0, 3.0, 15.0])
        )
        fg.setup()
        # raw 3 * 3 = 9 < mpa 10 → 0; raw 15 * 3 = 45 > mpa 10 → kept
        assert fg.amplitude[0] == 0.0
        assert fg.amplitude[1] > 0.0
        assert fg.amplitude[2] == 0.0
        assert fg.amplitude[3] > 0.0

    def test_offset_0_maps_to_minus_20khz(self):
        fg = make_frequency_group(offset=0)
        fg.setup()
        assert fg.offset == -20e3

    def test_offset_1_maps_to_minus_10khz(self):
        fg = make_frequency_group(offset=1)
        fg.setup()
        assert fg.offset == -10e3

    def test_offset_2_maps_to_zero(self):
        fg = make_frequency_group(offset=2)
        fg.setup()
        assert fg.offset == 0

    def test_offset_3_maps_to_plus_10khz(self):
        fg = make_frequency_group(offset=3)
        fg.setup()
        assert fg.offset == 10e3

    def test_offset_4_maps_to_plus_20khz(self):
        fg = make_frequency_group(offset=4)
        fg.setup()
        assert fg.offset == 20e3

    def test_offset_5_maps_to_search_failed(self):
        fg = make_frequency_group(offset=5)
        fg.setup()
        assert fg.offset == "Search Failed"

    def test_offset_E_maps_to_forced(self):
        fg = make_frequency_group(offset="E")
        fg.setup()
        assert fg.offset == "Forced"

    def test_offset_F_maps_to_no_tx(self):
        fg = make_frequency_group(offset="F")
        fg.setup()
        assert fg.offset == "No Tx"

    def test_additional_gain_converted(self):
        fg = make_frequency_group(additional_gain=4.0)
        fg.setup()
        assert fg.additional_gain == pytest.approx(12.0)  # 4 × 3 dB

    def test_frequency_reading_converted_to_hz(self):
        fg = make_frequency_group(frequency_reading=300.0)
        fg.setup()
        assert fg.frequency_reading == pytest.approx(300.0 * 10e3)

    def test_azm_directions_set(self):
        fg = make_frequency_group(
            n=6, azimuth=np.array([0, 1, 2, 3, 4, 5], dtype=float)
        )
        fg.setup()
        expected = ["N", "NE", "SE", "S", "SW", "NW"]
        assert fg.azm_directions == expected

    def test_height_array_initialized(self):
        fg = make_frequency_group(n=4)
        fg.setup()
        assert fg.height is not None
        assert len(fg.height) == 4


# ---------------------------------------------------------------------------
# RsfDataUnit.setup()
# ---------------------------------------------------------------------------


class TestRsfDataUnit:
    def test_setup_populates_height_vectors(self):
        header = make_header(range_start=80, range_increment=5)
        groups = [make_frequency_group(n=4), make_frequency_group(n=4)]
        unit = RsfDataUnit(header=header, frequency_groups=groups)
        unit.setup()
        for g in unit.frequency_groups:
            expected = np.array([80, 85, 90, 95], dtype=float)
            np.testing.assert_array_almost_equal(g.height, expected)

    def test_setup_multiple_groups(self):
        header = make_header(range_start=100, range_increment=10)
        groups = [make_frequency_group(n=3) for _ in range(5)]
        unit = RsfDataUnit(header=header, frequency_groups=groups)
        unit.setup()
        assert len(unit.frequency_groups) == 5
        for g in unit.frequency_groups:
            np.testing.assert_array_almost_equal(g.height, [100, 110, 120])


# ---------------------------------------------------------------------------
# RsfDataFile
# ---------------------------------------------------------------------------


class TestRsfDataFile:
    def test_container_holds_units(self):
        header = make_header()
        unit = RsfDataUnit(header=header, frequency_groups=[])
        f = RsfDataFile(rsf_data_units=[unit])
        assert len(f.rsf_data_units) == 1
        assert f.rsf_data_units[0] is unit

    def test_empty_file(self):
        f = RsfDataFile(rsf_data_units=[])
        assert f.rsf_data_units == []
