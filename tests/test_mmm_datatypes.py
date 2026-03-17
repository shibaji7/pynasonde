"""Tests for pynasonde.digisonde.datatypes.mmmdatatypes.

Covers ModMaxHeader unit-conversion __post_init__, ModMaxFreuencyGroup
default field values, and the ModMaxDataUnit container.
"""

import pytest

from pynasonde.digisonde.datatypes.mmmdatatypes import (
    ModMaxDataUnit,
    ModMaxFreuencyGroup,
    ModMaxHeader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_modmax_header(**overrides):
    defaults = dict(
        record_type=1,
        header_length=64,
        version_maker=hex(1),
        year=2023,
        doy=287,
        hour=0,
        minute=9,
        second=15,
        program_set=None,
        program_type=None,
        journal=None,
        nom_frequency=10000.0,      # ×100 → 1 000 000 Hz
        tape_ctrl=None,
        print_ctrl=None,
        mmm_opt=None,
        print_clean_ctrl=None,
        print_gain_lev=None,
        ctrl_intm_tx=None,
        drft_use=None,
        start_frequency=1.0,        # ×1e6 → 1 000 000 Hz
        freq_step=0.1,              # ×1e6 → 100 000 Hz
        stop_frequency=5.0,         # ×1e6 → 5 000 000 Hz
        trg=None,
        ch_a=None,
        ch_b=None,
        sta_id="KR835",
        phase_code=1,
        ant_azm=0,
        ant_scan=0,
        ant_opt=0,
        num_samples=32,
        rep_rate=100,
        pwd_code=0,
        time_ctrl=0,
        freq_cor=0,
        gain_cor=0,
        range_inc=5,
        range_start=80,
        f_search=0,
        nom_gain=3,
    )
    defaults.update(overrides)
    return ModMaxHeader(**defaults)


# ---------------------------------------------------------------------------
# ModMaxHeader
# ---------------------------------------------------------------------------

class TestModMaxHeader:
    def test_nom_frequency_conversion(self):
        h = make_modmax_header(nom_frequency=10000.0)
        assert h.nom_frequency == pytest.approx(1_000_000.0)   # ×100

    def test_start_frequency_conversion(self):
        h = make_modmax_header(start_frequency=1.0)
        assert h.start_frequency == pytest.approx(1_000_000.0) # ×1e6

    def test_freq_step_conversion(self):
        h = make_modmax_header(freq_step=0.5)
        assert h.freq_step == pytest.approx(500_000.0)         # ×1e6

    def test_stop_frequency_conversion(self):
        h = make_modmax_header(stop_frequency=5.0)
        assert h.stop_frequency == pytest.approx(5_000_000.0)  # ×1e6

    def test_zero_frequencies(self):
        h = make_modmax_header(nom_frequency=0.0, start_frequency=0.0,
                               freq_step=0.0, stop_frequency=0.0)
        assert h.nom_frequency == pytest.approx(0.0)
        assert h.start_frequency == pytest.approx(0.0)
        assert h.freq_step == pytest.approx(0.0)
        assert h.stop_frequency == pytest.approx(0.0)

    def test_scalar_fields_stored(self):
        h = make_modmax_header(sta_id="AB123", phase_code=2, num_samples=64,
                               rep_rate=200, range_inc=10, range_start=100)
        assert h.sta_id == "AB123"
        assert h.phase_code == 2
        assert h.num_samples == 64
        assert h.rep_rate == 200
        assert h.range_inc == 10
        assert h.range_start == 100

    def test_record_type_and_lengths(self):
        h = make_modmax_header(record_type=3, header_length=128)
        assert h.record_type == 3
        assert h.header_length == 128

    def test_optional_hex_fields_none(self):
        h = make_modmax_header(program_set=None, tape_ctrl=None, mmm_opt=None)
        assert h.program_set is None
        assert h.tape_ctrl is None
        assert h.mmm_opt is None

    def test_journal_field(self):
        h = make_modmax_header(journal=[0, 1, 0, 1])
        assert h.journal == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# ModMaxFreuencyGroup
# ---------------------------------------------------------------------------

class TestModMaxFrequencyGroup:
    def test_defaults(self):
        g = ModMaxFreuencyGroup()
        assert g.blk_type == 0
        assert g.frequency == 0
        assert g.frequency_k == 0
        assert g.frequency_search == 0
        assert g.gain_param == 0
        assert g.sec == 0
        assert g.mpa == 0.0

    def test_custom_values(self):
        g = ModMaxFreuencyGroup(blk_type=2, frequency=5, frequency_k=5000,
                                gain_param=3, sec=30, mpa=15.5)
        assert g.blk_type == 2
        assert g.frequency == 5
        assert g.frequency_k == 5000
        assert g.gain_param == 3
        assert g.sec == 30
        assert g.mpa == pytest.approx(15.5)

    def test_frequency_search_field(self):
        g = ModMaxFreuencyGroup(frequency_search=1)
        assert g.frequency_search == 1


# ---------------------------------------------------------------------------
# ModMaxDataUnit
# ---------------------------------------------------------------------------

class TestModMaxDataUnit:
    def test_empty_unit(self):
        unit = ModMaxDataUnit()
        assert unit.header is None
        assert unit.frequency_groups is None

    def test_unit_with_data(self):
        header = make_modmax_header()
        groups = [ModMaxFreuencyGroup(frequency=5), ModMaxFreuencyGroup(frequency=6)]
        unit = ModMaxDataUnit(header=header, frequency_groups=groups)
        assert unit.header is header
        assert len(unit.frequency_groups) == 2
        assert unit.frequency_groups[0].frequency == 5
        assert unit.frequency_groups[1].frequency == 6

    def test_unit_preserves_header_fields(self):
        header = make_modmax_header(sta_id="TEST", range_start=100)
        unit = ModMaxDataUnit(header=header, frequency_groups=[])
        assert unit.header.sta_id == "TEST"
        assert unit.header.range_start == 100
