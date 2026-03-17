"""Tests for SctType.dump_sct() — covers lines 760-984 of sct.py."""

import pytest

from pynasonde.vipir.riq.datatypes.sct import SctType, StationType


class TestSctTypeDumpSct:
    def test_dump_to_logger_does_not_raise(self):
        """dump_sct() with no file writes to logger — should not raise."""
        sct = SctType()
        sct.dump_sct()  # to_file=None path (lines 982-983)

    def test_dump_to_file(self, tmp_path):
        """dump_sct(to_file=...) writes text to the given path (lines 979-981)."""
        sct = SctType()
        out = tmp_path / "sct_dump.txt"
        sct.dump_sct(to_file=str(out))
        assert out.exists()
        content = out.read_text()
        assert "General:" in content
        assert "Station:" in content
        assert "Timing:" in content
        assert "Frequency:" in content

    def test_dump_contains_magic(self, tmp_path):
        sct = SctType()
        out = tmp_path / "magic.txt"
        sct.dump_sct(to_file=str(out))
        content = out.read_text()
        assert "sct.magic" in content

    def test_dump_with_rx_count_gt_zero(self, tmp_path):
        """rx_count > 0 → the zip loop at line 792 runs at least one iteration."""
        sct = SctType()
        sct.station.rx_count = 1  # enable one antenna entry loop iteration
        out = tmp_path / "rx.txt"
        sct.dump_sct(to_file=str(out))
        content = out.read_text()
        assert "rx_antenna_type" in content

    def test_dump_with_drive_band_count_gt_zero(self, tmp_path):
        """drive_band_count > 0 → drive_band loop (line 812) runs."""
        sct = SctType()
        sct.station.drive_band_count = 1
        out = tmp_path / "drive.txt"
        sct.dump_sct(to_file=str(out))
        content = out.read_text()
        assert "drive_band_bounds" in content

    def test_dump_receiver_section_present(self, tmp_path):
        sct = SctType()
        out = tmp_path / "recv.txt"
        sct.dump_sct(to_file=str(out))
        content = out.read_text()
        assert "Reciever:" in content
        assert "sct.receiver" in content

    def test_dump_exciter_section_present(self, tmp_path):
        sct = SctType()
        out = tmp_path / "excit.txt"
        sct.dump_sct(to_file=str(out))
        content = out.read_text()
        assert "Exciter:" in content

    def test_dump_monitor_section_present(self, tmp_path):
        sct = SctType()
        out = tmp_path / "mon.txt"
        sct.dump_sct(to_file=str(out))
        content = out.read_text()
        assert "Monitor:" in content
        assert "sct.monitor.balun_status" in content
