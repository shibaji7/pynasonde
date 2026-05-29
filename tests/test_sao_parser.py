"""Tests for selected helper methods in the SAO parser."""

import datetime as dt

import pytest

import pynasonde.digisonde.parsers.sao as sao_module
from pynasonde.digisonde.parsers.sao import SaoExtractor


def test_pad_and_parse_line(tmp_path):
    extractor = SaoExtractor("KR835_2024099160913.SAO", extract_time_from_name=True)

    padded = extractor.pad("TEST", 6, pad_char="0")
    assert padded == "TEST00"

    line = "12345678901234567890"
    parsed = extractor.parse_line(line, "%10s", 10)
    assert parsed == ["1234567890", "1234567890"]


def test_read_file(tmp_path):
    sao_file = tmp_path / "sample.SAO"
    sao_file.write_text("LINE1\nLINE2\n")

    extractor = SaoExtractor(str(sao_file))
    lines = extractor.read_file()
    assert lines == ["LINE1", "LINE2"]


def test_parse_ff_datetime():
    ff = "FF202429910250003040120126101600007520000000001107301000080202560000820040000"
    parsed = SaoExtractor._parse_ff_datetime(ff)
    assert parsed == dt.datetime(2024, 10, 25, 0, 3, 0)


def test_parse_ff_datetime_malformed_returns_none():
    assert SaoExtractor._parse_ff_datetime("FF_BROKEN_LINE") is None


def test_parse_ff_datetime_mmdd_mismatch_warns(monkeypatch):
    warnings = []
    monkeypatch.setattr(sao_module.logger, "warning", lambda msg: warnings.append(msg))
    ff = "FF202429910260003"
    parsed = SaoExtractor._parse_ff_datetime(ff)
    assert parsed == dt.datetime(2024, 10, 25, 0, 3, 0)
    assert len(warnings) == 1
    assert "month/day mismatch" in warnings[0]


def test_resolve_record_index_negative():
    assert SaoExtractor._resolve_record_index(-1, 5) == 4


def test_resolve_record_index_out_of_bounds():
    with pytest.raises(IndexError):
        SaoExtractor._resolve_record_index(2, 2)
