"""Tests for selected helper methods in the SAO parser."""

import pathlib

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
