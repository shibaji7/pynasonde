"""Regression tests for shared Digisonde filename metadata parsing."""

import datetime as dt

import pytest

from pynasonde.digisonde.parsers.dft import DftExtractor
from pynasonde.digisonde.parsers.dvl import DvlExtractor
from pynasonde.digisonde.parsers.edp import EdpExtractor
from pynasonde.digisonde.parsers.mmm import ModMaxExtractor
from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.parsers.sbf import SbfExtractor
from pynasonde.digisonde.parsers.sky import SkyExtractor


def _binary_file(tmp_path, suffix: str):
    path = tmp_path / f"KR835_2024299123456.{suffix}"
    path.write_bytes(bytes(4096))
    return path


def _text_file(tmp_path, suffix: str):
    path = tmp_path / f"KR835_2024299123456.{suffix}"
    path.write_text("")
    return path


@pytest.mark.parametrize(
    "extractor_cls, suffix, binary",
    [
        (SaoExtractor, "SAO", False),
        (DvlExtractor, "DVL", False),
        (DftExtractor, "DFT", True),
        (RsfExtractor, "RSF", True),
        (SbfExtractor, "SBF", True),
        (ModMaxExtractor, "MMM", True),
        (SkyExtractor, "SKY", False),
        (EdpExtractor, "EDP", False),
    ],
)
def test_extractors_parse_filename_station_and_datetime(
    tmp_path, extractor_cls, suffix, binary
):
    path = _binary_file(tmp_path, suffix) if binary else _text_file(tmp_path, suffix)

    extractor = extractor_cls(
        str(path),
        extract_time_from_name=True,
        extract_stn_from_name=True,
    )

    assert extractor.stn_code == "KR835"
    assert extractor.date == dt.datetime(2024, 10, 25, 12, 34, 56)


@pytest.mark.parametrize("extractor_cls", [SaoExtractor, SkyExtractor])
def test_extractors_parse_day_file_filename_format(tmp_path, extractor_cls):
    suffix = "SAO" if extractor_cls is SaoExtractor else "SKY"
    path = tmp_path / f"KR835_20241025(299).{suffix}"
    path.write_text("")

    extractor = extractor_cls(str(path), extract_time_from_name=True)

    assert extractor.date == dt.datetime(2024, 10, 25)


def test_dvl_constructor_without_filename_time_does_not_raise(tmp_path):
    path = tmp_path / "KR835_no_time.DVL"
    path.write_text("")

    extractor = DvlExtractor(str(path))

    assert extractor.dvl_struct["date"] is None
    assert extractor.dvl_struct["time"] is None
