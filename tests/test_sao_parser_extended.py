"""Extended tests for pynasonde.digisonde.parsers.sao.

Covers SaoExtractor.pad(), parse_line() (all format codes), extract()
with a synthetic all-zero SAO header, and get_scaled_datasets() /
get_height_profile() with manually-injected sao data.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pynasonde.digisonde.digi_utils import to_namespace
from pynasonde.digisonde.parsers.sao import SaoExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal SAO file: 2 header lines, each with 40 3-char fields of "  0"
# → noe[0..79] = 0 for all entries → extract() skips all sections
_ZERO_HDR = "  0" * 40 + "\n" + "  0" * 40 + "\n"


@pytest.fixture
def sao_file(tmp_path):
    p = tmp_path / "KR835_2024099160913.SAO"
    p.write_text(_ZERO_HDR)
    return str(p)


@pytest.fixture
def extractor(sao_file):
    return SaoExtractor(sao_file)


# ---------------------------------------------------------------------------
# SaoExtractor.pad()
# ---------------------------------------------------------------------------


class TestSaoExtractorPad:
    def test_pad_shorter(self, extractor):
        assert extractor.pad("abc", 6) == "abc   "

    def test_pad_equal_length(self, extractor):
        assert extractor.pad("abc", 3) == "abc"

    def test_pad_longer_than_length(self, extractor):
        # ljust never truncates
        assert extractor.pad("abcdef", 3) == "abcdef"

    def test_pad_custom_char(self, extractor):
        assert extractor.pad("ab", 5, "0") == "ab000"

    def test_pad_empty_string(self, extractor):
        assert extractor.pad("", 4) == "    "


# ---------------------------------------------------------------------------
# SaoExtractor.parse_line()
# ---------------------------------------------------------------------------


class TestSaoExtractorParseLine:
    def test_7_3f_numeric(self, extractor):
        assert extractor.parse_line("  5.000", "%7.3f", 7) == [pytest.approx(5.0)]

    def test_8_3f_numeric(self, extractor):
        assert extractor.parse_line("   5.500", "%8.3f", 8) == [pytest.approx(5.5)]

    def test_8_3f_multiple_chunks(self, extractor):
        result = extractor.parse_line("  80.000 100.000", "%8.3f", 8)
        assert result == [pytest.approx(80.0), pytest.approx(100.0)]

    def test_11_6f_numeric(self, extractor):
        result = extractor.parse_line("  5.000000", "%11.6f", 11)
        assert result[0] == pytest.approx(5.0)

    def test_20_12f_numeric(self, extractor):
        result = extractor.parse_line("   5.000000000000", "%20.12f", 20)
        assert result[0] == pytest.approx(5.0)

    def test_1c_returns_string(self, extractor):
        result = extractor.parse_line("ABC", "%1c", 1)
        assert result == ["A", "B", "C"]

    def test_120c_returns_string(self, extractor):
        chunk = "X" * 120
        result = extractor.parse_line(chunk, "%120c", 120)
        assert result == [chunk]

    def test_3d_integer(self, extractor):
        result = extractor.parse_line("  7", "%3d", 3)
        assert result == [pytest.approx(7.0)]

    def test_2d_integer(self, extractor):
        result = extractor.parse_line(" 3", "%2d", 2)
        assert result == [pytest.approx(3.0)]

    def test_1d_integer(self, extractor):
        result = extractor.parse_line("5", "%1d", 1)
        assert result == [pytest.approx(5.0)]

    def test_invalid_numeric_returns_none(self, extractor):
        # Whitespace-only chunk → float("") raises ValueError → None
        result = extractor.parse_line("   ", "%7.3f", 7)
        assert result == [None]

    def test_unknown_fmt_returns_chunk_as_string(self, extractor):
        result = extractor.parse_line("abc", "%xyz", 3)
        assert result == ["abc"]


# ---------------------------------------------------------------------------
# SaoExtractor.__init__ and read_file
# ---------------------------------------------------------------------------


class TestSaoExtractorInit:
    def test_stn_code_parsed(self, sao_file):
        ex = SaoExtractor(sao_file)
        assert ex.stn_code == "KR835"

    def test_not_xml(self, sao_file):
        ex = SaoExtractor(sao_file)
        assert ex.xml_file is False

    def test_extract_time_from_name(self, tmp_path):
        p = tmp_path / "KR835_2024099160913.SAO"
        p.write_text(_ZERO_HDR)
        ex = SaoExtractor(str(p), extract_time_from_name=True)
        expected = dt.datetime(2024, 1, 1) + dt.timedelta(99 - 1)
        expected = expected.replace(hour=16, minute=9, second=13)
        assert ex.date == expected

    def test_read_file_returns_list(self, sao_file):
        ex = SaoExtractor(sao_file)
        lines = ex.read_file()
        assert isinstance(lines, list)
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# SaoExtractor.extract() — all-zero header (all sections skipped)
# ---------------------------------------------------------------------------


class TestSaoExtractorExtractZero:
    def test_extract_returns_dict(self, extractor):
        result = extractor.extract()
        assert isinstance(result, dict)

    def test_extract_empty_result(self, extractor):
        result = extractor.extract()
        assert result == {}

    def test_sao_attr_set(self, extractor):
        extractor.extract()
        assert hasattr(extractor, "sao")


# ---------------------------------------------------------------------------
# SaoExtractor.extract() — with actual Scaled + TH/PF/ED data
# ---------------------------------------------------------------------------

# SAO file with:
#   Dindex1: noe[3]=1  (Scaled/foF2, %8.3f)
#   Dindex2: noe[50]=2 (TH), noe[51]=2 (PF), noe[52]=2 (ED)  all %8.3f
_SCALED_HDR = (
    "  0  0  0  1"
    + "  0" * 36
    + "\n"  # Dindex1: only noe[3]=1
    + "  0" * 10
    + "  2  2  2"
    + "  0" * 27
    + "\n"  # Dindex2: noe[50/51/52]=2
    + "   5.500\n"  # Scaled foF2 = 5.5 (1 item × 8 chars)
    + "  80.000 100.000\n"  # TH = [80.0, 100.0] (2 × 8 chars)
    + "   3.000   5.000\n"  # PF = [3.0, 5.0]
    + " 100.000 200.000\n"  # ED = [100.0, 200.0]
)


@pytest.fixture
def sao_file_with_data(tmp_path):
    p = tmp_path / "KR835_2024099160913.SAO"
    p.write_text(_SCALED_HDR)
    return str(p)


class TestSaoExtractorExtractData:
    def test_extract_scaled_foF2(self, sao_file_with_data):
        ex = SaoExtractor(sao_file_with_data)
        result = ex.extract()
        assert "Scaled" in result
        assert result["Scaled"]["foF2"] == pytest.approx(5.5)

    def test_extract_th_values(self, sao_file_with_data):
        ex = SaoExtractor(sao_file_with_data)
        result = ex.extract()
        assert "TH" in result
        assert len(result["TH"]) == 2
        assert result["TH"][0] == pytest.approx(80.0)

    def test_extract_pf_values(self, sao_file_with_data):
        ex = SaoExtractor(sao_file_with_data)
        result = ex.extract()
        assert "PF" in result
        assert result["PF"][0] == pytest.approx(3.0)

    def test_extract_ed_values(self, sao_file_with_data):
        ex = SaoExtractor(sao_file_with_data)
        result = ex.extract()
        assert "ED" in result
        assert result["ED"][1] == pytest.approx(200.0)

    def test_sao_namespace_has_th(self, sao_file_with_data):
        ex = SaoExtractor(sao_file_with_data)
        ex.extract()
        assert hasattr(ex.sao, "TH")
        assert hasattr(ex.sao, "PF")
        assert hasattr(ex.sao, "ED")


# ---------------------------------------------------------------------------
# SaoExtractor.get_scaled_datasets() — manual sao injection
# ---------------------------------------------------------------------------


class TestGetScaledDatasets:
    def test_returns_dataframe(self, extractor):
        extractor.extract()
        extractor.sao = to_namespace({"Scaled": {"foF2": 5.5, "hmF2": 300.0}})
        df = extractor.get_scaled_datasets()
        assert isinstance(df, pd.DataFrame)

    def test_columns_from_scaled_keys(self, extractor):
        extractor.extract()
        extractor.sao = to_namespace({"Scaled": {"foF2": 5.5, "foF1": 4.0}})
        df = extractor.get_scaled_datasets()
        assert "foF2" in df.columns
        assert "foF1" in df.columns

    def test_replaces_9999_with_nan(self, extractor):
        extractor.extract()
        extractor.sao = to_namespace({"Scaled": {"foF2": 9999.0, "hmF2": 300.0}})
        df = extractor.get_scaled_datasets()
        assert np.isnan(df["foF2"].iloc[0])
        assert df["hmF2"].iloc[0] == pytest.approx(300.0)

    def test_datetime_column_added_when_date_present(self, tmp_path):
        p = tmp_path / "KR835_2024099160913.SAO"
        p.write_text(_ZERO_HDR)
        ex = SaoExtractor(str(p), extract_time_from_name=True)
        ex.extract()
        ex.sao = to_namespace({"Scaled": {"foF2": 5.0}})
        df = ex.get_scaled_datasets()
        assert "datetime" in df.columns

    def test_string_values_replaced_with_nan(self, extractor):
        # Scaled values that are strings get replaced with [np.nan]
        extractor.extract()
        extractor.sao = to_namespace({"Scaled": {"foF2": "missing"}})
        df = extractor.get_scaled_datasets()
        assert np.isnan(df["foF2"].iloc[0])


# ---------------------------------------------------------------------------
# SaoExtractor.get_height_profile() — manual sao injection
# ---------------------------------------------------------------------------


class TestGetHeightProfile:
    def _set_sao_data(self, extractor):
        """Inject TH/PF/ED and stn_info needed by get_height_profile."""
        extractor.extract()
        extractor.sao = to_namespace(
            {
                "TH": [100.0, 150.0, 200.0],
                "PF": [3.5, 4.0, 5.0],
                "ED": [1e10, 1.5e10, 2e10],
            }
        )
        extractor.stn_info = {"LAT": 55.8, "LONG": 48.8}

    def test_returns_dataframe(self, extractor):
        self._set_sao_data(extractor)
        df = extractor.get_height_profile()
        assert isinstance(df, pd.DataFrame)

    def test_th_column_present(self, extractor):
        self._set_sao_data(extractor)
        df = extractor.get_height_profile()
        assert "th" in df.columns

    def test_pf_column_present(self, extractor):
        self._set_sao_data(extractor)
        df = extractor.get_height_profile()
        assert "pf" in df.columns

    def test_ed_column_present(self, extractor):
        self._set_sao_data(extractor)
        df = extractor.get_height_profile()
        assert "ed" in df.columns

    def test_height_values_correct(self, extractor):
        self._set_sao_data(extractor)
        df = extractor.get_height_profile()
        assert df["th"].iloc[0] == pytest.approx(100.0)
        assert df["th"].iloc[2] == pytest.approx(200.0)

    def test_lat_lon_columns(self, extractor):
        self._set_sao_data(extractor)
        df = extractor.get_height_profile()
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert df["lat"].iloc[0] == pytest.approx(55.8)

    def test_empty_df_when_no_th(self, extractor):
        # When TH/PF/ED are absent, returns empty DataFrame
        extractor.extract()
        extractor.sao = to_namespace({})
        df = extractor.get_height_profile()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_datetime_column_when_date_set(self, tmp_path):
        p = tmp_path / "KR835_2024099160913.SAO"
        p.write_text(_ZERO_HDR)
        ex = SaoExtractor(str(p), extract_time_from_name=True)
        ex.extract()
        ex.sao = to_namespace(
            {
                "TH": [100.0, 200.0],
                "PF": [3.0, 4.0],
                "ED": [1e10, 2e10],
            }
        )
        ex.stn_info = {"LAT": 55.8, "LONG": 48.8}
        df = ex.get_height_profile()
        assert "datetime" in df.columns
