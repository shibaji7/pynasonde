"""Extra SAO parser tests targeting uncovered branches in sao.py."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pynasonde.digisonde.digi_utils import to_namespace
from pynasonde.digisonde.parsers.sao import SaoExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# __init__: xml_file branch (lines 70-72)
# ---------------------------------------------------------------------------


class TestSaoExtractorXmlInit:
    def test_xml_file_flag_set(self, tmp_path):
        """Filename ending in .xml triggers xml_file=True branch."""
        xml_path = str(tmp_path / "KR835_2024099_160913.xml")
        ex = SaoExtractor(xml_path)
        assert ex.xml_file is True
        assert ex.stn_code == "KR835"

    def test_sao_file_flag_not_set(self, sao_file):
        ex = SaoExtractor(sao_file)
        assert ex.xml_file is False


# ---------------------------------------------------------------------------
# display_struct (lines 600-601)
# ---------------------------------------------------------------------------


class TestDisplayStruct:
    def test_no_raise(self, extractor, sao_file):
        extractor.extract()
        extractor.display_struct()  # just logs, should not raise


# ---------------------------------------------------------------------------
# get_scaled_datasets with local_time present (line 547)
# ---------------------------------------------------------------------------


class TestGetScaledDatasetsLocalTime:
    def test_local_time_column_added(self, extractor):
        extractor.sao = to_namespace({"Scaled": {"foF2": 5.5, "hmF2": 300.0}})
        extractor.date = dt.datetime(2024, 4, 9, 12, 0, 0)
        extractor.local_time = dt.datetime(2024, 4, 9, 8, 0, 0)
        df = extractor.get_scaled_datasets()
        assert "local_datetime" in df.columns
        assert "datetime" in df.columns


# ---------------------------------------------------------------------------
# get_height_profile branches (lines 566-591)
# ---------------------------------------------------------------------------


class TestGetHeightProfileBranches:
    def _set_sao(self, extractor, *, th, pf, ed):
        extractor.sao = to_namespace({"TH": th, "PF": pf, "ED": ed})

    def test_pf_length_mismatch_skips_pf(self, extractor):
        """When len(PF) != len(TH), pf column is not added (line 572)."""
        self._set_sao(
            extractor,
            th=[100.0, 200.0, 300.0],
            pf=[5.0, 6.0],  # wrong length
            ed=[1e5, 2e5, 3e5],
        )
        extractor.stn_info = {"LAT": 55.0, "LONG": 48.0}
        df = extractor.get_height_profile()
        assert "th" in df.columns
        assert "pf" not in df.columns

    def test_ed_length_mismatch_skips_ed(self, extractor):
        """When len(ED) != len(TH), ed column is not added (line 575)."""
        self._set_sao(
            extractor, th=[100.0, 200.0], pf=[5.0, 6.0], ed=[1e5]
        )  # wrong length
        extractor.stn_info = {"LAT": 55.0, "LONG": 48.0}
        df = extractor.get_height_profile()
        assert "pf" in df.columns
        assert "ed" not in df.columns

    def test_with_date_and_local_time(self, extractor):
        """date and local_time present → datetime/local_datetime columns (lines 579-582)."""
        self._set_sao(extractor, th=[100.0, 200.0], pf=[5.0, 6.0], ed=[1e5, 2e5])
        extractor.stn_info = {"LAT": 55.0, "LONG": 48.0}
        extractor.date = dt.datetime(2024, 4, 9, 12, 0, 0)
        extractor.local_time = dt.datetime(2024, 4, 9, 9, 0, 0)
        df = extractor.get_height_profile()
        assert "datetime" in df.columns
        assert "local_datetime" in df.columns

    def test_empty_when_no_pf_th_ed(self, extractor):
        """When sao has no TH/PF/ED, returns empty DataFrame."""
        extractor.sao = to_namespace({})
        extractor.stn_info = {"LAT": 55.0, "LONG": 48.0}
        df = extractor.get_height_profile()
        assert df.empty


# ---------------------------------------------------------------------------
# extract_SAO static method (lines 626-643)
# ---------------------------------------------------------------------------


class TestExtractSaoStatic:
    def test_scaled_func_name(self, tmp_path):
        """extract_SAO with func_name='scaled' calls get_scaled_datasets()."""
        # Build a minimal SAO with noe[3]=1 so Scaled is extracted
        # Dindex1 field 3 = 1 → one scaled value
        hdr = "  0  0  0  1" + "  0" * 36 + "\n" + "  0" * 40 + "\n"
        data = "   5.500\n"
        sao_path = tmp_path / "KR835_2024099160913.SAO"
        sao_path.write_text(hdr + data)
        df = SaoExtractor.extract_SAO(
            str(sao_path),
            extract_time_from_name=False,
            extract_stn_from_name=False,
            func_name="scaled",
        )
        assert isinstance(df, pd.DataFrame)

    def test_unknown_func_name_returns_empty(self, sao_file):
        """func_name that is not 'height_profile' or 'scaled' → empty DataFrame."""
        df = SaoExtractor.extract_SAO(
            sao_file,
            extract_time_from_name=False,
            extract_stn_from_name=False,
            func_name="unknown_view",
        )
        assert df.empty

    def test_height_profile_func_name(self, sao_file):
        """func_name='height_profile' (default) returns DataFrame."""
        df = SaoExtractor.extract_SAO(
            sao_file,
            extract_time_from_name=False,
            extract_stn_from_name=False,
            func_name="height_profile",
        )
        assert isinstance(df, pd.DataFrame)
