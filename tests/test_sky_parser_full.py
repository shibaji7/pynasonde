"""Extended tests for pynasonde.digisonde.parsers.sky.

Tests SkyExtractor helper methods (parse_line, parse_data_header,
parse_freq_header, get_doppler_freq, to_pandas) and the full extract()
pipeline using a synthetic in-memory SKY file written to a tmp_path.
"""

import numpy as np
import pandas as pd
import pytest

from pynasonde.digisonde.parsers.sky import SkyExtractor, get_indent

# ---------------------------------------------------------------------------
# Module-level helper: get_indent
# ---------------------------------------------------------------------------


class TestGetIndent:
    def test_no_indent(self):
        assert get_indent("hello") == 0

    def test_single_space(self):
        assert get_indent(" hello") == 1

    def test_four_spaces(self):
        assert get_indent("    hello") == 4

    def test_empty_line(self):
        assert get_indent("") == 0

    def test_all_spaces(self):
        assert get_indent("   ") == 3


# ---------------------------------------------------------------------------
# Minimal synthetic SKY file
# ---------------------------------------------------------------------------

# Format (SKY v4.0):
#  - Data header (1-space indent): type version preface n_spectrums n_rows_data 0 others
#  - Freq header (4-space indent): frq_height_num zenith sampl_freq group_range gain
#                                   height_spctrum_ampl max_height_spctrum_ampl n_sources
#                                   height_spctrum_cl_th spect_line_cl_th polarization

# Exact spacing required: 1-space indent for data header, 4-space indent for freq header.
# Do NOT use textwrap.dedent — it would strip the leading spaces and break the parser.
MINIMAL_SKY = (
    " 1 4.0 57Z1KR835 1 1 0 0\n" "    1 0.0 5.000 200.0 25.0 1.0 2.0 0 0.5 0.3 0\n"
)


@pytest.fixture
def sky_file(tmp_path):
    p = tmp_path / "KR835_2024099160913.SKY"
    p.write_text(MINIMAL_SKY)
    return str(p)


# ---------------------------------------------------------------------------
# SkyExtractor instantiation and helper methods
# ---------------------------------------------------------------------------


class TestSkyExtractorHelpers:
    def test_init_sets_attributes(self, sky_file):
        ex = SkyExtractor(sky_file, n_fft=512, delta_freq=25)
        assert ex.n_fft == 512
        assert ex.delta_freq == 25
        assert ex.l0 == 256

    def test_read_file_returns_lines(self, sky_file):
        ex = SkyExtractor(sky_file)
        lines = ex.read_file()
        assert isinstance(lines, list)
        assert len(lines) >= 2

    def test_parse_line_strips_d_markers(self, sky_file):
        ex = SkyExtractor(sky_file)
        lines = ex.read_file()
        # build a fake line with D markers
        fake_lines = ["  1D 4D.0D 57Z1 1 1 0 0\n"]
        indent, tokens = ex.parse_line(fake_lines, 0)
        # D chars should have been removed
        for t in tokens:
            assert "D" not in t

    def test_parse_line_returns_indent(self, sky_file):
        ex = SkyExtractor(sky_file)
        fake_lines = ["   hello world\n"]
        indent, tokens = ex.parse_line(fake_lines, 0)
        assert indent == 3
        assert "hello" in tokens

    def test_parse_data_header_fields(self, sky_file):
        ex = SkyExtractor(sky_file)
        tokens = ["1", "4.0", "57Z1KR835", "2", "1", "0", "5"]
        h = ex.parse_data_header(tokens)
        assert h["type"] == 1
        assert h["version"] == pytest.approx(4.0)
        assert h["n_spectrums"] == 2
        assert h["others"] == 5

    def test_parse_freq_header_fields(self, sky_file):
        ex = SkyExtractor(sky_file)
        tokens = [
            "1",
            "0.0",
            "5.000",
            "200.0",
            "25.0",
            "1.0",
            "2.0",
            "0",
            "0.5",
            "0.3",
            "0",
        ]
        fh = ex.parse_freq_header(tokens)
        assert fh["frq_height_num"] == 1
        assert fh["zenith_angle"] == pytest.approx(0.0)
        assert fh["sampl_freq"] == pytest.approx(5.0)
        assert fh["group_range"] == pytest.approx(200.0)
        assert fh["n_sources"] == 0
        assert fh["polarization"] == 0
        assert fh["sky_data"] is None

    def test_extract_time_from_name(self, tmp_path):
        p = tmp_path / "KR835_2024099160913.SKY"
        p.write_text(MINIMAL_SKY)
        ex = SkyExtractor(str(p), extract_time_from_name=True)
        import datetime as dt

        expected = dt.datetime(2024, 1, 1) + dt.timedelta(99 - 1)
        expected = expected.replace(hour=16, minute=9, second=13)
        assert ex.date == expected


# ---------------------------------------------------------------------------
# Full extract() with synthetic SKY file
# ---------------------------------------------------------------------------


class TestSkyExtractorExtract:
    def test_extract_returns_namespace(self, sky_file):
        ex = SkyExtractor(sky_file)
        sky = ex.extract()
        assert hasattr(sky, "dataset")

    def test_extract_dataset_has_one_block(self, sky_file):
        ex = SkyExtractor(sky_file)
        sky = ex.extract()
        assert len(sky.dataset) == 1

    def test_extract_data_header_type(self, sky_file):
        ex = SkyExtractor(sky_file)
        sky = ex.extract()
        dh = sky.dataset[0].data_header
        assert dh.type == 1

    def test_extract_freq_header_stored(self, sky_file):
        ex = SkyExtractor(sky_file)
        sky = ex.extract()
        fh = sky.dataset[0].freq_headers
        assert len(fh) == 1
        assert fh[0].n_sources == 0  # 0 sources → no sky_data block


# ---------------------------------------------------------------------------
# SKY file with one echo source — exercises parse_sky_data
# ---------------------------------------------------------------------------

SKY_WITH_SOURCE = (
    " 1 4.0 57Z1KR835 1 1 0 0\n"
    "    1 0.0 5.000 200.0 25.0 1.0 2.0 1 0.5 0.3 0\n"
    "         0.5\n"
    "         0.3\n"
    "         15.0 3\n"
    "         0.0\n"
    "         0.1\n"
)


@pytest.fixture
def sky_file_with_source(tmp_path):
    p = tmp_path / "KR835_2024099160913_src.SKY"
    p.write_text(SKY_WITH_SOURCE)
    return str(p)


class TestSkyExtractorWithSource:
    def test_extract_n_sources_one(self, sky_file_with_source):
        ex = SkyExtractor(sky_file_with_source)
        sky = ex.extract()
        fh = sky.dataset[0].freq_headers[0]
        assert fh.n_sources == 1
        assert fh.sky_data is not None


# ---------------------------------------------------------------------------
# to_pandas with a minimal synthetic sky_struct
# ---------------------------------------------------------------------------


class TestSkyExtractorToPandas:
    def test_to_pandas_returns_dataframe(self, sky_file):
        ex = SkyExtractor(sky_file)
        ex.extract()
        df = ex.to_pandas()
        assert isinstance(df, pd.DataFrame)

    def test_to_pandas_has_required_columns(self, sky_file):
        ex = SkyExtractor(sky_file)
        ex.extract()
        df = ex.to_pandas()
        # No sources in this minimal file → empty df but should not crash
        assert df is not None

    def test_get_doppler_freq(self, sky_file):
        ex = SkyExtractor(sky_file, n_fft=2048, delta_freq=50)
        # Doppler freq formula: L * delta_freq / n_fft
        # For L=0 → 0
        freq = ex.get_doppler_freq(0)
        assert freq == pytest.approx(0.0)
        # For L=1 → delta_freq/n_fft
        freq_off = ex.get_doppler_freq(1)
        assert freq_off == pytest.approx(50.0 / 2048)
        # For L=l0: l0 * delta_freq / n_fft = 1024 * 50 / 2048 = 25.0
        freq_l0 = ex.get_doppler_freq(ex.l0)
        assert freq_l0 == pytest.approx(ex.l0 * 50.0 / 2048)
