"""Focused tests for the SKY parser helper methods."""

from pathlib import Path

import numpy as np

from pynasonde.digisonde.parsers.sky import SkyExtractor, get_indent


def test_get_indent_counts_spaces():
    assert get_indent("    line") == 4
    assert get_indent("line") == 0


def test_parse_header_helpers():
    root = Path(__file__).resolve().parents[1]
    sky_file = root / "examples/data/KR835_2024099160913.SKY"

    extractor = SkyExtractor(
        str(sky_file), extract_time_from_name=True, extract_stn_from_name=True
    )
    extractor.extract()

    dataset = extractor.sky_struct["dataset"][0]
    assert dataset not in (None, {})
    assert extractor.stn_code == "KR835"

    df = extractor.to_pandas()
    assert not df.empty
    assert np.allclose(
        df["spect_dop_freq"].values,
        df["spect_dop"].values * extractor.delta_freq / extractor.n_fft,
    )
