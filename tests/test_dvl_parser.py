"""Unit tests for the DVL parser utilities."""

import pathlib

import numpy as np

from pynasonde.digisonde.parsers.dvl import DvlExtractor


def _create_sample_dvl(tmp_path: pathlib.Path) -> pathlib.Path:
    fname = tmp_path / "KR835_2024099160913.DVL"
    # Provide a single-line record with the expected 24 whitespace-separated tokens.
    content = " ".join(
        [
            "DVL",
            "1",
            "123",
            "KR835",
            "35.0",
            "-106.5",
            "2024099",
            "283",
            "12:00:00",
            "10",
            "1",
            "5",
            "1",
            "180",
            "5",
            "11",
            "1",
            "2",
            "1",
            "GEO",
            "200",
            "250",
            "5",
            "6",
        ]
    )
    fname.write_text(content)
    return fname


def test_extract_single_dvl_record(tmp_path):
    dvl_path = _create_sample_dvl(tmp_path)

    extractor = DvlExtractor(
        str(dvl_path), extract_time_from_name=True, extract_stn_from_name=True
    )
    record = extractor.extract()

    assert record["ursi_tag"] == "KR835"
    assert np.isclose(record["lat"], 35.0)
    assert np.isclose(record["lon"], -106.5)
    assert record["Cord"] == "GEO"

    df = DvlExtractor.extract_DVL_pandas(str(dvl_path))
    assert "datetime" in df.columns
    assert len(df) == 1


def test_load_dvl_files_compiles_dataframe(tmp_path):
    folder = tmp_path / "dvl"
    folder.mkdir()
    path = _create_sample_dvl(folder)

    df = DvlExtractor.load_DVL_files([str(folder)], ext="*.DVL", n_procs=1)
    assert not df.empty
    assert set(["Vx", "Vy", "Cord"]).issubset(df.columns)
