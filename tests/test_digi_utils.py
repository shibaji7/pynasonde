"""Unit tests targeting helpers in ``pynasonde.digisonde.digi_utils``."""

import numpy as np
import pandas as pd
import pytest

from pynasonde.digisonde import digi_utils


def test_to_namespace_nested_conversion():
    payload = {
        "station": {"code": "KR835", "meta": {"country": "US"}},
        "frequencies": [{"foF2": 5.2}, {"foF2": 5.4}],
    }

    result = digi_utils.to_namespace(payload)
    assert result.station.code == "KR835"
    assert result.station.meta.country == "US"
    assert result.frequencies[1].foF2 == pytest.approx(5.4)


def test_get_digisonde_info_longitude_wrapping():
    info = digi_utils.get_digisonde_info("KR835")
    assert info["STATIONNAME"].upper() == "KIRTLAND"
    assert info["LONG"] == pytest.approx(-106.53, rel=1e-2)


def test_get_gridded_parameters_rounding_modes():
    data = pd.DataFrame(
        {
            "time": [0.1, 0.2, 0.2, 0.1],
            "height": [100.1, 100.4, 100.6, 100.9],
            "value": [1, 2, 3, 4],
        }
    )

    X, Y, Z = digi_utils.get_gridded_parameters(data, "time", "height", "value")
    assert X.shape == Y.shape == Z.T.shape

    X2, Y2, Z2 = digi_utils.get_gridded_parameters(
        data, "time", "height", "value", rounding=False
    )
    assert X2.shape == Y2.shape == Z2.T.shape


def test_load_station_csv_custom_file(tmp_path):
    csv_path = tmp_path / "stations.csv"
    csv_path.write_text("URSI,LAT,LONG\nTEST,0,200\n")

    stations = digi_utils.load_station_csv(str(csv_path))
    assert set(["URSI", "LAT", "LONG"]).issubset(stations.columns)


def test_load_dtd_file_returns_parser():
    parser = digi_utils.load_dtd_file()
    assert getattr(parser, "dtd_validation", True) is True


def test_is_valid_xml_string():
    assert digi_utils.is_valid_xml_data_string("1.0 2.0 3")
    assert not digi_utils.is_valid_xml_data_string("bad-data")
