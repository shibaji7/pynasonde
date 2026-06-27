"""Unit tests targeting helpers in ``pynasonde.digisonde.digi_utils``."""

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


def test_merge_dicts_selected_keys_all_keys_does_not_mutate_inputs():
    base = {"a": 1}
    update = {"b": 2, "c": 3}

    result = digi_utils.merge_dicts_selected_keys(base, update)

    assert result == {"a": 1, "b": 2, "c": 3}
    assert base == {"a": 1}
    assert update == {"b": 2, "c": 3}


def test_merge_dicts_selected_keys_subset_preserves_order():
    result = digi_utils.merge_dicts_selected_keys(
        {"a": 1}, {"b": 2, "c": 3, "d": 4}, keys=["d", "b"]
    )

    assert list(result) == ["a", "d", "b"]
    assert result == {"a": 1, "d": 4, "b": 2}


def test_merge_dicts_selected_keys_missing_key_raises():
    with pytest.raises(KeyError):
        digi_utils.merge_dicts_selected_keys({}, {"a": 1}, keys=["missing"])


def test_flatten_dict_flattens_nested_mappings_only():
    payload = {
        "header": {"year": 2024, "station": {"code": "KR835"}},
        "values": [1, 2, 3],
    }

    result = digi_utils.flatten_dict(payload)

    assert result == {
        "header_year": 2024,
        "header_station_code": "KR835",
        "values": [1, 2, 3],
    }


def test_flatten_dict_custom_separator():
    assert digi_utils.flatten_dict({"a": {"b": 1}}, sep=".") == {"a.b": 1}


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


def test_split_packed_byte_5_3():
    assert digi_utils.split_packed_byte(0b00101011, 5, 3) == (5, 3)
    assert digi_utils.unpack_5_3_byte(0xFF) == [31, 7]


def test_unpack_bcd_byte_formats():
    assert digi_utils.unpack_bcd_byte(0x42) == 42
    assert digi_utils.unpack_bcd_byte(0xAB, format="tuple") == (10, 11)


def test_packed_byte_helpers_validate_byte_range():
    with pytest.raises(ValueError, match="Expected byte value"):
        digi_utils.unpack_bcd_byte(256)


def test_read_exact_raises_on_short_stream():
    from io import BytesIO

    with pytest.raises(EOFError, match="Expected 2 bytes"):
        digi_utils.read_exact(BytesIO(b"\x01"), 2)


def test_extract_station_and_datetime_token_from_filename():
    filename = "/tmp/KR835_20241025(299).SAO"

    assert digi_utils.extract_station_code_from_filename(filename) == "KR835"
    assert digi_utils.extract_datetime_token_from_filename(filename) == "20241025(299)"


@pytest.mark.parametrize(
    "token, expected",
    [
        ("20241025123456", pd.Timestamp("2024-10-25T12:34:56").to_pydatetime()),
        ("2024299123456", pd.Timestamp("2024-10-25T12:34:56").to_pydatetime()),
        ("20241025(299)", pd.Timestamp("2024-10-25").to_pydatetime()),
        ("20241025", pd.Timestamp("2024-10-25").to_pydatetime()),
        ("2024299", pd.Timestamp("2024-10-25").to_pydatetime()),
    ],
)
def test_parse_digisonde_datetime_token_supported_formats(token, expected):
    assert digi_utils.parse_digisonde_datetime_token(token) == expected


def test_parse_digisonde_datetime_token_invalid_returns_none():
    assert digi_utils.parse_digisonde_datetime_token("not-a-date") is None


def test_apply_filename_metadata_station_only():
    class Target:
        pass

    target = Target()
    meta = digi_utils.apply_filename_metadata(
        target,
        "KR835_2024299123456.DFT",
        extract_time_from_name=True,
        extract_stn_from_name=True,
        load_station_info=False,
    )

    assert target.stn_code == "KR835"
    assert target.date == pd.Timestamp("2024-10-25T12:34:56").to_pydatetime()
    assert meta["station"] == "KR835"
    assert meta["datetime"] == target.date


def test_collect_files_supports_multiple_patterns(tmp_path):
    (tmp_path / "a.SAO").write_text("")
    (tmp_path / "b.XML").write_text("")
    (tmp_path / "ignore.txt").write_text("")

    files = digi_utils.collect_files([str(tmp_path)], ["*.SAO", "*.XML"])

    assert [path.split("/")[-1] for path in files] == ["a.SAO", "b.XML"]


def test_load_files_to_dataframe_drops_empty_frames(tmp_path):
    (tmp_path / "a.dat").write_text("")
    (tmp_path / "b.dat").write_text("")

    def extractor(file):
        if file.endswith("a.dat"):
            return pd.DataFrame()
        return pd.DataFrame({"source": [file]})

    out = digi_utils.load_files_to_dataframe(
        folders=[str(tmp_path)],
        exts="*.dat",
        extractor=extractor,
        n_procs=1,
    )

    assert len(out) == 1
    assert out["source"].iloc[0].endswith("b.dat")


def test_load_files_to_dataframe_empty_search_returns_empty(tmp_path):
    out = digi_utils.load_files_to_dataframe(
        folders=[str(tmp_path)],
        exts="*.missing",
        extractor=lambda file: pd.DataFrame({"source": [file]}),
        n_procs=1,
    )

    assert out.empty
