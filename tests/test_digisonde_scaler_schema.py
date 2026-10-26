"""Tests for Digisonde scaler DataFrame schema normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from pynasonde.digisonde.scaler import (
    detect_echo_schema,
    normalize_echo_dataframe,
    normalize_mode_values,
)


def test_detect_rsf_sbf_schema() -> None:
    df = pd.DataFrame(
        {
            "frequency_reading": [3_000_000.0],
            "height": [120.0],
            "amplitude": [24.0],
            "pol": ["O"],
        }
    )

    assert detect_echo_schema(df) == "rsf"


def test_normalize_rsf_sbf_style_dataframe() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-10-25T00:00:00Z", "2024-10-25T00:00:00Z"],
            "frequency_reading": [3_000_000.0, 4_000_000.0],
            "height": [120.0, 240.0],
            "amplitude": [24.0, 30.0],
            "pol": ["O", "X"],
            "mpa": [18.0, 18.0],
            "dop_num": [1, -1],
            "phase": [12.0, 15.0],
            "azimuth": [90.0, 180.0],
        }
    )

    out = normalize_echo_dataframe(df, source_format="rsf")

    assert list(out["frequency_mhz"]) == [3.0, 4.0]
    assert list(out["height_km"]) == [120.0, 240.0]
    assert list(out["amplitude_db"]) == [24.0, 30.0]
    assert list(out["mode"]) == ["O", "X"]
    assert list(out["mpa_db"]) == [18.0, 18.0]
    assert list(out["doppler_bin"]) == [1, -1]
    assert out["time"].dt.tz is not None
    assert set(out["source_format"]) == {"rsf"}


def test_normalize_mmm_style_dataframe() -> None:
    df = pd.DataFrame(
        {
            "datetime": ["2024-10-25T00:00:00Z"],
            "frequency_mhz": [3.5],
            "range_km": [118.0],
            "amplitude_dB": [36.0],
            "polarization": ["o-mode"],
            "doppler_channel": [3],
            "doppler_hz": [0.25],
            "mpa_dB": [12.0],
        }
    )

    out = normalize_echo_dataframe(df)

    assert out.loc[0, "frequency_mhz"] == pytest.approx(3.5)
    assert out.loc[0, "height_km"] == pytest.approx(118.0)
    assert out.loc[0, "amplitude_db"] == pytest.approx(36.0)
    assert out.loc[0, "mode"] == "O"
    assert out.loc[0, "doppler_bin"] == pytest.approx(3)
    assert out.loc[0, "doppler_hz"] == pytest.approx(0.25)
    assert out.loc[0, "source_format"] == "mmm"


def test_normalize_canonical_dataframe_preserves_schema() -> None:
    df = pd.DataFrame(
        {
            "time": ["2024-10-25T00:00:00Z"],
            "frequency_mhz": [5.0],
            "height_km": [250.0],
            "amplitude_db": [42.0],
            "mode": ["x"],
        }
    )

    out = normalize_echo_dataframe(df)

    assert out.loc[0, "frequency_mhz"] == pytest.approx(5.0)
    assert out.loc[0, "height_km"] == pytest.approx(250.0)
    assert out.loc[0, "amplitude_db"] == pytest.approx(42.0)
    assert out.loc[0, "mode"] == "X"
    assert out.loc[0, "source_format"] == "canonical"


def test_mode_filter_keeps_requested_mode_only() -> None:
    df = pd.DataFrame(
        {
            "frequency_reading": [3_000_000.0, 3_000_000.0],
            "height": [120.0, 122.0],
            "amplitude": [24.0, 20.0],
            "pol": ["O", "X"],
        }
    )

    out = normalize_echo_dataframe(df, mode="O")

    assert len(out) == 1
    assert out.loc[0, "mode"] == "O"


def test_custom_column_map() -> None:
    df = pd.DataFrame(
        {
            "f_mhz": [2.5],
            "virtual_height": [110.0],
            "power": [10.0],
            "trace": ["O"],
        }
    )

    out = normalize_echo_dataframe(
        df,
        source_format="canonical",
        col_map={
            "frequency_mhz": "f_mhz",
            "height_km": "virtual_height",
            "amplitude_db": "power",
            "mode": "trace",
        },
    )

    assert out.loc[0, "frequency_mhz"] == pytest.approx(2.5)
    assert out.loc[0, "height_km"] == pytest.approx(110.0)
    assert out.loc[0, "amplitude_db"] == pytest.approx(10.0)
    assert out.loc[0, "mode"] == "O"


def test_unsupported_schema_raises() -> None:
    df = pd.DataFrame({"frequency": [1.0], "height": [100.0]})

    with pytest.raises(ValueError, match="Could not infer"):
        normalize_echo_dataframe(df)


def test_normalize_mode_values() -> None:
    values = pd.Series(["o", "X", "O-mode", "x mode"])

    assert list(normalize_mode_values(values)) == ["O", "X", "O", "X"]
