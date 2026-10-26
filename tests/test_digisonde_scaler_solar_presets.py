"""Tests for SZA-aware Digisonde scaler presets."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="needs pynasonde[digisonde-scaler]")
pytest.importorskip("joblib", reason="needs pynasonde[digisonde-scaler]")

from pynasonde.digisonde.scaler import (
    ScalerPreset,
    classify_solar_regime,
    get_viterbi_preset,
    scale_dataframe_with_preset,
    scale_dataframes_with_preset,
    solar_zenith_angle,
)


def _echo_dataframe() -> pd.DataFrame:
    rows = []
    for base_freq in [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]:
        base_height = 140.0 + 2.0 * (base_freq - 2.0)
        for delta in [-0.006, -0.003, 0.0, 0.003, 0.006]:
            rows.append(
                {
                    "time": "2023-10-14T18:00:00Z",
                    "frequency_mhz": base_freq + delta,
                    "height_km": base_height + delta,
                    "amplitude_db": 40.0,
                    "mpa_db": 5.0,
                    "mode": "O",
                }
            )
    for base_freq in [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]:
        base_height = 260.0
        for delta in [-0.006, -0.003, 0.0, 0.003, 0.006]:
            rows.append(
                {
                    "time": "2023-10-14T18:00:00Z",
                    "frequency_mhz": base_freq + delta,
                    "height_km": base_height + delta,
                    "amplitude_db": 40.0,
                    "mpa_db": 5.0,
                    "mode": "O",
                }
            )
    return pd.DataFrame(rows)


def test_solar_zenith_angle_classifies_day() -> None:
    sza = solar_zenith_angle(
        dt.datetime(2023, 6, 21, 18, 0, 0),
        lat_deg=35.0,
        lon_deg=-106.0,
    )

    assert 0.0 <= sza <= 180.0
    assert classify_solar_regime(sza) == "day"


def test_classify_solar_regime_boundaries() -> None:
    assert classify_solar_regime(79.9) == "day"
    assert classify_solar_regime(80.0) == "twilight"
    assert classify_solar_regime(100.0) == "night"


def test_get_viterbi_preset_uses_sza_and_tuning() -> None:
    preset = get_viterbi_preset(sza_deg=60.0, tuning_preset="e_sensitive")

    assert isinstance(preset, ScalerPreset)
    assert preset.regime == "day"
    assert preset.enabled["E"] is True
    assert preset.layer("E")[0]["name"] == "E"
    assert preset.layer("E")[0]["f_hi"] == pytest.approx(3.0)
    assert preset.kwargs("E")["consecutive_gap"] == 10


def test_get_viterbi_preset_can_disable_e_layer() -> None:
    preset = get_viterbi_preset(sza_deg=105.0, e_layer_max_sza=100.0)

    assert preset.regime == "night"
    assert preset.enabled["E"] is False
    assert preset.enabled["F2"] is True


def test_scale_dataframe_with_preset_merges_e_and_f2() -> None:
    result, details = scale_dataframe_with_preset(
        _echo_dataframe(),
        regime="day",
        tuning_preset="permissive",
        e_layer_max_sza=120.0,
        return_details=True,
    )

    assert result.foE_mhz == pytest.approx(3.006)
    assert result.foF2_mhz == pytest.approx(5.803)
    assert "E" in details["traces"]
    assert "F2" in details["traces"]
    assert details["preset"].regime == "day"


def test_scale_dataframes_with_preset_returns_metadata_rows() -> None:
    out = scale_dataframes_with_preset(
        [_echo_dataframe(), _echo_dataframe()],
        regime="day",
        tuning_preset="permissive",
    )

    assert len(out) == 2
    assert set(out["sza_regime"]) == {"day"}
    assert set(out["tuning_preset"]) == {"permissive"}
