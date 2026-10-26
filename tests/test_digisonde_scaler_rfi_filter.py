"""Tests for Digisonde scaler RFI filters on canonical echo DataFrames."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="needs pynasonde[digisonde-scaler]")
pytest.importorskip("joblib", reason="needs pynasonde[digisonde-scaler]")

from pynasonde.digisonde.scaler import (
    ColumnDensityFilter,
    NeighborPredictedFilter,
    VerticalRfiFilter,
)


def test_neighbor_predicted_filter_removes_height_outlier() -> None:
    df = pd.DataFrame(
        {
            "frequency_mhz": [1.0, 2.0, 3.0, 3.0, 4.0],
            "height_km": [100.0, 110.0, 120.0, 400.0, 130.0],
            "amplitude_db": [20.0, 20.0, 20.0, 5.0, 20.0],
            "mode": ["O"] * 5,
        }
    )

    out = NeighborPredictedFilter(
        tolerance_km=40.0,
        n_neighbors=2,
        freq_bin=1.0,
        min_neighbors=2,
    ).clean(df)

    assert 400.0 not in out["height_km"].to_list()
    assert 120.0 in out["height_km"].to_list()
    assert "npf_h_pred_km" in out.columns


def test_column_density_filter_hard_deletes_wide_column() -> None:
    df = pd.DataFrame(
        {
            "frequency_mhz": [2.0, 2.0, 2.0, 3.0, 3.0],
            "height_km": [100.0, 200.0, 350.0, 110.0, 120.0],
            "amplitude_db": [10.0] * 5,
            "mode": ["O"] * 5,
        }
    )

    out = ColumnDensityFilter(
        rfi_span_km=100.0,
        freq_bin=1.0,
        hard_delete=True,
    ).clean(df)

    assert set(out["frequency_mhz"]) == {3.0}
    assert "cdf_rfi_col" in out.columns


def test_vertical_rfi_filter_removes_all_modes_at_flagged_frequency() -> None:
    df = pd.DataFrame(
        {
            "frequency_mhz": [3.0] * 5 + [4.0, 4.0],
            "height_km": [100.0, 120.0, 140.0, 160.0, 180.0, 110.0, 130.0],
            "amplitude_db": [20.0] * 7,
            "mode": ["O", "O", "O", "O", "X", "O", "X"],
        }
    )

    out = VerticalRfiFilter(max_echoes_per_bin=3, freq_bin=1.0).clean(df)

    assert set(out["frequency_mhz"]) == {4.0}
    assert out["mode"].to_list() == ["O", "X"]
