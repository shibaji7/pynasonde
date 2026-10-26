"""Tests for generic Digisonde ionogram scaler APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="needs pynasonde[digisonde-scaler]")
pytest.importorskip("joblib", reason="needs pynasonde[digisonde-scaler]")

from pynasonde.digisonde.scaler import (
    IonoscaleResult,
    ScaleHmF2,
    ViterbiTracer,
    scale_dataframe,
    scale_dataframes,
)


def _canonical_echoes(mode: str = "O") -> pd.DataFrame:
    rows = []
    for f, h in [
        (2.0, 110.0),
        (3.0, 112.0),
        (4.0, 220.0),
        (5.0, 240.0),
        (6.0, 280.0),
        (7.0, 360.0),
    ]:
        rows.append(
            {
                "time": "2024-10-25T00:00:00Z",
                "frequency_mhz": f,
                "height_km": h,
                "amplitude_db": 40.0,
                "mode": mode,
                "mpa_db": 5.0,
            }
        )
    return pd.DataFrame(rows)


def test_scale_dataframe_ransac_returns_result() -> None:
    result, details = scale_dataframe(
        _canonical_echoes(),
        method="ransac",
        scaler_kwargs={"min_echoes": 2, "dbscan_min_samples": 1},
        return_details=True,
    )

    assert isinstance(result, IonoscaleResult)
    assert result.foE_mhz == pytest.approx(3.0)
    assert result.foF2_mhz == pytest.approx(7.0)
    assert result.h_prime_E_km == pytest.approx(110.0)
    assert not details["cleaned"].empty


def test_scale_dataframe_viterbi_returns_traces() -> None:
    result, details = scale_dataframe(
        _canonical_echoes(),
        method="viterbi",
        scaler_kwargs={"min_echoes": 2, "dbscan_min_samples": 1},
        return_details=True,
    )

    assert result.foF2_mhz == pytest.approx(7.0)
    assert "F2" in details["traces"]
    assert not details["traces"]["F2"].empty


def test_scale_dataframe_can_use_x_mode() -> None:
    o_df = _canonical_echoes("O")
    x_df = _canonical_echoes("X")
    x_df["frequency_mhz"] += 0.5
    df = pd.concat([o_df, x_df], ignore_index=True)

    result = scale_dataframe(
        df,
        mode="X",
        method="ransac",
        scaler_kwargs={"min_echoes": 2, "dbscan_min_samples": 1},
    )

    assert result.foF2_mhz == pytest.approx(7.5)


def test_scale_dataframes_returns_one_row_per_input() -> None:
    df0 = _canonical_echoes()
    df1 = _canonical_echoes()
    df1["time"] = "2024-10-25T00:05:00Z"

    out = scale_dataframes(
        [df0, df1],
        method="ransac",
        scaler_kwargs={"min_echoes": 2, "dbscan_min_samples": 1},
    )

    assert len(out) == 2
    assert list(out["foF2_mhz"]) == [7.0, 7.0]
    assert list(out["method"]) == ["ransac", "ransac"]


def test_viterbi_build_grid_uses_canonical_amplitude_column() -> None:
    grid, freq_axis, height_axis = ViterbiTracer.build_grid(
        _canonical_echoes(),
        h_lo=100.0,
        h_hi=400.0,
    )

    assert grid.shape == (6, 6)
    assert freq_axis[0] == pytest.approx(2.0)
    assert height_axis[0] == pytest.approx(110.0)


def test_scale_hmf2_calculate_m3000f2() -> None:
    m3000 = ScaleHmF2.calculate_m3000f2(
        frequencies=np.array([4.0, 5.0, 6.0]),
        virtual_heights=np.array([220.0, 260.0, 320.0]),
        fof2=6.0,
    )

    assert m3000 > 1.0
