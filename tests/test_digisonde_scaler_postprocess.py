"""Tests for Digisonde scaler post-processing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pynasonde.digisonde.scaler import despike_scaled_parameters


def test_despike_scaled_parameters_masks_rolling_outlier() -> None:
    df = pd.DataFrame(
        {
            "foF2_mhz": [5.0, 5.1, 5.0, 8.0, 5.1, 5.0, 5.1],
            "h_prime_F2_km": [250.0, 252.0, 251.0, 320.0, 252.0, 250.0, 251.0],
        }
    )

    out = despike_scaled_parameters(
        df,
        col_thresholds={"foF2_mhz": 0.5, "h_prime_F2_km": 20.0},
        window=5,
    )

    assert np.isnan(out.loc[3, "foF2_mhz"])
    assert np.isnan(out.loc[3, "h_prime_F2_km"])
    assert out.loc[2, "foF2_mhz"] == 5.0
