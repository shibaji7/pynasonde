"""Post-scaling utilities for Digisonde scaler outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def despike_scaled_parameters(
    df: pd.DataFrame,
    *,
    col_thresholds: dict[str, float] | None = None,
    window: int = 15,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Mask outlier scaled parameters using rolling-median deviations.

    Args:
        df: Scaled-parameter table sorted by time.
        col_thresholds: Mapping of column name to maximum allowed absolute
            deviation from the centered rolling median. Defaults cover common
            frequency and height products.
        window: Rolling window in rows.
        min_periods: Minimum valid rows needed for a rolling median.

    Returns:
        Copy of ``df`` with outlier cells replaced by ``NaN``.
    """
    if col_thresholds is None:
        col_thresholds = {
            "foF2_mhz": 0.4,
            "foF1_mhz": 0.4,
            "foE_mhz": 0.4,
            "foEs_mhz": 0.4,
            "h_prime_F2_km": 20.0,
            "h_prime_F1_km": 20.0,
            "h_prime_E_km": 20.0,
            "h_prime_Es_km": 20.0,
            "hmF2_km": 20.0,
        }

    out = df.copy()
    for col, threshold in col_thresholds.items():
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        roll_med = values.rolling(
            window,
            center=True,
            min_periods=min_periods,
        ).median()
        outlier = np.abs(values - roll_med) > threshold
        n_masked = int(outlier.sum())
        if n_masked:
            logger.debug(
                "despike_scaled_parameters: masked "
                f"{n_masked} outliers in {col!r} with threshold={threshold}"
            )
        out.loc[outlier, col] = np.nan
    return out
