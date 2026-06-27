"""VIPIR RIQ adapter for NN-POLAN.

Converts a filtered VIPIR echo DataFrame (output of EchoExtractor +
IonogramFilter) into a list of (hv_obs, obs_mask, cond) records ready for
NNInversion or Stage 2 training.

Expected DataFrame columns
--------------------------
    time          : datetime-like — ionogram timestamp
    frequency_khz : float — sounding frequency [kHz]
    height_km     : float — virtual height [km]
    lat, lon      : float — station coordinates [deg]  (optional, fall back to args)
    kp            : float — Kp index (optional, default 2.0)
    f107          : float — F10.7 flux [sfu] (optional, default 130.0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pynasonde.nn_inversion.adapters.base import resample_trace
from pynasonde.nn_inversion.forward_model import F_GRID_MHZ


class VipirAdapter:
    """Convert VIPIR echo DataFrame → NN-POLAN trace records.

    Parameters
    ----------
    station_lat : float — fallback station latitude  [deg]
    station_lon : float — fallback station longitude [deg]
    min_points  : int   — minimum valid grid points to keep an ionogram
    """

    def __init__(
        self,
        station_lat: float = 0.0,
        station_lon: float = 0.0,
        min_points: int = 5,
    ) -> None:
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.min_points = min_points

    def to_records(self, df: pd.DataFrame) -> list[dict]:
        """Convert echo DataFrame to list of trace records.

        Returns
        -------
        list of dict with keys:
            hv_obs   : (N_f,) float32 — interpolated virtual height [km]
            obs_mask : (N_f,) bool
            cond     : (6,)  float32  — [lat, lon, doy, ut_h, Kp, F10.7]
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        records = []
        for ts, grp in df.groupby("time"):
            freq_mhz = grp["frequency_khz"].to_numpy() / 1e3
            h_km = grp["height_km"].to_numpy()

            hv_obs, obs_mask = resample_trace(freq_mhz, h_km, F_GRID_MHZ)
            if obs_mask.sum() < self.min_points:
                continue

            lat = (
                float(grp["lat"].iloc[0]) if "lat" in grp.columns else self.station_lat
            )
            lon = (
                float(grp["lon"].iloc[0]) if "lon" in grp.columns else self.station_lon
            )
            kp = float(grp["kp"].iloc[0]) if "kp" in grp.columns else 2.0
            f107 = float(grp["f107"].iloc[0]) if "f107" in grp.columns else 130.0
            doy = float(ts.day_of_year)
            ut_h = float(ts.hour) + float(ts.minute) / 60.0

            cond = np.array([lat, lon, doy, ut_h, kp, f107], dtype=np.float32)
            records.append(dict(hv_obs=hv_obs, obs_mask=obs_mask, cond=cond))

        return records
