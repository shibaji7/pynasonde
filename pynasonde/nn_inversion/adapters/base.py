"""Shared interpolation utility for NN-POLAN instrument adapters.

Any ionosonde that produces (frequency_mhz, height_km) pairs can be mapped
onto the canonical F_GRID_MHZ via resample_trace().  Interpolation preserves
the shape of the observed trace regardless of the instrument's native step
size or frequency range.
"""

from __future__ import annotations

import numpy as np

from pynasonde.nn_inversion.forward_model import F_GRID_MHZ


def resample_trace(
    freq_mhz: np.ndarray,
    h_km: np.ndarray,
    f_grid: np.ndarray = F_GRID_MHZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a sparse ionogram trace onto the canonical frequency grid.

    Takes raw (freq, h') scatter from any ionosonde — any step size, any
    frequency range — and produces a dense trace on *f_grid* via linear
    interpolation.  Grid points outside the observed frequency range are
    marked NaN in the returned mask.

    Parameters
    ----------
    freq_mhz : (N,) array  — observed sounding frequencies [MHz]
    h_km     : (N,) array  — corresponding virtual heights  [km]
    f_grid   : (M,) array  — target frequency grid (default F_GRID_MHZ)

    Returns
    -------
    hv_obs  : (M,) float32 — interpolated virtual height [km]; NaN outside range
    obs_mask: (M,) bool    — True where interpolation is valid (within observed range)
    """
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    h_km = np.asarray(h_km, dtype=np.float64)

    # Remove NaNs and sort by frequency
    valid = np.isfinite(freq_mhz) & np.isfinite(h_km)
    if valid.sum() < 2:
        return (
            np.full(len(f_grid), np.nan, dtype=np.float32),
            np.zeros(len(f_grid), dtype=bool),
        )

    freq_sorted = freq_mhz[valid]
    h_sorted = h_km[valid]
    order = np.argsort(freq_sorted)
    freq_sorted = freq_sorted[order]
    h_sorted = h_sorted[order]

    # Deduplicate — keep median h' per unique frequency
    uniq_f, idx = np.unique(freq_sorted, return_inverse=True)
    uniq_h = np.array([np.median(h_sorted[idx == i]) for i in range(len(uniq_f))])

    # Interpolate onto f_grid; extrapolation returns NaN
    f_min, f_max = uniq_f[0], uniq_f[-1]
    hv_obs = np.interp(f_grid, uniq_f, uniq_h, left=np.nan, right=np.nan)

    obs_mask = (f_grid >= f_min) & (f_grid <= f_max) & np.isfinite(hv_obs)

    return hv_obs.astype(np.float32), obs_mask
