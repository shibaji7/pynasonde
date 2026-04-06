"""irregularities.py — Small-scale ionospheric irregularity analysis via EP.

The EP (residual phase) parameter from the Dynasonde/VIPIR signal model
carries information about sub-wavelength irregularities in the ionospheric
reflection layer.  Irregularities scatter the wave coherently or
incoherently, imprinting a structured signature on EP as a function of
sounding frequency *f*.

**Structure function approach**
    The second-order structure function of EP as a function of frequency lag
    Δf is defined as:

        D_EP(Δf) = ⟨ [EP(f + Δf) − EP(f)]² ⟩

    For a power-law irregularity spectrum with spectral index α, the
    structure function follows a power law::

        D_EP(Δf) ∝ Δf^α

    A log–log fit of D_EP(Δf) vs Δf yields the spectral index α and the
    amplitude coefficient A₀.  The outer scale L_outer is estimated as the
    lag at which D_EP saturates (flattens).

**Height-resolved analysis**
    Echoes are binned by virtual height.  The structure function and spectral
    fit are computed independently for each height bin, yielding a profile of
    α(h) that maps how irregularity characteristics change with altitude
    (e.g., stronger irregularities at the F-layer base during equatorial
    spread-F events).

**Anisotropy proxy**
    When both O-mode and X-mode EP values are available for the same
    frequency step the ratio σ_EP(O)/σ_EP(X) provides a proxy for the
    anisotropy of field-aligned irregularities.  A ratio close to unity
    indicates isotropic scattering; deviations indicate anisotropy.

This module provides:

:class:`IrregularityAnalyzer`
    Processor — computes the EP structure function, spectral index, outer
    scale, and anisotropy proxy from a labelled echo DataFrame.

:class:`IrregularityProfile`
    Output dataclass — holds the structure-function table, spectral-fit
    parameters, and per-height profile.

References
----------
Hysell, D. L., & Burcham, J. D. (1998). JULIA radar studies of equatorial
spread F. *Journal of Geophysical Research*, 103(A12), 29155–29167.

Zabotin, N. A., Wright, J. W., & Zhbankov, G. A. (2006). NeXtYZ: Three-
dimensional electron density inversion for Dynasonde and ARTIST ionosondes.
*Radio Science*, 41, RS6S32.

Kintner, P. M., & Seyler, C. E. (1985). The status of observations and
theory of high latitude ionospheric and magnetospheric plasma turbulence.
*Space Science Reviews*, 41, 91–129.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EP_COL = "residual_deg"
_FREQ_COL = "frequency_khz"
_HEIGHT_COL = "height_km"
_MODE_COL = "mode"

# Minimum number of lag pairs required to attempt a power-law fit
_MIN_PAIRS_FOR_FIT = 4

# Saturation detection: the outer scale is the smallest Δf at which
# D_EP exceeds this fraction of its maximum value
_SATURATION_FRACTION = 0.85


# ===========================================================================
# Output dataclass
# ===========================================================================


@dataclass
class IrregularityProfile:
    """EP structure-function and spectral-index results for one sounding.

    Parameters
    ----------
    structure_function: pd.DataFrame
        Columns: ``delta_f_mhz``, ``D_EP_deg2``, ``n_pairs``.
        One row per frequency lag Δf.
    spectral_index:
        Power-law exponent α from log–log fit of D_EP vs Δf.
        ``NaN`` when fit failed.
    amplitude_coeff:
        Amplitude coefficient A₀ (in deg²) from the fit:
        D_EP(Δf) ≈ A₀ × Δf^α.
        ``NaN`` when fit failed.
    outer_scale_mhz:
        Estimated outer scale L_outer (MHz) — the lag at which D_EP
        first exceeds ``_SATURATION_FRACTION`` × D_EP_max.
        ``NaN`` when saturation was not observed.
    anisotropy_ratio:
        σ_EP(O-mode) / σ_EP(X-mode).  ``NaN`` when X-mode EP data
        are unavailable.
    height_profile: pd.DataFrame
        Columns: ``height_bin_km``, ``spectral_index``, ``amplitude_coeff``,
        ``outer_scale_mhz``, ``n_echoes``.
        One row per height bin.
    n_echoes_total:
        Total number of echoes used in the analysis.
    """

    structure_function: pd.DataFrame
    spectral_index: float
    amplitude_coeff: float
    outer_scale_mhz: float
    anisotropy_ratio: float
    height_profile: pd.DataFrame
    n_echoes_total: int

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return the structure-function table."""
        return self.structure_function.copy()

    def summary(self) -> str:
        """One-line text summary."""
        return (
            f"IrregularityProfile: "
            f"α={self.spectral_index:.3f}  "
            f"A₀={self.amplitude_coeff:.3f} deg²  "
            f"L_outer={self.outer_scale_mhz:.3f} MHz  "
            f"anisotropy={self.anisotropy_ratio:.2f}  "
            f"n_echoes={self.n_echoes_total}"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Two-panel plot: structure function (left) and α profile (right).

        Parameters
        ----------
        ax:
            If provided, used for the structure function only.  A new
            two-panel figure is created when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
            Left axes (structure function).
        """
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        else:
            axes = [ax, ax]

        # ── left: structure function ────────────────────────────────────
        ax0 = axes[0]
        sf = self.structure_function
        if not sf.empty and sf["D_EP_deg2"].notna().any():
            df_mhz = sf["delta_f_mhz"]
            d_ep = sf["D_EP_deg2"]
            ax0.scatter(df_mhz, d_ep, s=20, color="steelblue", zorder=3, label="D_EP")

            # Overlay power-law fit
            if not (np.isnan(self.spectral_index) or np.isnan(self.amplitude_coeff)):
                f_fit = np.linspace(df_mhz.min(), df_mhz.max(), 80)
                d_fit = self.amplitude_coeff * f_fit**self.spectral_index
                ax0.plot(
                    f_fit,
                    d_fit,
                    "--",
                    color="tab:red",
                    linewidth=1.5,
                    label=f"α = {self.spectral_index:.2f}",
                )

            # Mark outer scale
            if not np.isnan(self.outer_scale_mhz):
                ax0.axvline(
                    self.outer_scale_mhz,
                    color="grey",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"L_outer = {self.outer_scale_mhz:.3f} MHz",
                )

        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlabel("Δf  (MHz)")
        ax0.set_ylabel("D_EP  (deg²)")
        ax0.set_title("EP structure function")
        ax0.legend(fontsize=8)

        # ── right: α height profile ─────────────────────────────────────
        ax1 = axes[1]
        hp = self.height_profile
        if not hp.empty and hp["spectral_index"].notna().any():
            valid = hp.dropna(subset=["spectral_index"])
            ax1.plot(
                valid["spectral_index"],
                valid["height_bin_km"],
                "o-",
                color="darkorange",
                ms=4,
            )
        ax1.set_xlabel("Spectral index α")
        ax1.set_ylabel("Virtual height (km)")
        ax1.set_title("α vs height")

        if ax is None:
            plt.tight_layout()
        return axes[0]


# ===========================================================================
# Processor class
# ===========================================================================


class IrregularityAnalyzer:
    """Estimate ionospheric irregularity spectral properties from EP.

    The analysis requires a column ``residual_deg`` (EP) in the echo
    DataFrame, typically produced by the Dynasonde/VIPIR signal model.
    If the column is absent the processor logs a warning and returns
    empty results.

    Parameters
    ----------
    f_layer_height_range_km:
        ``(min, max)`` height window for the analysis (km).
        Default ``(160, 800)``.
    height_bin_km:
        Bin size for the height-resolved profile (km).  Default ``50.0``.
    max_delta_f_mhz:
        Maximum frequency lag Δf included in the structure function (MHz).
        Default ``2.0``.
    min_pairs_for_fit:
        Minimum number of distinct lag values with valid D_EP before
        the power-law fit is attempted.  Default ``4``.
    mode_col:
        Name of the wave-mode column (added by
        :class:`~pynasonde.vipir.analysis.polarization.PolarizationClassifier`).
        Used for the anisotropy proxy only.  Default ``"mode"``.

    Examples
    --------
    >>> from pynasonde.vipir.analysis.irregularities import IrregularityAnalyzer
    >>> irr = IrregularityAnalyzer().fit(pol.annotated_df)
    >>> print(irr.summary())
    """

    def __init__(
        self,
        f_layer_height_range_km: tuple = (160.0, 800.0),
        height_bin_km: float = 50.0,
        max_delta_f_mhz: float = 2.0,
        min_pairs_for_fit: int = _MIN_PAIRS_FOR_FIT,
        mode_col: str = _MODE_COL,
    ) -> None:
        self.f_height_range = f_layer_height_range_km
        self.height_bin_km = height_bin_km
        self.max_delta_f_mhz = max_delta_f_mhz
        self.min_pairs = min_pairs_for_fit
        self.mode_col = mode_col

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _f_layer_echoes(self, df: pd.DataFrame) -> pd.DataFrame:
        h_lo, h_hi = self.f_height_range
        return df[df[_HEIGHT_COL].between(h_lo, h_hi)].copy()

    def _structure_function(
        self, ep_series: pd.Series, freq_mhz_series: pd.Series
    ) -> pd.DataFrame:
        """Compute D_EP(Δf) from EP values and their frequencies.

        Groups echoes by frequency step, then for all pairs of frequency
        steps with lag Δf ≤ max_delta_f_mhz computes:

            D_EP(Δf) = mean over all pairs of [EP(f+Δf) − EP(f)]²

        Returns
        -------
        pd.DataFrame
            Columns: ``delta_f_mhz``, ``D_EP_deg2``, ``n_pairs``.
        """
        # Build per-frequency-step median EP table
        combined = pd.DataFrame(
            {
                "freq_mhz": freq_mhz_series.values,
                "ep_deg": ep_series.values,
            }
        ).dropna()
        if combined.empty:
            return pd.DataFrame(columns=["delta_f_mhz", "D_EP_deg2", "n_pairs"])

        grp = (
            combined.groupby("freq_mhz")["ep_deg"]
            .median()
            .reset_index()
            .sort_values("freq_mhz")
            .reset_index(drop=True)
        )
        freqs = grp["freq_mhz"].values
        eps = grp["ep_deg"].values

        lag_dict: dict[float, list[float]] = {}

        for i in range(len(freqs)):
            for j in range(i + 1, len(freqs)):
                delta_f = round(freqs[j] - freqs[i], 4)
                if delta_f > self.max_delta_f_mhz:
                    break
                sq_diff = (eps[j] - eps[i]) ** 2
                lag_dict.setdefault(delta_f, []).append(sq_diff)

        rows = []
        for delta_f, vals in sorted(lag_dict.items()):
            rows.append(
                {
                    "delta_f_mhz": delta_f,
                    "D_EP_deg2": float(np.mean(vals)),
                    "n_pairs": len(vals),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["delta_f_mhz", "D_EP_deg2", "n_pairs"])
        return pd.DataFrame(rows)

    def _power_law_fit(self, sf: pd.DataFrame) -> Tuple[float, float]:
        """Fit D_EP(Δf) = A₀ × Δf^α in log–log space.

        Returns
        -------
        (alpha, A0)  — both NaN when fit failed.
        """
        valid = sf[sf["n_pairs"] >= 1].dropna(subset=["delta_f_mhz", "D_EP_deg2"])
        valid = valid[(valid["delta_f_mhz"] > 0) & (valid["D_EP_deg2"] > 0)]
        if len(valid) < self.min_pairs:
            return np.nan, np.nan

        log_df = np.log(valid["delta_f_mhz"].values)
        log_D = np.log(valid["D_EP_deg2"].values)
        try:
            slope, intercept = np.polyfit(log_df, log_D, 1)
        except (np.linalg.LinAlgError, ValueError):
            return np.nan, np.nan
        return float(slope), float(np.exp(intercept))

    def _outer_scale(self, sf: pd.DataFrame) -> float:
        """Outer scale = lag at which D_EP first reaches 85 % of its max."""
        if sf.empty or sf["D_EP_deg2"].isna().all():
            return np.nan
        d_max = sf["D_EP_deg2"].max()
        threshold = _SATURATION_FRACTION * d_max
        saturated = sf[sf["D_EP_deg2"] >= threshold]
        if saturated.empty:
            return np.nan
        return float(saturated["delta_f_mhz"].min())

    def _anisotropy(self, df: pd.DataFrame) -> float:
        """σ_EP(O) / σ_EP(X) — requires mode column and residual_deg."""
        if self.mode_col not in df.columns or _EP_COL not in df.columns:
            return np.nan
        o_ep = df[df[self.mode_col] == "O"][_EP_COL].dropna()
        x_ep = df[df[self.mode_col] == "X"][_EP_COL].dropna()
        if len(o_ep) < 2 or len(x_ep) < 2:
            return np.nan
        sigma_o = float(o_ep.std())
        sigma_x = float(x_ep.std())
        if sigma_x == 0:
            return np.nan
        return sigma_o / sigma_x

    def _height_profile(self, f_echoes: pd.DataFrame) -> pd.DataFrame:
        """Compute spectral-index profile per height bin."""
        if _EP_COL not in f_echoes.columns or f_echoes.empty:
            return pd.DataFrame(
                columns=[
                    "height_bin_km",
                    "spectral_index",
                    "amplitude_coeff",
                    "outer_scale_mhz",
                    "n_echoes",
                ]
            )
        h_max = f_echoes[_HEIGHT_COL].max()
        bins = np.arange(
            self.f_height_range[0], h_max + self.height_bin_km, self.height_bin_km
        )
        f_echoes = f_echoes.copy()
        f_echoes["height_bin_km"] = pd.cut(
            f_echoes[_HEIGHT_COL],
            bins=bins,
            labels=bins[:-1] + self.height_bin_km / 2,
        ).astype(float)

        rows = []
        for h_bin, grp in f_echoes.dropna(subset=["height_bin_km"]).groupby(
            "height_bin_km"
        ):
            ep = grp[_EP_COL].dropna()
            frq = grp.loc[ep.index, _FREQ_COL] / 1e3
            sf = self._structure_function(ep, frq)
            alpha, a0 = self._power_law_fit(sf)
            l_out = self._outer_scale(sf)
            rows.append(
                {
                    "height_bin_km": float(h_bin),
                    "spectral_index": alpha,
                    "amplitude_coeff": a0,
                    "outer_scale_mhz": l_out,
                    "n_echoes": len(grp),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> IrregularityProfile:
        """Run irregularity analysis on an echo DataFrame.

        Parameters
        ----------
        df:
            Echo DataFrame — must contain ``frequency_khz`` and
            ``height_km``.  Should also contain ``residual_deg``
            (EP); if absent, empty results are returned with a warning.

        Returns
        -------
        IrregularityProfile
        """
        for col in (_FREQ_COL, _HEIGHT_COL):
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in DataFrame.")

        if _EP_COL not in df.columns:
            logger.warning(
                "IrregularityAnalyzer: 'residual_deg' column not found — "
                "returning empty IrregularityProfile."
            )
            empty_sf = pd.DataFrame(columns=["delta_f_mhz", "D_EP_deg2", "n_pairs"])
            empty_hp = pd.DataFrame(
                columns=[
                    "height_bin_km",
                    "spectral_index",
                    "amplitude_coeff",
                    "outer_scale_mhz",
                    "n_echoes",
                ]
            )
            return IrregularityProfile(
                structure_function=empty_sf,
                spectral_index=np.nan,
                amplitude_coeff=np.nan,
                outer_scale_mhz=np.nan,
                anisotropy_ratio=np.nan,
                height_profile=empty_hp,
                n_echoes_total=0,
            )

        f_echoes = self._f_layer_echoes(df)
        n_total = len(f_echoes)

        ep_col = f_echoes[_EP_COL].dropna()
        frq_col = f_echoes.loc[ep_col.index, _FREQ_COL] / 1e3

        sf = self._structure_function(ep_col, frq_col)
        alpha, a0 = self._power_law_fit(sf)
        l_out = self._outer_scale(sf)
        aniso = self._anisotropy(f_echoes)
        hp = self._height_profile(f_echoes)

        logger.info(
            f"IrregularityAnalyzer: α={alpha:.3f}  "
            f"A₀={a0:.3f} deg²  L_outer={l_out:.3f} MHz  "
            f"anisotropy={aniso:.2f}  n_echoes={n_total}"
        )

        return IrregularityProfile(
            structure_function=sf,
            spectral_index=alpha,
            amplitude_coeff=a0,
            outer_scale_mhz=l_out,
            anisotropy_ratio=aniso,
            height_profile=hp,
            n_echoes_total=n_total,
        )
