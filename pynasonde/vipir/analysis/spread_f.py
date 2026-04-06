"""spread_f.py — Spread-F detection and characterisation.

Spread-F refers to the diffuse scattering of radio waves from irregular
structures in the F-layer, producing a "spread" appearance on the ionogram
instead of a clean single-layer trace.  Two principal manifestations exist:

* **Range spread-F** — echoes at a given frequency are spread over a wide range
  of virtual heights (> ~100 km IQR).  Caused by large-scale irregularities
  in the bottomside F-layer.

* **Frequency spread-F** — echoes persist beyond the critical frequency foF2
  (``fsF2 > foF2``).  Caused by field-aligned irregularities that scatter the
  signal obliquely, allowing returns above the vertical-incidence critical
  frequency.

* **Mixed spread-F** — both height and frequency spreading are present
  simultaneously.

This module provides:

:class:`SpreadFAnalyzer`
    Processor — computes spread-F metrics from a filtered echo DataFrame and
    (optionally) a mode-labelled DataFrame produced by
    :class:`~pynasonde.vipir.analysis.polarization.PolarizationClassifier`.

:class:`SpreadFResult`
    Output dataclass — holds the classification, scalar metrics, and a
    per-height-bin EP statistics table.

References
----------
Aarons, J. (1993). The longitudinal morphology of equatorial F-layer
irregularities relevant to their occurrence. *Space Science Reviews*, 63,
209–243.

Hysell, D. L. (2000). An overview and synthesis of plasma irregularities in
equatorial spread F. *Journal of Atmospheric and Solar-Terrestrial Physics*,
62, 1037–1056.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLASSIFICATION_NONE = "none"
_CLASSIFICATION_RANGE = "range"
_CLASSIFICATION_FREQUENCY = "frequency"
_CLASSIFICATION_MIXED = "mixed"


# ===========================================================================
# Output dataclass
# ===========================================================================


@dataclass
class SpreadFResult:
    """Spread-F detection and characterisation for one ionogram sounding.

    Parameters
    ----------
    classification:
        One of ``"none"``, ``"range"``, ``"frequency"``, or ``"mixed"``.
    freq_spread_mhz:
        ``fsF2 − foF2`` (MHz).  Positive values indicate frequency spread-F.
        ``NaN`` when foF2 could not be determined.
    height_iqr_km:
        Median inter-quartile range of echo heights across all F-layer
        frequency steps (km).  Large values indicate range spread-F.
    spread_onset_freq_mhz:
        Frequency (MHz) at which height spreading first exceeds the threshold.
        ``NaN`` when no range spread-F is detected.
    fo_f2_mhz:
        Estimated foF2 used as the reference for frequency-spread assessment
        (MHz).  ``NaN`` when insufficient O-mode echoes were available.
    ep_by_height: pd.DataFrame
        Columns: ``height_bin_km``, ``ep_mean_deg``, ``ep_std_deg``,
        ``n_echoes``.  One row per height bin.
    range_spread_flags: pd.DataFrame
        Columns: ``frequency_mhz``, ``height_iqr_km``, ``is_spread``.
        One row per frequency step in the F-layer.
    """

    classification: str
    freq_spread_mhz: float
    height_iqr_km: float
    spread_onset_freq_mhz: float
    fo_f2_mhz: float
    ep_by_height: pd.DataFrame
    range_spread_flags: pd.DataFrame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return the per-height-bin EP statistics table."""
        return self.ep_by_height.copy()

    def summary(self) -> str:
        """One-line text summary."""
        return (
            f"SpreadFResult: classification='{self.classification}'  "
            f"foF2={self.fo_f2_mhz:.2f} MHz  "
            f"freq_spread={self.freq_spread_mhz:.2f} MHz  "
            f"height_IQR={self.height_iqr_km:.1f} km"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot height IQR vs frequency and EP mean vs height bin.

        Parameters
        ----------
        ax:
            Existing axes.  A new figure is created when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        else:
            axes = [ax, ax]

        # ── left panel: height IQR vs frequency ────────────────────────
        ax0 = axes[0]
        if not self.range_spread_flags.empty:
            freq = self.range_spread_flags["frequency_mhz"]
            iqr = self.range_spread_flags["height_iqr_km"]
            spread = self.range_spread_flags["is_spread"]
            ax0.bar(
                freq[~spread],
                iqr[~spread],
                width=0.05,
                color="steelblue",
                alpha=0.8,
                label="normal",
            )
            ax0.bar(
                freq[spread],
                iqr[spread],
                width=0.05,
                color="tab:red",
                alpha=0.8,
                label="spread-F",
            )
        ax0.set_xlabel("Frequency (MHz)")
        ax0.set_ylabel("Height IQR (km)")
        ax0.set_title(f"Range spread-F  [{self.classification}]")
        ax0.legend(fontsize=8)

        # ── right panel: EP mean vs height bin ─────────────────────────
        ax1 = axes[1]
        if not self.ep_by_height.empty:
            h = self.ep_by_height["height_bin_km"]
            ep = self.ep_by_height["ep_mean_deg"]
            eps = self.ep_by_height["ep_std_deg"]
            ax1.plot(ep, h, "o-", color="darkorange", ms=4)
            ax1.fill_betweenx(h, ep - eps, ep + eps, alpha=0.25, color="darkorange")
        ax1.set_xlabel("Mean EP  (degrees)")
        ax1.set_ylabel("Virtual height (km)")
        ax1.set_title("EP irregularity proxy")

        if ax is None:
            plt.tight_layout()
        return axes[0]


# ===========================================================================
# Processor class
# ===========================================================================


class SpreadFAnalyzer:
    """Detect and characterise spread-F from a filtered echo DataFrame.

    Parameters
    ----------
    e_layer_height_range_km:
        ``(min, max)`` height window used to isolate E-layer echoes (km).
        Default ``(90, 160)``.
    f_layer_height_range_km:
        ``(min, max)`` height window used to isolate F-layer echoes (km).
        Default ``(160, 800)``.
    height_spread_threshold_km:
        An F-layer frequency step is flagged as range-spread when its echo
        height IQR exceeds this value (km).  Default ``100.0``.
    freq_spread_threshold_mhz:
        Frequency spread is reported when ``fsF2 − foF2`` exceeds this value
        (MHz).  Default ``0.5``.
    height_bin_km:
        Bin size for the per-height EP statistics table (km).  Default ``50.0``.
    mode_col:
        Name of the wave-mode column in the DataFrame (added by
        :class:`~pynasonde.vipir.analysis.polarization.PolarizationClassifier`).
        When the column is absent all echoes are treated as O-mode.
        Default ``"mode"``.
    min_echoes_per_freq:
        Minimum number of echoes at a frequency step before the height-IQR
        test is applied.  Default ``3``.

    Examples
    --------
    >>> from pynasonde.vipir.analysis.polarization import PolarizationClassifier
    >>> from pynasonde.vipir.analysis.spread_f import SpreadFAnalyzer
    >>> pol  = PolarizationClassifier().fit(df)
    >>> sfr  = SpreadFAnalyzer().fit(pol.annotated_df)
    >>> print(sfr.summary())
    """

    def __init__(
        self,
        e_layer_height_range_km: tuple = (90, 160),
        f_layer_height_range_km: tuple = (160, 800),
        height_spread_threshold_km: float = 100.0,
        freq_spread_threshold_mhz: float = 0.5,
        height_bin_km: float = 50.0,
        mode_col: str = "mode",
        min_echoes_per_freq: int = 3,
    ) -> None:
        self.e_height_range = e_layer_height_range_km
        self.f_height_range = f_layer_height_range_km
        self.height_spread_thr = height_spread_threshold_km
        self.freq_spread_thr = freq_spread_threshold_mhz
        self.height_bin_km = height_bin_km
        self.mode_col = mode_col
        self.min_echoes_per_freq = min_echoes_per_freq

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_o_mode_echoes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return O-mode (or all, if mode column absent) F-layer echoes."""
        if self.mode_col in df.columns:
            mask = df[self.mode_col] == "O"
        else:
            mask = pd.Series(True, index=df.index)
        h_lo, h_hi = self.f_height_range
        return df[mask & df["height_km"].between(h_lo, h_hi)].copy()

    def _fo_f2(self, f_echoes: pd.DataFrame) -> float:
        """Estimate foF2 as the maximum O-mode frequency in the F-layer."""
        if f_echoes.empty:
            return np.nan
        return float(f_echoes["frequency_khz"].max()) / 1e3  # → MHz

    def _fs_f2(self, df: pd.DataFrame) -> float:
        """Estimate fsF2 as the maximum frequency of any echo above F-layer bottom."""
        h_lo = self.f_height_range[0]
        f_echoes = df[df["height_km"] >= h_lo]
        if f_echoes.empty:
            return np.nan
        return float(f_echoes["frequency_khz"].max()) / 1e3  # → MHz

    def _range_spread_flags(self, f_echoes: pd.DataFrame) -> pd.DataFrame:
        """Compute height IQR per frequency step and flag spread steps."""
        if f_echoes.empty:
            return pd.DataFrame(columns=["frequency_mhz", "height_iqr_km", "is_spread"])
        rows = []
        for freq_khz, grp in f_echoes.groupby("frequency_khz"):
            if len(grp) < self.min_echoes_per_freq:
                continue
            iqr = float(
                grp["height_km"].quantile(0.75) - grp["height_km"].quantile(0.25)
            )
            rows.append(
                {
                    "frequency_mhz": freq_khz / 1e3,
                    "height_iqr_km": iqr,
                    "is_spread": iqr > self.height_spread_thr,
                }
            )
        return pd.DataFrame(rows)

    def _ep_by_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean/std of EP per height bin across all echoes."""
        if "residual_deg" not in df.columns or df.empty:
            return pd.DataFrame(
                columns=["height_bin_km", "ep_mean_deg", "ep_std_deg", "n_echoes"]
            )
        h_max = df["height_km"].max()
        bins = np.arange(
            self.f_height_range[0], h_max + self.height_bin_km, self.height_bin_km
        )
        df = df.copy()
        df["height_bin_km"] = pd.cut(
            df["height_km"],
            bins=bins,
            labels=bins[:-1] + self.height_bin_km / 2,
        ).astype(float)
        agg = (
            df.dropna(subset=["height_bin_km", "residual_deg"])
            .groupby("height_bin_km")["residual_deg"]
            .agg(ep_mean_deg="mean", ep_std_deg="std", n_echoes="count")
            .reset_index()
        )
        return agg

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> SpreadFResult:
        """Run spread-F analysis on a filtered echo DataFrame.

        Parameters
        ----------
        df:
            Echo DataFrame — must contain ``frequency_khz``, ``height_km``,
            and optionally ``residual_deg`` and ``mode`` columns.

        Returns
        -------
        SpreadFResult
        """
        for col in ("frequency_khz", "height_km"):
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in DataFrame.")

        f_echoes = self._get_o_mode_echoes(df)
        fo_f2 = self._fo_f2(f_echoes)
        fs_f2 = self._fs_f2(df)

        freq_spread = (
            (fs_f2 - fo_f2) if not (np.isnan(fo_f2) or np.isnan(fs_f2)) else np.nan
        )

        range_flags = self._range_spread_flags(f_echoes)
        ep_tbl = self._ep_by_height(df)

        # Height IQR — median across frequency steps with enough echoes
        if range_flags.empty:
            median_iqr = np.nan
        else:
            median_iqr = float(range_flags["height_iqr_km"].median())

        # Spread onset frequency (first freq step flagged as spread)
        spread_steps = (
            range_flags[range_flags["is_spread"]]
            if not range_flags.empty
            else pd.DataFrame()
        )
        onset_freq = (
            float(spread_steps["frequency_mhz"].min())
            if not spread_steps.empty
            else np.nan
        )

        # Classification
        range_spread = (not np.isnan(median_iqr)) and (
            median_iqr > self.height_spread_thr
        )
        freq_spread_flag = (not np.isnan(freq_spread)) and (
            freq_spread > self.freq_spread_thr
        )

        if range_spread and freq_spread_flag:
            classification = _CLASSIFICATION_MIXED
        elif range_spread:
            classification = _CLASSIFICATION_RANGE
        elif freq_spread_flag:
            classification = _CLASSIFICATION_FREQUENCY
        else:
            classification = _CLASSIFICATION_NONE

        logger.info(
            f"SpreadFAnalyzer: classification='{classification}'  "
            f"foF2={fo_f2:.2f} MHz  freq_spread={freq_spread:.2f} MHz  "
            f"height_IQR={median_iqr:.1f} km"
        )

        return SpreadFResult(
            classification=classification,
            freq_spread_mhz=float(freq_spread) if not np.isnan(freq_spread) else np.nan,
            height_iqr_km=float(median_iqr) if not np.isnan(median_iqr) else np.nan,
            spread_onset_freq_mhz=onset_freq,
            fo_f2_mhz=fo_f2,
            ep_by_height=ep_tbl,
            range_spread_flags=range_flags,
        )
