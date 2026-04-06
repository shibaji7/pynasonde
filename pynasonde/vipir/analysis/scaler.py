"""scaler.py — Automatic ionogram parameter scaling.

Derives the standard URSI/CCIR scaled parameters from a filtered,
O-mode-labelled echo DataFrame:

* **foE**   — ordinary-mode critical frequency of the E-layer (MHz)
* **h'E**   — minimum virtual height of E-layer echoes (km)
* **foF1**  — ordinary-mode critical frequency of the F1-layer (MHz)
              (may be absent; ``NaN`` when not detected)
* **h'F**   — minimum virtual height of F-layer echoes (km)
* **foF2**  — ordinary-mode critical frequency of the F2-layer (MHz)
* **h'F2**  — minimum virtual height of F2-layer echoes (km)
* **MUF(3000)** — maximum usable frequency for a 3 000 km path (MHz)
* **M(3000)F2** — transmission factor MUF(3000)/foF2 (dimensionless)

Bootstrap uncertainty (``foF2_sigma_mhz``, ``h_prime_F2_sigma_km``) is
estimated by resampling echoes at each layer cluster.

This module provides:

:class:`IonogramScaler`
    Processor — derives ionospheric parameters from an O-mode echo
    DataFrame (optionally pre-labelled by
    :class:`~pynasonde.vipir.analysis.polarization.PolarizationClassifier`).

:class:`ScaledParameters`
    Output dataclass — holds all scaled parameters, uncertainties, and
    quality flags.

References
----------
Reinisch, B. W., & Huang, X. (1983). Automatic calculation of electron
density profiles from digital ionograms: 3. Processing of bottomside
ionograms. *Radio Science*, 18(3), 477–492.

Piggott, W. R., & Rawer, K. (1972). *URSI Handbook of Ionogram
Interpretation and Reduction* (2nd ed.). World Data Center A for
Solar-Terrestrial Physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Height windows (km) for layer identification
_E_H_LO, _E_H_HI = 90.0, 160.0
_F1_H_LO, _F1_H_HI = 160.0, 250.0
_F2_H_LO, _F2_H_HI = 160.0, 800.0

# Frequency windows (MHz) for layer identification
_E_F_LO, _E_F_HI = 1.0, 4.5
_F2_F_LO = 2.0  # foF2 lower bound

# MUF half-path distance (km)
_D_HALF_KM = 1500.0  # half of 3 000 km

# Bootstrap parameters
_N_BOOTSTRAP = 200
_RNG_SEED = 42


# ===========================================================================
# Output dataclass
# ===========================================================================


@dataclass
class ScaledParameters:
    """Scaled ionospheric parameters for one ionogram sounding.

    Parameters
    ----------
    foE_mhz:
        E-layer critical frequency (MHz).  ``NaN`` when no E-layer
        echoes are found.
    h_prime_E_km:
        E-layer minimum virtual height (km).  ``NaN`` when absent.
    foF1_mhz:
        F1-layer critical frequency (MHz).  Often absent / ``NaN``.
    h_prime_F1_km:
        F1-layer minimum virtual height (km).  ``NaN`` when absent.
    foF2_mhz:
        F2-layer critical frequency (MHz).  ``NaN`` when absent.
    h_prime_F2_km:
        F2-layer minimum virtual height (km).  ``NaN`` when absent.
    MUF3000_mhz:
        MUF for a 3 000 km path (MHz).  ``NaN`` when foF2 is absent.
    M3000F2:
        Transmission factor MUF(3000)/foF2 (dimensionless).
        ``NaN`` when foF2 is absent.
    foF2_sigma_mhz:
        Bootstrap standard deviation of foF2 (MHz).
    h_prime_F2_sigma_km:
        Bootstrap standard deviation of h'F2 (km).
    quality_flags:
        Dict of boolean flags: ``"E_detected"``, ``"F1_detected"``,
        ``"F2_detected"``, ``"foF2_reliable"`` (≥ 5 O-mode echoes).
    """

    foE_mhz: float
    h_prime_E_km: float
    foF1_mhz: float
    h_prime_F1_km: float
    foF2_mhz: float
    h_prime_F2_km: float
    MUF3000_mhz: float
    M3000F2: float
    foF2_sigma_mhz: float
    h_prime_F2_sigma_km: float
    quality_flags: Dict[str, bool] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame of scalar parameters."""
        row = {
            "foE_mhz": self.foE_mhz,
            "h_prime_E_km": self.h_prime_E_km,
            "foF1_mhz": self.foF1_mhz,
            "h_prime_F1_km": self.h_prime_F1_km,
            "foF2_mhz": self.foF2_mhz,
            "h_prime_F2_km": self.h_prime_F2_km,
            "MUF3000_mhz": self.MUF3000_mhz,
            "M3000F2": self.M3000F2,
            "foF2_sigma_mhz": self.foF2_sigma_mhz,
            "h_prime_F2_sigma_km": self.h_prime_F2_sigma_km,
        }
        row.update({f"flag_{k}": v for k, v in self.quality_flags.items()})
        return pd.DataFrame([row])

    def summary(self) -> str:
        """One-line text summary."""
        return (
            f"ScaledParameters: "
            f"foE={self.foE_mhz:.2f} MHz  h'E={self.h_prime_E_km:.0f} km  "
            f"foF2={self.foF2_mhz:.2f} MHz  h'F2={self.h_prime_F2_km:.0f} km  "
            f"MUF(3000)={self.MUF3000_mhz:.2f} MHz  M(3000)F2={self.M3000F2:.2f}"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Bar chart of the key scaled parameters with uncertainty bars.

        Parameters
        ----------
        ax:
            Existing axes.  A new figure is created when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        params = {
            "foE": (self.foE_mhz, np.nan),
            "foF2": (self.foF2_mhz, self.foF2_sigma_mhz),
            "MUF\n(3000)": (self.MUF3000_mhz, np.nan),
        }
        labels = list(params.keys())
        vals = [v for v, _ in params.values()]
        errs = [e if not np.isnan(e) else 0.0 for _, e in params.values()]

        x = np.arange(len(labels))
        ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=5,
            color="steelblue",
            alpha=0.8,
            edgecolor="white",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title("Scaled ionospheric parameters")

        # Annotate h'F2
        if not np.isnan(self.h_prime_F2_km):
            ax.text(
                0.98,
                0.95,
                f"h'F2 = {self.h_prime_F2_km:.0f} ± "
                f"{self.h_prime_F2_sigma_km:.0f} km",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"),
            )
        return ax


# ===========================================================================
# Processor class
# ===========================================================================


class IonogramScaler:
    """Derive standard ionospheric parameters from a filtered echo DataFrame.

    The scaler operates on O-mode echoes only.  If the ``mode_col`` column
    is absent all echoes are treated as O-mode (consistent with
    :class:`~pynasonde.vipir.analysis.spread_f.SpreadFAnalyzer`).

    Parameters
    ----------
    e_layer_height_range_km:
        Height window for E-layer detection (km).  Default ``(90, 160)``.
    f1_layer_height_range_km:
        Height window for F1-layer detection (km).  Default ``(160, 250)``.
    f2_layer_height_range_km:
        Height window for F2-layer detection (km).  Default ``(160, 800)``.
    e_freq_range_mhz:
        Frequency window for E-layer cluster selection (MHz).
        Default ``(1.0, 4.5)``.
    f1_detection_threshold_mhz:
        A local maximum in the O-mode trace between foE and foF2 is
        interpreted as foF1 only if it exceeds foE by this margin (MHz).
        Default ``0.3``.
    min_echoes_for_layer:
        Minimum O-mode echo count required before a layer is considered
        detected.  Default ``3``.
    n_bootstrap:
        Number of bootstrap resamples for uncertainty estimation.
        Default ``200``.
    mode_col:
        Name of the wave-mode column.  Default ``"mode"``.

    Examples
    --------
    >>> from pynasonde.vipir.analysis.polarization import PolarizationClassifier
    >>> from pynasonde.vipir.analysis.scaler import IonogramScaler
    >>> pol    = PolarizationClassifier().fit(df)
    >>> params = IonogramScaler().fit(pol.annotated_df)
    >>> print(params.summary())
    """

    def __init__(
        self,
        e_layer_height_range_km: tuple = (_E_H_LO, _E_H_HI),
        f1_layer_height_range_km: tuple = (_F1_H_LO, _F1_H_HI),
        f2_layer_height_range_km: tuple = (_F2_H_LO, _F2_H_HI),
        e_freq_range_mhz: tuple = (_E_F_LO, _E_F_HI),
        f1_detection_threshold_mhz: float = 0.3,
        min_echoes_for_layer: int = 3,
        n_bootstrap: int = _N_BOOTSTRAP,
        mode_col: str = "mode",
    ) -> None:
        self.e_height_range = e_layer_height_range_km
        self.f1_height_range = f1_layer_height_range_km
        self.f2_height_range = f2_layer_height_range_km
        self.e_freq_range = e_freq_range_mhz
        self.f1_threshold_mhz = f1_detection_threshold_mhz
        self.min_echoes = min_echoes_for_layer
        self.n_bootstrap = n_bootstrap
        self.mode_col = mode_col

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _o_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return O-mode (or all, when mode column absent) echoes."""
        if self.mode_col in df.columns:
            return df[df[self.mode_col] == "O"].copy()
        return df.copy()

    def _e_layer_echoes(self, o_df: pd.DataFrame) -> pd.DataFrame:
        h_lo, h_hi = self.e_height_range
        f_lo, f_hi = self.e_freq_range
        mask = o_df["height_km"].between(h_lo, h_hi) & (
            o_df["frequency_khz"] / 1e3
        ).between(f_lo, f_hi)
        return o_df[mask]

    def _f2_layer_echoes(self, o_df: pd.DataFrame) -> pd.DataFrame:
        h_lo, h_hi = self.f2_height_range
        return o_df[o_df["height_km"].between(h_lo, h_hi)]

    def _f1_layer_echoes(self, o_df: pd.DataFrame) -> pd.DataFrame:
        h_lo, h_hi = self.f1_height_range
        return o_df[o_df["height_km"].between(h_lo, h_hi)]

    @staticmethod
    def _fo_from_echoes(echoes: pd.DataFrame) -> float:
        """Critical frequency = max frequency in echo cluster (MHz)."""
        if echoes.empty:
            return np.nan
        return float(echoes["frequency_khz"].max()) / 1e3

    @staticmethod
    def _h_prime_from_echoes(echoes: pd.DataFrame) -> float:
        """Minimum virtual height in echo cluster (km)."""
        if echoes.empty:
            return np.nan
        return float(echoes["height_km"].min())

    def _muf3000(self, fo_f2: float, h_prime_f2: float) -> float:
        """MUF(3000) = foF2 × √(1 + (D_half/h'F2)²)."""
        if np.isnan(fo_f2) or np.isnan(h_prime_f2) or h_prime_f2 <= 0:
            return np.nan
        return fo_f2 * np.sqrt(1.0 + (_D_HALF_KM / h_prime_f2) ** 2)

    def _bootstrap_f2(self, f2_echoes: pd.DataFrame) -> tuple[float, float]:
        """Bootstrap σ estimates for foF2 and h'F2.

        Returns
        -------
        (sigma_foF2_mhz, sigma_h_prime_F2_km)
        """
        if len(f2_echoes) < self.min_echoes:
            return np.nan, np.nan

        rng = np.random.default_rng(_RNG_SEED)
        fo_samples = np.empty(self.n_bootstrap)
        hp_samples = np.empty(self.n_bootstrap)
        n = len(f2_echoes)

        for i in range(self.n_bootstrap):
            sample = f2_echoes.sample(
                n=n, replace=True, random_state=int(rng.integers(1e9))
            )
            fo_samples[i] = self._fo_from_echoes(sample)
            hp_samples[i] = self._h_prime_from_echoes(sample)

        return float(np.std(fo_samples)), float(np.std(hp_samples))

    def _detect_f1(
        self,
        o_df: pd.DataFrame,
        fo_e: float,
        fo_f2: float,
    ) -> tuple[float, float]:
        """Detect an F1 layer as a local maximum between foE and foF2.

        A frequency step is a candidate for foF1 when:
        1. It lies between ``foE + f1_threshold`` and ``foF2 − f1_threshold``.
        2. The minimum virtual height shows a local minimum (cusping)
           relative to adjacent frequency steps.

        Returns
        -------
        (foF1_mhz, h_prime_F1_km) — both ``NaN`` when not detected.
        """
        if np.isnan(fo_e) or np.isnan(fo_f2):
            return np.nan, np.nan

        f_lo = (fo_e + self.f1_threshold_mhz) * 1e3  # kHz
        f_hi = (fo_f2 - self.f1_threshold_mhz) * 1e3  # kHz
        if f_lo >= f_hi:
            return np.nan, np.nan

        f1_echoes = self._f1_layer_echoes(o_df)
        window = f1_echoes[f1_echoes["frequency_khz"].between(f_lo, f_hi)]
        if len(window) < self.min_echoes:
            return np.nan, np.nan

        # Group by frequency step; find the step with the local h' minimum
        grp = (
            window.groupby("frequency_khz")["height_km"]
            .min()
            .reset_index()
            .sort_values("frequency_khz")
        )
        if len(grp) < 3:
            return np.nan, np.nan

        # Cusping: find index where h' has a local minimum
        h_vals = grp["height_km"].values
        cusp_idx = np.argmin(np.gradient(h_vals))  # steepest descent → F1 ledge
        fo_f1 = float(grp["frequency_khz"].iloc[cusp_idx]) / 1e3
        hp_f1 = float(grp["height_km"].iloc[cusp_idx])
        return fo_f1, hp_f1

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> ScaledParameters:
        """Scale ionospheric parameters from an echo DataFrame.

        Parameters
        ----------
        df:
            Echo DataFrame — must contain ``frequency_khz`` and
            ``height_km``.  Optionally contains a ``mode`` column
            (from :class:`~pynasonde.vipir.analysis.polarization.PolarizationClassifier`).

        Returns
        -------
        ScaledParameters
        """
        for col in ("frequency_khz", "height_km"):
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in DataFrame.")

        o_df = self._o_mode(df)

        # ── E layer ────────────────────────────────────────────────────
        e_echoes = self._e_layer_echoes(o_df)
        fo_e = self._fo_from_echoes(e_echoes)
        hp_e = self._h_prime_from_echoes(e_echoes)
        e_detected = len(e_echoes) >= self.min_echoes

        # ── F2 layer ───────────────────────────────────────────────────
        f2_echoes = self._f2_layer_echoes(o_df)
        fo_f2 = self._fo_from_echoes(f2_echoes)
        hp_f2 = self._h_prime_from_echoes(f2_echoes)
        f2_detected = len(f2_echoes) >= self.min_echoes

        fo_f2_sigma, hp_f2_sigma = self._bootstrap_f2(f2_echoes)

        # ── F1 layer (optional) ────────────────────────────────────────
        fo_f1, hp_f1 = self._detect_f1(o_df, fo_e, fo_f2)
        f1_detected = not (np.isnan(fo_f1) or np.isnan(hp_f1))

        # ── MUF / M-factor ─────────────────────────────────────────────
        muf3000 = self._muf3000(fo_f2, hp_f2)
        m3000 = (muf3000 / fo_f2) if (not np.isnan(fo_f2) and fo_f2 > 0) else np.nan

        quality_flags = {
            "E_detected": e_detected,
            "F1_detected": f1_detected,
            "F2_detected": f2_detected,
            "foF2_reliable": len(f2_echoes) >= 5,
        }

        logger.info(
            f"IonogramScaler: foE={fo_e:.2f} MHz  foF2={fo_f2:.2f} MHz  "
            f"h'F2={hp_f2:.0f} km  MUF(3000)={muf3000:.2f} MHz  "
            f"M(3000)F2={m3000:.2f}"
        )

        return ScaledParameters(
            foE_mhz=fo_e,
            h_prime_E_km=hp_e,
            foF1_mhz=fo_f1,
            h_prime_F1_km=hp_f1,
            foF2_mhz=fo_f2,
            h_prime_F2_km=hp_f2,
            MUF3000_mhz=muf3000,
            M3000F2=m3000,
            foF2_sigma_mhz=fo_f2_sigma,
            h_prime_F2_sigma_km=hp_f2_sigma,
            quality_flags=quality_flags,
        )
