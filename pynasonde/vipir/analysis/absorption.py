"""absorption.py — HF radio absorption estimation from VIPIR ionosonde echoes.

Theory
------
An HF radio wave traversing the ionosphere is attenuated whenever free
electrons oscillate in the presence of collisions with neutral molecules and
ions.  The physics is captured by the **complex** refractive index under the
collisional Appleton-Hartree formula.

**Complex refractive index with collisions**

Introduce the dimensionless parameters

    X = fp² / f²          (electron plasma frequency ratio)
    Z = ν / (2π f)        (collision frequency ratio)
    Y_L = fH cos α / f   (longitudinal gyrofrequency ratio)
    Y_T = fH sin α / f   (transverse gyrofrequency ratio)

where fp = (Ne²/ε₀m)^½/(2π) is the plasma frequency, fH = eB/(2πm) is the
electron gyrofrequency, ν is the electron-neutral/ion collision frequency, and
α is the angle between the wave normal and the geomagnetic field B.  The
Appleton-Hartree formula then gives the complex refractive index for O (+) and
X (−) modes:

    n²_O,X = 1 −   2X(1 − X − jZ)
                  ─────────────────────────────────────────────
                  2(1−X−jZ) − Y_T² ± √(Y_T⁴ + 4Y_L²(1−X−jZ)²)

Writing n = n_r + j·n_i, the spatial power absorption coefficient is

    κ(z) = −2(ω/c)·n_i   [Np/km]   →   κ_dB(z) = 8.686·κ  [dB/km]

**No-field, weak-collision limit**

For Z ≪ 1 and Y = 0 (no magnetic field):

    n² ≈ (1 − X + Z²) / (1 + Z²) − j·XZ / (1 + Z²)

so that

    n_i ≈ −XZ / (2√(1−X))          (negative → attenuating)

    κ_dB(z) ≈ 8.686 · (ω/c) · XZ / (2√(1−X))
             = 4.343 · ν · (fp²/f²) / (c_km · √(1 − fp²/f²))   [dB/km]

For f ≫ fp (non-deviative regime), √(1−X) ≈ 1 and absorption ∝ N·ν/f².  Near
reflection where X → 1, √(1−X) → 0 and the integrand diverges — this is the
**deviative absorption** contribution.

**One-way absorption (dB)**

The total one-way absorption from the ground to the reflection height h_r(f) is

    L(f) = ∫₀^{h_r} κ_dB(z) dz   [dB one-way]    (round-trip × 2)

In the no-field limit this becomes the classical formula of Davies (1990, §5):

    L_nd(f) ≈ 4.343 ∫ N(z)·ν(z) / [c·f²·√(1 − fp²(z)/f²)] dz

**Differential O/X absorption**

Since both modes travel the same path, their common free-space loss, antenna
gain, and reflection coefficient cancel in the SNR difference:

    ΔL(f) = SNR_O(f) − SNR_X(f)   [dB]

The sign of ΔL encodes the polarisation dependence of Im[n_O − n_X], which
is proportional to Z·Y_L/(1−X) at leading order and scales with the
geomagnetic dip and fH/f.  ΔL > 0 implies stronger X-mode absorption
(higher-frequency cutoff of X mode below O mode's), the normal D-layer sense.

**Lowest observed frequency (LOF) / A3 index**

The minimum frequency fmin at which any echo is detected is set by the
condition that the round-trip absorption equals the system's echo-detection
threshold.  Following Davies (1990, §9) and McNamara (1991), an absorption
index that removes the normal diurnal foE variation is

    A = fmin² − f_ref²   [MHz²]

where f_ref is a quiet-day reference (default 1 MHz).  Larger A means more
absorption.

References
----------
Davies, K. (1990). *Ionospheric Radio*. Peter Peregrinus, London.
  Chapters 5 (propagation), 9 (absorption measurements).

McNamara, L. F. (1991). *The Ionosphere: Communications, Surveillance, and
Direction Finding*. Krieger, Malabar, Florida.  Chapter 6.

Rawer, K. (1993). *Wave Propagation in the Ionosphere*. Kluwer Academic.

Budden, K. G. (1985). *The Propagation of Radio Waves*. Cambridge University
Press.  Chapter 12 (collisional absorption).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_C_KM: float = 299_792.458  # speed of light (km/s)
_DB_PER_NP: float = 8.686  # dB per neper (20/ln10)
_TWO_PI: float = 2.0 * np.pi


# ===========================================================================
# Result dataclasses
# ===========================================================================


@dataclass
class LOFResult:
    """Output of :meth:`AbsorptionAnalyzer.lof_absorption`.

    Parameters
    ----------
    fmin_mhz:
        Lowest observed frequency (MHz).  This is the minimum sounding
        frequency at which any echo was detected above the SNR threshold.
        Higher fmin indicates stronger D-layer absorption.
    lof_index_mhz2:
        LOF absorption index (MHz²): ``fmin² − f_ref²``.  Positive values
        indicate excess absorption above the quiet-day reference ``f_ref``.
    f_ref_mhz:
        Reference frequency used to compute the index (MHz).
    n_echoes:
        Total number of echoes in the DataFrame from which fmin was derived.
    """

    fmin_mhz: float = np.nan
    lof_index_mhz2: float = np.nan
    f_ref_mhz: float = 1.0
    n_echoes: int = 0

    def summary(self) -> str:
        return (
            f"LOFResult: fmin={self.fmin_mhz:.3f} MHz  "
            f"A={self.lof_index_mhz2:.3f} MHz²  "
            f"(f_ref={self.f_ref_mhz:.2f} MHz)"
        )


@dataclass
class DifferentialResult:
    """Output of :meth:`AbsorptionAnalyzer.differential_absorption`.

    Parameters
    ----------
    profile_df:
        Per-frequency-bin DataFrame with columns::

            frequency_mhz  snr_o_db  snr_x_db  delta_snr_db  n_o  n_x

        where ``delta_snr_db = SNR_O − SNR_X``.  Empty when mode labels
        are unavailable or no co-frequency O/X pairs are found.
    mean_delta_db:
        Mean ΔL = SNR_O − SNR_X (dB) averaged over all frequency bins.
        Positive values are the normal D-layer sense (O absorbed more than
        X, consistent with fH·cosα > 0 in the Northern Hemisphere).
        ``NaN`` when no paired bins are available.
    n_echoes_o:
        Number of O-mode echoes used.
    n_echoes_x:
        Number of X-mode echoes used.
    """

    profile_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "frequency_mhz",
                "snr_o_db",
                "snr_x_db",
                "delta_snr_db",
                "n_o",
                "n_x",
            ]
        )
    )
    mean_delta_db: float = np.nan
    n_echoes_o: int = 0
    n_echoes_x: int = 0

    def summary(self) -> str:
        return (
            f"DifferentialResult: {len(self.profile_df)} freq bins  "
            f"mean ΔL={self.mean_delta_db:.2f} dB  "
            f"n_O={self.n_echoes_o}  n_X={self.n_echoes_x}"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot ΔL(f) = SNR_O − SNR_X vs frequency."""
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        df = self.profile_df
        if df.empty:
            ax.text(
                0.5,
                0.5,
                "No paired O/X data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return ax
        ax.plot(
            df["frequency_mhz"],
            df["delta_snr_db"],
            "o-",
            color="tab:blue",
            ms=4,
            lw=1.4,
        )
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Frequency  (MHz)")
        ax.set_ylabel("ΔL = SNR_O − SNR_X  (dB)")
        ax.set_title("Differential O/X HF Absorption")
        if not np.isnan(self.mean_delta_db):
            ax.axhline(
                self.mean_delta_db,
                color="tab:red",
                lw=1,
                ls=":",
                label=f"mean = {self.mean_delta_db:.1f} dB",
            )
            ax.legend(fontsize=8)
        return ax


@dataclass
class TotalAbsorptionResult:
    """Output of :meth:`AbsorptionAnalyzer.total_absorption`.

    Parameters
    ----------
    profile_df:
        Per-frequency DataFrame with columns::

            frequency_mhz  virtual_height_km  fsl_db  absorption_db

        where ``absorption_db`` is the calibrated one-way absorption (dB)
        derived from the radar equation.
    tx_eirp_dbw:
        Transmit EIRP (dBW) used in the calculation.
    rx_gain_dbi:
        Receive antenna gain (dBi) used.
    reflection_coeff_db:
        Ionospheric reflection coefficient (dB) assumed.
    """

    profile_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "frequency_mhz",
                "virtual_height_km",
                "fsl_db",
                "absorption_db",
            ]
        )
    )
    tx_eirp_dbw: float = np.nan
    rx_gain_dbi: float = 0.0
    reflection_coeff_db: float = 0.0

    def summary(self) -> str:
        if self.profile_df.empty:
            return "TotalAbsorptionResult: empty (no calibration data)"
        l_mean = float(self.profile_df["absorption_db"].mean())
        return (
            f"TotalAbsorptionResult: {len(self.profile_df)} freq points  "
            f"mean L={l_mean:.1f} dB  "
            f"EIRP={self.tx_eirp_dbw:.1f} dBW"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot absolute one-way absorption L(f) vs frequency."""
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        df = self.profile_df
        if df.empty:
            ax.text(
                0.5,
                0.5,
                "No calibrated data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return ax
        ax.plot(
            df["frequency_mhz"],
            df["absorption_db"],
            "s-",
            color="tab:orange",
            ms=4,
            lw=1.4,
        )
        ax.set_xlabel("Frequency  (MHz)")
        ax.set_ylabel("One-way absorption  L(f)  (dB)")
        ax.set_title("Absolute HF Absorption")
        return ax


@dataclass
class AbsorptionProfileResult:
    """Output of :meth:`AbsorptionAnalyzer.absorption_profile`.

    Parameters
    ----------
    profile_df:
        Height-resolved DataFrame with columns::

            height_km  nu_hz  fp_mhz  X  kappa_dB_per_km

        ``kappa_dB_per_km`` is the local one-way power absorption rate
        (dB/km) computed from the no-field Appleton-Hartree formula with the
        user-supplied collision-frequency profile.
    cumulative_df:
        Cumulative one-way absorption from the ground (dB) vs height::

            height_km  L_oneway_db

    total_absorption_db:
        Total one-way absorption to the top of the provided EDP (dB).
    """

    profile_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "height_km",
                "nu_hz",
                "fp_mhz",
                "X",
                "kappa_dB_per_km",
            ]
        )
    )
    cumulative_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "height_km",
                "L_oneway_db",
            ]
        )
    )
    total_absorption_db: float = np.nan

    def summary(self) -> str:
        return (
            f"AbsorptionProfileResult: {len(self.profile_df)} height levels  "
            f"total L={self.total_absorption_db:.2f} dB (one-way)"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot κ(z) and cumulative L(z) vs height."""
        if ax is None:
            _, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        else:
            axes = [ax, ax]

        df = self.profile_df
        cdf = self.cumulative_df

        ax0 = axes[0]
        if not df.empty:
            ax0.plot(df["kappa_dB_per_km"], df["height_km"], color="tab:blue", lw=1.4)
        ax0.set_xlabel("κ  (dB km⁻¹)")
        ax0.set_ylabel("Height  (km)")
        ax0.set_title("Local absorption rate")

        ax1 = axes[1]
        if not cdf.empty:
            ax1.plot(cdf["L_oneway_db"], cdf["height_km"], color="tab:orange", lw=1.4)
        ax1.set_xlabel("Cumulative L  (dB, one-way)")
        ax1.set_title("Cumulative absorption")

        plt.tight_layout()
        return axes[0]


# ===========================================================================
# Processor class
# ===========================================================================


class AbsorptionAnalyzer:
    """HF radio absorption estimation from VIPIR ionosonde echo DataFrames.

    Four independent estimators are exposed as separate methods so that callers
    can use whichever combination their data support:

    * :meth:`lof_absorption` — Lowest Observed Frequency (LOF) index.
      No calibration required.  Works on any echo DataFrame.

    * :meth:`differential_absorption` — Per-frequency O minus X SNR
      difference ΔL(f).  No calibration required.  Needs mode-labelled data
      (output of :class:`~pynasonde.vipir.analysis.polarization.PolarizationClassifier`).

    * :meth:`total_absorption` — Calibrated one-way absorption L(f) from the
      radar equation.  Requires transmit EIRP and receive gain.

    * :meth:`absorption_profile` — Height-resolved absorption rate κ(z) and
      cumulative L(z) from the no-field Appleton-Hartree formula.  Requires an
      :class:`~pynasonde.vipir.analysis.inversion.EDPResult` (electron density
      profile) and a user-supplied collision-frequency profile ν(z).

    Parameters
    ----------
    freq_col:
        Frequency column name in the echo DataFrame.  Default
        ``"frequency_khz"`` (kHz; auto-detected if median > 100).
    height_col:
        Virtual-height column name (km).  Default ``"height_km"``.
    snr_col:
        SNR column name (dB).  Default ``"snr_db"``.
    mode_col:
        Magnetoionic mode column name (``"O"`` / ``"X"``).
        Default ``"mode"``.
    freq_bin_mhz:
        Frequency bin width (MHz) for grouping echoes in
        :meth:`differential_absorption` and :meth:`total_absorption`.
        Default ``0.1``.
    f_ref_mhz:
        Quiet-day reference frequency for the LOF absorption index
        ``A = fmin² − f_ref²``.  Default ``1.0``.
    """

    def __init__(
        self,
        freq_col: str = "frequency_khz",
        height_col: str = "height_km",
        snr_col: str = "snr_db",
        mode_col: str = "mode",
        freq_bin_mhz: float = 0.1,
        f_ref_mhz: float = 1.0,
    ) -> None:
        self.freq_col = freq_col
        self.height_col = height_col
        self.snr_col = snr_col
        self.mode_col = mode_col
        self.freq_bin_mhz = freq_bin_mhz
        self.f_ref_mhz = f_ref_mhz

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_mhz(self, df: pd.DataFrame) -> pd.Series:
        """Return frequency in MHz, auto-detecting kHz vs MHz input."""
        col = df[self.freq_col]
        return col / 1e3 if float(col.median()) > 100 else col

    def _bin_column(self, freq_mhz: pd.Series) -> pd.Series:
        """Round frequencies to the nearest freq_bin_mhz centre."""
        return ((freq_mhz / self.freq_bin_mhz).round() * self.freq_bin_mhz).round(4)

    @staticmethod
    def _fsl_db(height_km: float, freq_mhz: float) -> float:
        """Two-way free-space path loss (dB).

        FSL = 20·log10(4π · 2h' · f / c)

        Parameters
        ----------
        height_km : virtual height h' in km.
        freq_mhz  : sounding frequency in MHz.
        """
        path_km = 2.0 * height_km  # round-trip
        freq_khz = freq_mhz * 1e3
        return float(20.0 * np.log10(4.0 * np.pi * path_km * freq_khz / _C_KM))

    # ------------------------------------------------------------------
    # Method 1 — Lowest Observed Frequency
    # ------------------------------------------------------------------

    def lof_absorption(self, df: pd.DataFrame) -> LOFResult:
        """Compute the LOF-based absorption index.

        The lowest observed frequency ``fmin`` is the minimum sounding
        frequency at which any echo survives the SNR filter.  The absorption
        index is ``A = fmin² − f_ref²`` (MHz²).  Larger A values indicate
        stronger D-layer absorption.

        Parameters
        ----------
        df:
            Filtered echo DataFrame (any subset of echoes is acceptable;
            the method simply finds the minimum frequency present).

        Returns
        -------
        LOFResult
        """
        if df.empty:
            logger.warning("lof_absorption: empty DataFrame.")
            return LOFResult(f_ref_mhz=self.f_ref_mhz, n_echoes=0)

        if self.freq_col not in df.columns:
            raise KeyError(f"Frequency column '{self.freq_col}' not found.")

        f_mhz = self._to_mhz(df)
        fmin = float(f_mhz.min())
        index = fmin**2 - self.f_ref_mhz**2

        result = LOFResult(
            fmin_mhz=fmin,
            lof_index_mhz2=index,
            f_ref_mhz=self.f_ref_mhz,
            n_echoes=len(df),
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Method 2 — Differential O/X absorption
    # ------------------------------------------------------------------

    def differential_absorption(self, df: pd.DataFrame) -> DifferentialResult:
        """Compute the per-frequency differential O/X SNR profile ΔL(f).

        For each frequency bin the method computes the median SNR of O-mode
        echoes (``SNR_O``) and X-mode echoes (``SNR_X``) independently, then
        forms::

            ΔL(f) = SNR_O(f) − SNR_X(f)   [dB]

        Because both modes travel the same ray path, free-space loss, antenna
        gain, and reflection coefficient cancel exactly.  The residual ΔL
        encodes only the differential magneto-ionic absorption.

        Parameters
        ----------
        df:
            Echo DataFrame with a ``mode`` column (``"O"`` / ``"X"``) and an
            ``snr_db`` column.  Typically the output DataFrame from
            :class:`~pynasonde.vipir.analysis.polarization.PolarizationClassifier`.

        Returns
        -------
        DifferentialResult
        """
        _empty = DifferentialResult()

        if df.empty:
            logger.warning("differential_absorption: empty DataFrame.")
            return _empty

        for col in (self.freq_col, self.snr_col, self.mode_col):
            if col not in df.columns:
                logger.warning(
                    f"differential_absorption: column '{col}' not found — "
                    "returning empty result."
                )
                return _empty

        work = df.copy()
        work["_freq_mhz"] = self._to_mhz(df).values
        work["_bin"] = self._bin_column(work["_freq_mhz"]).values

        o_df = work[work[self.mode_col] == "O"]
        x_df = work[work[self.mode_col] == "X"]
        n_o = len(o_df)
        n_x = len(x_df)

        if o_df.empty or x_df.empty:
            logger.debug(
                f"differential_absorption: O echoes={n_o}  X echoes={n_x}  "
                "— need both modes."
            )
            return DifferentialResult(n_echoes_o=n_o, n_echoes_x=n_x)

        o_grp = o_df.groupby("_bin").agg(
            snr_o_db=(self.snr_col, "median"),
            n_o=(self.snr_col, "count"),
            freq_mean=("_freq_mhz", "mean"),
        )
        x_grp = x_df.groupby("_bin").agg(
            snr_x_db=(self.snr_col, "median"),
            n_x=(self.snr_col, "count"),
        )

        merged = o_grp.join(x_grp, how="inner").reset_index(drop=True)
        if merged.empty:
            logger.debug("differential_absorption: no co-frequency O/X bins.")
            return DifferentialResult(n_echoes_o=n_o, n_echoes_x=n_x)

        merged["delta_snr_db"] = merged["snr_o_db"] - merged["snr_x_db"]
        profile = (
            merged[["freq_mean", "snr_o_db", "snr_x_db", "delta_snr_db", "n_o", "n_x"]]
            .rename(columns={"freq_mean": "frequency_mhz"})
            .sort_values("frequency_mhz")
            .reset_index(drop=True)
        )

        mean_delta = float(profile["delta_snr_db"].mean())
        result = DifferentialResult(
            profile_df=profile,
            mean_delta_db=mean_delta,
            n_echoes_o=n_o,
            n_echoes_x=n_x,
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Method 3 — Total calibrated absorption
    # ------------------------------------------------------------------

    def total_absorption(
        self,
        df: pd.DataFrame,
        tx_eirp_dbw: float,
        rx_gain_dbi: float = 0.0,
        reflection_coeff_db: float = 0.0,
    ) -> TotalAbsorptionResult:
        """Compute calibrated one-way absorption L(f) from the radar equation.

        Uses the ionosonde radar equation::

            L(f) = EIRP + G_rx − FSL(h', f) − R_coeff − SNR(f)

        where FSL = 20·log10(4π·2h'·f/c) is the two-way free-space loss.
        The O-mode median SNR in each frequency bin is used (falls back to all
        echoes when mode labels are absent).

        Parameters
        ----------
        df:
            Filtered echo DataFrame with ``frequency_khz``, ``height_km``,
            and ``snr_db`` columns.
        tx_eirp_dbw:
            Transmit EIRP in dBW (= transmit power dBW + antenna gain dBi).
            Obtain from the VIPIR station hardware specification.
        rx_gain_dbi:
            Receive antenna gain (dBi).  Default ``0.0``.
        reflection_coeff_db:
            Ionospheric reflection coefficient (dB).  For vertical O-mode
            incidence this is approximately 0 dB.  Default ``0.0``.

        Returns
        -------
        TotalAbsorptionResult
        """
        _empty = TotalAbsorptionResult(
            tx_eirp_dbw=tx_eirp_dbw,
            rx_gain_dbi=rx_gain_dbi,
            reflection_coeff_db=reflection_coeff_db,
        )

        if df.empty:
            logger.warning("total_absorption: empty DataFrame.")
            return _empty

        for col in (self.freq_col, self.height_col, self.snr_col):
            if col not in df.columns:
                logger.warning(f"total_absorption: column '{col}' not found.")
                return _empty

        work = df.copy()
        work["_freq_mhz"] = self._to_mhz(df).values
        work["_bin"] = self._bin_column(work["_freq_mhz"]).values

        # Prefer O-mode echoes for the SNR reference
        if self.mode_col in work.columns:
            sub = work[work[self.mode_col] == "O"]
            if sub.empty:
                sub = work
        else:
            sub = work

        grp = (
            sub.groupby("_bin")
            .agg(
                frequency_mhz=("_freq_mhz", "mean"),
                virtual_height_km=(self.height_col, "median"),
                snr_db=(self.snr_col, "median"),
            )
            .reset_index(drop=True)
            .sort_values("frequency_mhz")
        )

        rows = []
        for _, row in grp.iterrows():
            f = float(row["frequency_mhz"])
            h = float(row["virtual_height_km"])
            snr = float(row["snr_db"])
            if not np.isfinite(f) or not np.isfinite(h) or h <= 0:
                continue
            fsl = self._fsl_db(h, f)
            # radar equation (all in dB / dBW / dBi)
            L = tx_eirp_dbw + rx_gain_dbi - fsl - reflection_coeff_db - snr
            rows.append(
                {
                    "frequency_mhz": f,
                    "virtual_height_km": h,
                    "fsl_db": fsl,
                    "absorption_db": L,
                }
            )

        if not rows:
            return _empty

        profile = pd.DataFrame(rows)
        result = TotalAbsorptionResult(
            profile_df=profile,
            tx_eirp_dbw=tx_eirp_dbw,
            rx_gain_dbi=rx_gain_dbi,
            reflection_coeff_db=reflection_coeff_db,
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Method 4 — Height-resolved absorption profile
    # ------------------------------------------------------------------

    def absorption_profile(
        self,
        edp,  # EDPResult
        nu_hz: Union[np.ndarray, Callable],
        heights_km: Optional[np.ndarray] = None,
        n_interp: int = 500,
        f_wave_mhz: float = 2.0,
    ) -> AbsorptionProfileResult:
        """Compute the height-resolved absorption rate κ(z) and cumulative L(z).

        Uses the no-field Appleton-Hartree formula with weak collisions:

            κ(z) = 4.343 · ν(z) · [fp²(z)/f²] / (c_km · √(1 − fp²(z)/f²))

        where f is the wave frequency ``f_wave_mhz``, fp(z) comes from the
        :class:`~pynasonde.vipir.analysis.inversion.EDPResult`, and ν(z) is the
        user-supplied collision-frequency profile.  Only heights where fp < f_wave
        (below the reflection level) contribute to the integral.

        The deviative absorption near the reflection level (where fp → f and
        the square-root denominator → 0) is handled by clamping X < 0.999 so
        that the integrand remains finite; this is consistent with the standard
        Titheridge treatment of the absorption integral.

        Parameters
        ----------
        edp:
            :class:`~pynasonde.vipir.analysis.inversion.EDPResult` from
            :class:`~pynasonde.vipir.analysis.inversion.TrueHeightInversion`.
            Provides ``true_height_km`` and ``plasma_freq_mhz``.
        nu_hz:
            Collision-frequency profile ν(z) in Hz, specified as either:

            * **callable** ``nu_hz(height_km) → float`` — function evaluated
              at each height (e.g. from a NRLMSISE-00 wrapper).
            * **np.ndarray** of shape ``(M,)`` — values corresponding to
              ``heights_km``.  ``heights_km`` must be provided in this case.
        heights_km:
            Height grid (km) for the array form of ``nu_hz``.  Ignored when
            ``nu_hz`` is a callable.  Must be the same length as ``nu_hz``.
        n_interp:
            Number of integration points used to interpolate the EDP from the
            lowest to the highest true-height layer.  Default ``500``.
        f_wave_mhz:
            Sounding wave frequency (MHz) used in the Appleton-Hartree formula.
            Heights where fp(z) ≥ f_wave_mhz are above the reflection level and
            contribute zero absorption.  Default ``2.0``.

        Returns
        -------
        AbsorptionProfileResult

        Raises
        ------
        ValueError
            If ``nu_hz`` is an array but ``heights_km`` is not provided or has
            a different length.
        """
        _empty = AbsorptionProfileResult()

        # Validate edp
        if edp is None or len(edp.true_height_km) < 2:
            logger.warning("absorption_profile: EDP has fewer than 2 layers.")
            return _empty

        # Build ν(z) callable
        if callable(nu_hz):
            nu_func = nu_hz
        else:
            nu_arr = np.asarray(nu_hz, dtype=float)
            if heights_km is None:
                raise ValueError(
                    "absorption_profile: 'heights_km' must be provided "
                    "when 'nu_hz' is an array."
                )
            h_arr = np.asarray(heights_km, dtype=float)
            if nu_arr.shape != h_arr.shape:
                raise ValueError(
                    f"absorption_profile: nu_hz ({nu_arr.shape}) and "
                    f"heights_km ({h_arr.shape}) must have the same length."
                )
            # Linear interpolation; extrapolate as boundary value
            nu_func = lambda h: float(np.interp(h, h_arr, nu_arr))

        # Build a fine height grid spanning the EDP
        h_min = float(edp.true_height_km[0])
        h_max = float(edp.true_height_km[-1])
        h_grid = np.linspace(h_min, h_max, n_interp)

        # Interpolate plasma frequency onto the fine grid
        fp_grid = np.interp(h_grid, edp.true_height_km, edp.plasma_freq_mhz)

        # Compute κ(z) at each grid point
        # f used here is the local plasma frequency (we're computing the
        # frequency-integrated profile, so κ is parameterised by height)
        kappa = np.zeros(n_interp)
        nu_grid = np.array([nu_func(h) for h in h_grid])

        for k in range(n_interp):
            fp = fp_grid[k]  # MHz — local plasma frequency
            nu = nu_grid[k]  # Hz
            # κ [dB/km] = 4.343 · ν [Hz] · X / (c [km/s] · √(1−X))
            # X = (fp/f_wave)²; heights where fp ≥ f_wave are above the
            # reflection level — no wave reaches there, so κ = 0.
            X = (fp / f_wave_mhz) ** 2
            if X >= 0.999:
                kappa[k] = 0.0
                continue
            denom = np.sqrt(max(1.0 - X, 1e-6))
            kappa[k] = 4.343 * nu * X / (_C_KM * denom)

        # Cumulative one-way absorption (trapezoidal integration)
        dh_km = np.diff(h_grid)
        kappa_db = kappa  # already in dB/km
        cumL = np.zeros(n_interp)
        for k in range(1, n_interp):
            cumL[k] = cumL[k - 1] + 0.5 * (kappa_db[k - 1] + kappa_db[k]) * dh_km[k - 1]

        X_grid = np.minimum((fp_grid / f_wave_mhz) ** 2, 1.0)
        profile_df = pd.DataFrame(
            {
                "height_km": h_grid,
                "nu_hz": nu_grid,
                "fp_mhz": fp_grid,
                "X": X_grid,
                "kappa_dB_per_km": kappa_db,
            }
        )
        cumulative_df = pd.DataFrame(
            {
                "height_km": h_grid,
                "L_oneway_db": cumL,
            }
        )

        result = AbsorptionProfileResult(
            profile_df=profile_df,
            cumulative_df=cumulative_df,
            total_absorption_db=float(cumL[-1]),
        )
        logger.info(result.summary())
        return result
