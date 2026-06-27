"""inversion.py — Virtual height → true height inversion (Abel / lamination).

An ionogram records the **virtual height** h'(f) — the height a pulse would
reach if it propagated at the speed of light c throughout.  Because the
ionospheric plasma slows the pulse (group refractive index μ' > 1 for O-mode),
the true reflection height is always *less* than the virtual height.

The relationship is the Abel integral:

    h'(f) = ∫₀^{h_ref(f)} μ'(f, z) dz

where h_ref(f) is the true reflection height (plasma frequency fₚ = f) and
μ'(f, N) = 1/√(1 − fₚ²/f²) for the O-mode without magnetic field.

Inverting this integral yields the true-height profile h(fₚ) and hence the
electron density profile N(h).

**Lamination method (Titheridge 1967, Paul 1975)**
Treat the ionosphere as N horizontal layers of uniform plasma frequency.  The
true height of the n-th layer is obtained iteratively from:

    r_n = h'(fₙ) − Σᵢ₌₀ⁿ⁻¹ (μ'(fₙ, fₚᵢ) − 1) × Δhᵢ

where Δhᵢ = rᵢ − rᵢ₋₁ is the thickness of layer i and fₚᵢ = fᵢ (the
plasma frequency at the reflection level equals the sounding frequency).

This module provides:

:class:`TrueHeightInversion`
    Processor — inverts an O-mode ``(frequency_mhz, h_virtual_km)`` trace.

:class:`EDPResult`
    Output dataclass — true-height profile, plasma frequency, electron
    density, and standard layer parameters.

References
----------
Titheridge, J. E. (1967). A new method for the analysis of ionospheric
h'(f) records. *Journal of Atmospheric and Terrestrial Physics*, 29, 763–778.

Paul, A. K. (1975). POLAN — A program for true-height analysis of
ionograms. *NOAA Technical Report ERL 324-SEL 31*.

Bilitza, D. (1990). International Reference Ionosphere 1990.
*NSSDC/WDC-A-R&S 90-22*, Greenbelt, Maryland.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Plasma frequency → electron density:
#   N (m⁻³) = fₚ (Hz)² / 80.64
#   ↔  fₚ (MHz) → N (cm⁻³) = fₚ² × 1.2399e4
_FP_MHZ_TO_N_CM3: float = 1.2399e4  # N_cm3 = fₚ_mhz² × this
_FP_MHZ_TO_N_M3: float = 1.2399e10  # N_m3  = fₚ_mhz² × this


# ===========================================================================
# Output dataclass
# ===========================================================================


@dataclass
class EDPResult:
    """Electron density profile from true-height inversion.

    Parameters
    ----------
    true_height_km:
        True reflection heights (km), one per input frequency.
    plasma_freq_mhz:
        Plasma frequency at each layer (MHz).  Equals the sounding frequency
        in the lamination approximation (fₚ ≈ f at the reflection level).
    electron_density_cm3:
        Electron density at each layer (cm⁻³).
    virtual_height_km:
        Input virtual heights (km) that were inverted.
    frequency_mhz:
        Input sounding frequencies (MHz).
    foF2_mhz:
        Critical frequency of the F2 layer — maximum plasma frequency (MHz).
    hmF2_km:
        True height of the F2 peak (km).
    NmF2_cm3:
        Peak electron density (cm⁻³).
    method:
        Inversion method used: ``"lamination"``.
    n_layers:
        Number of layers used in the inversion.
    """

    true_height_km: np.ndarray
    plasma_freq_mhz: np.ndarray
    electron_density_cm3: np.ndarray
    virtual_height_km: np.ndarray
    frequency_mhz: np.ndarray
    foF2_mhz: float
    hmF2_km: float
    NmF2_cm3: float
    method: str
    n_layers: int

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns: frequency_mhz, virtual_height_km,
        true_height_km, plasma_freq_mhz, electron_density_cm3."""
        return pd.DataFrame(
            {
                "frequency_mhz": self.frequency_mhz,
                "virtual_height_km": self.virtual_height_km,
                "true_height_km": self.true_height_km,
                "plasma_freq_mhz": self.plasma_freq_mhz,
                "electron_density_cm3": self.electron_density_cm3,
            }
        )

    def to_csv(self, path: str) -> None:
        """Write the EDP to a CSV file."""
        self.to_dataframe().to_csv(path, index=False, float_format="%.4f")
        logger.info(f"EDPResult written to {path}")

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"EDPResult ({self.method}): n_layers={self.n_layers}  "
            f"foF2={self.foF2_mhz:.2f} MHz  hmF2={self.hmF2_km:.1f} km  "
            f"NmF2={self.NmF2_cm3:.2e} cm⁻³"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot N(h) electron density profile alongside the virtual-height trace.

        Parameters
        ----------
        ax:
            Existing axes.  A new figure (two panels) is created when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
            Left panel axes (N vs h).
        """
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        else:
            axes = [ax, ax]

        # ── left: N(h) true height profile ─────────────────────────────
        ax0 = axes[0]
        ax0.plot(
            self.electron_density_cm3,
            self.true_height_km,
            "o-",
            color="tab:red",
            ms=4,
            lw=1.2,
            label="N(h)",
        )
        ax0.set_xlabel("Electron density  (cm⁻³)")
        ax0.set_ylabel("True height  (km)")
        ax0.set_title("Electron density profile")
        ax0.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
        ax0.legend(fontsize=8)

        # ── right: virtual vs true height trace ─────────────────────────
        ax1 = axes[1]
        ax1.plot(
            self.frequency_mhz,
            self.virtual_height_km,
            "s--",
            color="steelblue",
            ms=4,
            lw=1,
            label="h' (virtual)",
        )
        ax1.plot(
            self.frequency_mhz,
            self.true_height_km,
            "o-",
            color="tab:red",
            ms=4,
            lw=1.2,
            label="h (true)",
        )
        ax1.set_xlabel("Frequency  (MHz)")
        ax1.set_title("Virtual vs true height")
        ax1.legend(fontsize=8)

        if ax is None:
            plt.tight_layout()
        return axes[0]


# ===========================================================================
# Processor class
# ===========================================================================


class TrueHeightInversion:
    """Invert an O-mode ionogram trace to a true-height electron density profile.

    The lamination method (Titheridge 1967) discretises the Abel integral into
    N horizontal layers.  Starting from the bottommost layer and working upward,
    each true reflection height is computed by subtracting the accumulated group-
    path excess of all underlying layers.

    Parameters
    ----------
    method:
        Inversion algorithm.  Currently only ``"lamination"`` is implemented.
    min_freq_mhz:
        Frequencies below this value are excluded before inversion (MHz).
        Removes low-frequency noise below the E-layer.  Default ``1.0``.
    max_freq_mhz:
        Frequencies above this value are excluded (MHz).  Useful to limit
        the inversion to a specific layer.  ``None`` → no upper limit.
    monotone_enforce:
        If ``True`` (default), non-monotone true-height values (which are
        physically invalid) are removed from the output after inversion.
    freq_col:
        Name of the frequency column when fitting from a DataFrame.
        Default ``"frequency_mhz"``.
    height_col:
        Name of the virtual-height column when fitting from a DataFrame.
        Default ``"height_km"``.
    mode_col:
        Name of the mode column when filtering O-mode echoes from a full echo
        DataFrame.  Default ``"mode"``.
    bin_width_mhz:
        Frequency bin width (MHz) used to decimate the trace in
        :meth:`fit_from_df` before inversion.  Real RIQ traces have ~19 kHz
        steps which cause near-singular μ' contributions; binning to this
        width (default ``0.3``) gives 10–20 representative profile points
        consistent with the POLAN prescription.

    Examples
    --------
    Fit from a ``(freq_mhz, h_virtual_km)`` pair of arrays::

        from pynasonde.vipir.analysis.inversion import TrueHeightInversion

        inv = TrueHeightInversion()
        edp = inv.fit(freq_mhz=trace_freq, h_virtual_km=trace_h)
        print(edp.summary())

    Fit directly from an echo DataFrame with mode labels::

        edp = inv.fit_from_df(pol_result.o_mode_df())
    """

    def __init__(
        self,
        method: Literal["lamination"] = "lamination",
        min_freq_mhz: float = 1.0,
        max_freq_mhz: Optional[float] = None,
        monotone_enforce: bool = True,
        freq_col: str = "frequency_mhz",
        height_col: str = "height_km",
        mode_col: str = "mode",
        bin_width_mhz: float = 0.05,
    ) -> None:
        if method != "lamination":
            raise NotImplementedError(
                f"Method '{method}' is not yet implemented.  Use 'lamination'."
            )
        self.method = method
        self.min_freq_mhz = min_freq_mhz
        self.max_freq_mhz = max_freq_mhz
        self.monotone_enforce = monotone_enforce
        self.freq_col = freq_col
        self.height_col = height_col
        self.mode_col = mode_col
        self.bin_width_mhz = bin_width_mhz

    # ------------------------------------------------------------------
    # Core inversion
    # ------------------------------------------------------------------

    @staticmethod
    def _group_refractive_index(f_mhz: float, fp_mhz: float) -> float:
        """O-mode group refractive index μ'(f, fₚ) = 1/√(1 − (fₚ/f)²).

        Returns 1.0 when fₚ/f > 0.999 (near-singularity) to avoid overflow.
        """
        ratio = fp_mhz / f_mhz
        if ratio >= 0.999:
            return 1.0
        return 1.0 / np.sqrt(1.0 - ratio**2)

    def _lamination(
        self,
        freq_mhz: np.ndarray,
        h_virtual_km: np.ndarray,
    ) -> np.ndarray:
        """Apply the Titheridge lamination formula layer by layer.

        The Abel integral h'(f) = ∫₀^{h_r(f)} μ'(f,z) dz is discretised as:

            r_n = h'(f_n) − Σ_{j=1}^{n-1} (μ'(f_n, f_j) − 1) × (r_j − r_{j-1})

        Layer 0 is the ionosphere floor (r_{-1} = r_0, zero thickness).
        Each layer j contributes (μ'−1)×Δr_j to the group-path excess.

        IMPORTANT — h_prev must NOT be updated when a layer is skipped
        (delta_h ≤ 0).  If h_true[j] is already spuriously negative due to
        near-singular corrections from prior layers, updating h_prev to that
        value makes the next delta_h artificially large and positive, feeding
        an exponential runaway.  Instead, we keep h_prev at the last
        physically valid layer height.

        Parameters
        ----------
        freq_mhz:
            Sounding frequencies sorted in ascending order (MHz).
        h_virtual_km:
            Corresponding virtual heights (km).

        Returns
        -------
        np.ndarray
            True reflection heights (km), same length as input.
        """
        n = len(freq_mhz)
        h_true = np.empty(n)

        # Layer 0: ionosphere floor — no correction applied
        h_true[0] = h_virtual_km[0]

        for i in range(1, n):
            correction = 0.0
            # h_prev tracks the bottom of the current layer being integrated.
            # Start from r_0 (the floor); layers with non-positive thickness
            # are skipped WITHOUT updating h_prev so that spurious negative
            # h_true values from prior diverging layers do not contaminate the
            # reference height for subsequent valid layers.
            h_prev = h_true[0]
            for j in range(1, i):  # j=0 always has zero thickness
                delta_h = h_true[j] - h_prev
                if delta_h <= 0.0:
                    continue  # skip — do NOT move h_prev
                mu_prime = self._group_refractive_index(freq_mhz[i], freq_mhz[j])
                correction += (mu_prime - 1.0) * delta_h
                h_prev = h_true[j]  # advance only on valid layers

            h_true[i] = h_virtual_km[i] - correction

        return h_true

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        freq_mhz: np.ndarray,
        h_virtual_km: np.ndarray,
    ) -> EDPResult:
        """Invert a ``(frequency, virtual height)`` trace.

        Parameters
        ----------
        freq_mhz:
            Sounding frequencies (MHz).  Need not be sorted — the method
            sorts them internally.
        h_virtual_km:
            Corresponding virtual heights (km).

        Returns
        -------
        EDPResult

        Raises
        ------
        ValueError
            If fewer than 2 valid data points remain after filtering.
        """
        freq_mhz = np.asarray(freq_mhz, dtype=float)
        h_virtual = np.asarray(h_virtual_km, dtype=float)

        if freq_mhz.shape != h_virtual.shape:
            raise ValueError(
                f"freq_mhz and h_virtual_km must have the same shape; "
                f"got {freq_mhz.shape} and {h_virtual.shape}."
            )

        # Remove NaNs and apply frequency limits
        valid = np.isfinite(freq_mhz) & np.isfinite(h_virtual)
        valid &= freq_mhz >= self.min_freq_mhz
        if self.max_freq_mhz is not None:
            valid &= freq_mhz <= self.max_freq_mhz

        freq_mhz = freq_mhz[valid]
        h_virtual = h_virtual[valid]

        if len(freq_mhz) < 2:
            raise ValueError(
                "Need at least 2 valid (frequency, height) points for inversion."
            )

        # Sort by frequency ascending
        order = np.argsort(freq_mhz)
        freq_mhz = freq_mhz[order]
        h_virtual = h_virtual[order]

        logger.info(
            f"TrueHeightInversion ({self.method}): "
            f"inverting {len(freq_mhz)} layers  "
            f"f=[{freq_mhz[0]:.2f}, {freq_mhz[-1]:.2f}] MHz  "
            f"h'=[{h_virtual[0]:.1f}, {h_virtual[-1]:.1f}] km"
        )

        h_true = self._lamination(freq_mhz, h_virtual)

        # Enforce monotonicity: remove layers where h_true decreases
        if self.monotone_enforce:
            keep = np.ones(len(h_true), dtype=bool)
            prev = h_true[0]
            for k in range(1, len(h_true)):
                if h_true[k] <= prev:
                    keep[k] = False
                    logger.debug(
                        f"  layer {k}: h_true={h_true[k]:.1f} km not monotone — removed"
                    )
                else:
                    prev = h_true[k]
            freq_mhz = freq_mhz[keep]
            h_virtual = h_virtual[keep]
            h_true = h_true[keep]

        # Plasma frequency (= sounding frequency at reflection level)
        fp_mhz = freq_mhz.copy()

        # Electron density
        n_cm3 = fp_mhz**2 * _FP_MHZ_TO_N_CM3

        # Layer parameters
        fo_f2 = float(fp_mhz[-1])
        hm_f2 = float(h_true[-1])
        nm_f2 = float(n_cm3[-1])

        logger.info(
            f"TrueHeightInversion complete: foF2={fo_f2:.2f} MHz  "
            f"hmF2={hm_f2:.1f} km  NmF2={nm_f2:.2e} cm⁻³"
        )

        return EDPResult(
            true_height_km=h_true,
            plasma_freq_mhz=fp_mhz,
            electron_density_cm3=n_cm3,
            virtual_height_km=h_virtual,
            frequency_mhz=freq_mhz,
            foF2_mhz=fo_f2,
            hmF2_km=hm_f2,
            NmF2_cm3=nm_f2,
            method=self.method,
            n_layers=len(h_true),
        )

    def fit_from_df(self, df: pd.DataFrame) -> EDPResult:
        """Convenience wrapper — fit from an echo or trace DataFrame.

        The method extracts ``(frequency_mhz, height_km)`` from *df*, using
        the median virtual height at each unique frequency step as the trace.
        O-mode filtering is applied when a ``mode`` column is present.

        Parameters
        ----------
        df:
            Echo DataFrame (with ``frequency_khz`` and ``height_km`` columns)
            or a pre-scaled trace DataFrame (with ``frequency_mhz`` and
            ``height_km`` columns).

        Returns
        -------
        EDPResult
        """
        # Accept either frequency_khz (RIQ echo DF) or frequency_mhz (NGI trace)
        if "frequency_khz" in df.columns:
            df = df.copy()
            df["frequency_mhz"] = df["frequency_khz"] / 1e3
        elif self.freq_col not in df.columns:
            raise KeyError(
                f"Column '{self.freq_col}' not found.  "
                "Pass a DataFrame with 'frequency_khz' or 'frequency_mhz'."
            )

        # O-mode filter
        if self.mode_col in df.columns:
            df = df[df[self.mode_col] == "O"]

        if df.empty:
            raise ValueError("No O-mode echoes found after filtering.")

        # Build trace: median height per frequency step
        trace = (
            df.groupby("frequency_mhz")[self.height_col]
            .median()
            .reset_index()
            .sort_values("frequency_mhz")
        )

        # Decimate to ~0.3 MHz bins for lamination stability.
        # The Titheridge lamination formula is conditionally stable only when
        # adjacent frequency steps are well-separated (≥ 0.1 MHz).  Real RIQ
        # traces can have ~227 steps at ~19 kHz spacing; the near-singular
        # μ'(fᵢ, fⱼ) terms for i ≈ j accumulate and cause exponential
        # divergence.  Binning to ≈ 0.3 MHz gives 10–20 representative
        # profile values — consistent with the classic POLAN prescription.
        bin_width_mhz = self.bin_width_mhz
        trace["_bin"] = (
            (trace["frequency_mhz"] / bin_width_mhz).round() * bin_width_mhz
        ).round(4)
        trace = (
            trace.groupby("_bin")
            .agg(
                frequency_mhz=("frequency_mhz", "mean"),
                **{self.height_col: (self.height_col, "median")},
            )
            .reset_index(drop=True)
            .sort_values("frequency_mhz")
        )

        logger.debug(
            f"fit_from_df: decimated trace to {len(trace)} points "
            f"(bin_width={bin_width_mhz} MHz)"
        )

        # Enforce monotone virtual heights before inversion.
        # h'(f) should be non-decreasing; real traces can have a small
        # E-F valley dip.  We keep only the monotone-increasing envelope
        # so that the lamination corrections stay well-conditioned.
        h_arr = trace[self.height_col].to_numpy()
        f_arr = trace["frequency_mhz"].to_numpy()
        mono_mask = np.ones(len(h_arr), dtype=bool)
        h_run = h_arr[0]
        for k in range(1, len(h_arr)):
            if h_arr[k] >= h_run:
                h_run = h_arr[k]
            else:
                mono_mask[k] = False
        n_dropped = (~mono_mask).sum()
        if n_dropped:
            logger.debug(
                f"fit_from_df: dropped {n_dropped} non-monotone virtual-height "
                f"points before inversion"
            )
        f_arr = f_arr[mono_mask]
        h_arr = h_arr[mono_mask]

        return self.fit(freq_mhz=f_arr, h_virtual_km=h_arr)
