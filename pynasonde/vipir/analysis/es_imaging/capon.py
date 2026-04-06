"""capon.py — High-resolution sporadic-E layer imaging via Capon cross-spectrum analysis.

Theory
------
A pulse-compressed ionosonde range profile R_ss(t) = [R_s(t₁), …, R_s(t_V)] is a
complex sequence of V samples, one per range gate of width r₀ = c·t_p/2 (km), where
t_p is the code chip duration.  Taking the FFT along the gate axis yields the
**cross-power spectrum**:

    G_ss(f_m) = FFT{R_ss}(f_m) = U(f_m) · e^{j4πrf_m/c}          (1)

where U(f) = |S(f)|² encodes the transmitted signal power and the complex exponential
encodes target range r.  The phase difference between adjacent spectral components is

    Δφ = 2π · Q / V,   Q = r / r₀  (integer bin index)            (2)

which is purely proportional to range and free of integer ambiguity.

**Capon range-dimensional spectrum estimation**

Partition G_ss into a Hankel subband matrix G of shape (Z, V-Z+1), where Z is the
number of subbands:

    G[i, j] = G_ss[f_{i+j}],   i = 0…Z-1,  j = 0…V-Z              (3)

Form the spatial covariance (with diagonal loading ε for conditioning):

    R_f = G · G^H / (V-Z+1)  +  ε·tr(R_f)/Z · I                   (4)

Construct a steering matrix A of shape (K·V, Z), where K is the range resolution
improvement factor:

    A[l, k] = exp(j · k · ω_l),   ω_l = 2π·l / (K·V),  l = 1…K·V (5)

The Capon minimum-variance pseudospectrum at super-resolved range index l is:

    P(l · r₀/K) = 1 / (a^H(ω_l) · R_f⁻¹ · a(ω_l))                (6)

giving an effective range resolution of r₀/K.  Liu et al. (2023) achieve 384 m
(10× improvement) from an intrinsic 3.84 km range bin.

**VIPIR data mapping**

The VIPIR RIQ cube has shape (pulse_count, gate_count, rx_count):

    gate_count  ↔  V  (pulse-compressed range bins — the range profile R_ss)
    pulse_count ↔  256 duplicate soundings per frequency in Liu et al.
    rx_count    ↔  receive antenna channels (method is single-channel)

Applying the algorithm along the gate axis requires only a single receiver channel.
Multiple pulses can be coherently integrated (trading temporal resolution for SNR)
before imaging, exactly as in Liu et al. Fig 6.

**Practical constraints**

* Z must satisfy Z ≤ (V+1)/2 for R_f to be non-singular (full-rank condition).
  R_f has shape (Z×Z) but rank at most (V-Z+1); for non-singular inversion the
  rank must equal Z, requiring V-Z+1 ≥ Z → Z ≤ (V+1)/2.
  Z > (V+1)/2 is allowed (diagonal loading partially compensates) but imaging
  degrades — matches Liu et al. (2023) Figure 1d (Z=150, V=200 case).
  Practical recommendation: Z ≈ V/2 (e.g. Z=100 for V=200).
* K is a free parameter that only controls output grid spacing (Δr = r₀/K).
  It does NOT enter the covariance matrix R_f and has no singularity constraint.
* SNR > 10 dB is required for reliable layer separation (Liu et al. Fig 1f).

References
----------
Liu, T., Yang, G., & Jiang, C. (2023). High-resolution sporadic E layer observation
based on ionosonde using a cross-spectrum analysis imaging technique. *Space Weather*,
21, e2022SW003195. https://doi.org/10.1029/2022SW003195
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ===========================================================================
# Result dataclass
# ===========================================================================


@dataclass
class EsImagingResult:
    """High-resolution Es layer range imaging result for one sounding frequency.

    Parameters
    ----------
    pseudospectrum_db:
        Normalised Capon pseudospectrum in dB, shape ``(n_snapshots, n_hr_bins)``.
        Each row is one coherently-integrated snapshot.  Normalised so that the
        maximum over the full array is 0 dB.
    heights_km:
        Virtual height axis of the high-resolution grid (km), shape
        ``(n_hr_bins,)``.  Spacing is ``gate_spacing_km / resolution_factor``.
    gate_heights_km:
        Original gate heights (km), shape ``(n_gates,)``.
    n_subbands:
        Number of Capon subbands Z used.
    resolution_factor:
        Range resolution improvement factor K.  Effective resolution is
        ``gate_spacing_km / resolution_factor`` km.
    coherent_integrations:
        Number of pulses coherently integrated per snapshot.
    gate_spacing_km:
        Original gate spacing r₀ (km).
    """

    pseudospectrum_db: np.ndarray
    heights_km: np.ndarray
    gate_heights_km: np.ndarray
    n_subbands: int
    resolution_factor: int
    coherent_integrations: int
    gate_spacing_km: float

    @property
    def effective_resolution_km(self) -> float:
        """Effective range resolution after Capon imaging (km)."""
        return self.gate_spacing_km / self.resolution_factor

    @property
    def n_snapshots(self) -> int:
        return self.pseudospectrum_db.shape[0]

    def summary(self) -> str:
        return (
            f"EsImagingResult: snapshots={self.n_snapshots}  "
            f"Z={self.n_subbands}  K={self.resolution_factor}  "
            f"r₀={self.gate_spacing_km:.2f} km → "
            f"Δr={self.effective_resolution_km:.3f} km  "
            f"height={self.heights_km[0]:.1f}–{self.heights_km[-1]:.1f} km"
        )

    def to_dataframe(self, snapshot: int = 0) -> pd.DataFrame:
        """Return one snapshot as a DataFrame with columns height_km and power_db."""
        return pd.DataFrame(
            {
                "height_km": self.heights_km,
                "power_db": self.pseudospectrum_db[snapshot],
            }
        )

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        snapshot: Optional[int] = None,
        vmin: float = -60.0,
        cmap: str = "jet",
    ) -> plt.Axes:
        """Plot the imaging result.

        Parameters
        ----------
        ax:
            Existing axes.  A new figure is created when ``None``.
        snapshot:
            If given, plot only that snapshot as a 1-D profile.
            If ``None`` and ``n_snapshots > 1``, plot all snapshots as an
            intensity map (time × height).
        vmin:
            Minimum colour/y scale in dB.  Default ``-60``.
        cmap:
            Colour map for the 2-D plot.  Default ``"jet"``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        if snapshot is not None or self.n_snapshots == 1:
            idx = snapshot if snapshot is not None else 0
            ax.plot(self.pseudospectrum_db[idx], self.heights_km, color="steelblue")
            ax.set_xlabel("Normalised power (dB)")
            ax.set_ylabel("Virtual height (km)")
            ax.set_xlim(vmin, 2)
            ax.set_title(
                f"Es Capon image — snapshot {idx}  "
                f"(Δr = {self.effective_resolution_km:.3f} km)"
            )
            # overlay original gate positions
            ax.axhline(y=0, color="none")
            for gh in self.gate_heights_km:
                ax.axhline(y=gh, color="gray", lw=0.4, ls="--", alpha=0.5)
        else:
            t_axis = np.arange(self.n_snapshots) * self.coherent_integrations
            ax.pcolormesh(
                t_axis,
                self.heights_km,
                self.pseudospectrum_db.T,
                cmap=cmap,
                vmin=vmin,
                vmax=0,
                shading="auto",
            )
            ax.set_xlabel(f"Pulse index (× {self.coherent_integrations})")
            ax.set_ylabel("Virtual height (km)")
            ax.set_title(
                f"Es Capon image  Z={self.n_subbands}  K={self.resolution_factor}  "
                f"Δr={self.effective_resolution_km:.3f} km"
            )

        return ax


# ===========================================================================
# Processor class
# ===========================================================================


class EsCaponImager:
    """High-resolution Es layer range imager using Capon cross-spectrum analysis.

    Implements the algorithm of Liu et al. (2023) applied to the pulse-compressed
    range profile stored in the VIPIR RIQ I/Q cube.

    Parameters
    ----------
    n_subbands:
        Number of Capon subbands Z.  Must be less than ``gate_count``.
        Larger values improve resolution but increase the risk of covariance
        matrix singularity.  Default ``100``.
    resolution_factor:
        Range resolution improvement factor K.  Effective resolution becomes
        ``gate_spacing_km / K``.  Default ``10``.
    coherent_integrations:
        Number of pulses to coherently integrate per snapshot before imaging.
        ``1`` → per-pulse imaging (maximum temporal resolution).
        ``N`` → N-pulse integration (higher SNR, lower temporal resolution).
        Default ``1``.
    rx_index:
        Receive antenna channel to use when the IQ cube has an ``rx_count``
        third axis.  Default ``0``.
    diagonal_loading:
        Diagonal loading fraction for covariance matrix regularisation.
        R_f ← R_f + ε·tr(R_f)/Z · I.  Default ``1e-3``.
    gate_start_km:
        Virtual height of the first range gate (km).  Default ``90.0``
        (bottom of the E layer).
    gate_spacing_km:
        Spacing between adjacent range gates (km) = r₀ = c·t_p/2.
        For WISS this is 3.84 km; check your ionosonde parameters.
        Default ``3.84``.

    Examples
    --------
    >>> imager = EsCaponImager(n_subbands=100, resolution_factor=10,
    ...                        gate_spacing_km=3.84, gate_start_km=90.0)
    >>> result = imager.fit(iq_cube)          # shape (pulse_count, gate_count, rx)
    >>> print(result.summary())
    >>> result.plot()
    """

    def __init__(
        self,
        n_subbands: int = 100,
        resolution_factor: int = 10,
        coherent_integrations: int = 1,
        rx_index: int = 0,
        diagonal_loading: float = 1e-3,
        gate_start_km: float = 90.0,
        gate_spacing_km: float = 3.84,
    ) -> None:
        self.Z = n_subbands
        self.K = resolution_factor
        self.n_coh = max(1, coherent_integrations)
        self.rx_index = rx_index
        self.eps = diagonal_loading
        self.gate_start_km = gate_start_km
        self.gate_spacing_km = gate_spacing_km

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, V: int) -> None:
        """Check Z constraints against actual gate count V.

        The only hard singularity constraint is on Z, not K.  K merely sets
        the output grid density (ω_l = 2π·l/(K·V)) and does NOT affect the
        covariance matrix R_f.

        R_f = G · G^H / (V-Z+1)  has shape (Z, Z) and rank at most (V-Z+1).
        For R_f to be non-singular:  V - Z + 1 ≥ Z  →  Z ≤ (V+1)/2.
        Above this threshold diagonal loading helps but imaging degrades,
        matching Liu et al. (2023) Figure 1d (Z=150 with V=200).
        """
        if self.Z >= V:
            raise ValueError(
                f"n_subbands Z={self.Z} must be < gate_count V={V}.  "
                f"Reduce n_subbands or increase the range window."
            )
        z_crit = (V + 1) // 2
        if self.Z > z_crit:
            logger.warning(
                f"n_subbands Z={self.Z} > (V+1)/2={z_crit}; "
                f"covariance R_f is rank-deficient — imaging will degrade "
                f"(diagonal loading partially compensates)."
            )

    def _covariance(self, G_ss: np.ndarray) -> np.ndarray:
        """Build regularised Capon covariance matrix from one cross-power spectrum.

        Parameters
        ----------
        G_ss : complex ndarray, shape (V,)
            FFT of the pulse-compressed range profile (already frequency-domain).

        Returns
        -------
        R_f_inv : complex ndarray, shape (Z, Z)
            Inverse of the regularised covariance matrix.
        """
        V = len(G_ss)
        Z = self.Z
        cols = V - Z + 1
        idx = np.arange(Z)[:, None] + np.arange(cols)[None, :]
        G = G_ss[idx]  # (Z, cols)
        R_f = (G @ G.conj().T) / cols
        load = self.eps * np.real(np.trace(R_f)) / Z
        R_f += load * np.eye(Z, dtype=complex)
        return np.linalg.inv(R_f)

    def _covariance_multi(
        self, profiles: np.ndarray, gate_blank: int = 0
    ) -> np.ndarray:
        """Build regularised Capon covariance averaged over L range profiles.

        Averaging L covariance matrices before inversion is the standard
        multi-snapshot Capon estimator — a better-conditioned R_f than any
        single-snapshot estimate.  For L = window × n_pulse × n_rx this gives
        the maximum SNR improvement available from the stacked data.

        Parameters
        ----------
        profiles : complex ndarray, shape (L, V)
            L independent pulse-compressed range profiles (one per (pulse, Rx,
            file) combination after beamforming is replaced by stacking).

        Returns
        -------
        R_f_inv : complex ndarray, shape (Z, Z)
            Inverse of the averaged, regularised covariance matrix.
        """
        L, V = profiles.shape
        Z = self.Z
        cols = V - Z + 1

        # Apply gate blanking to suppress direct-wave / ground-clutter
        if gate_blank > 0:
            profiles = profiles.copy()
            profiles[:, :gate_blank] = 0.0

        # Batch FFT: (L, V) → (L, V) cross-power spectra
        G_ss = np.fft.fft(profiles, axis=1)  # (L, V)

        # Hankel index array (same for all profiles)
        idx = np.arange(Z)[:, None] + np.arange(cols)[None, :]  # (Z, cols)

        # Hankel matrices for all L profiles at once: (L, Z, cols)
        G_all = G_ss[:, idx]  # advanced indexing broadcasts correctly

        # Averaged covariance: R_f = (1/L·cols) Σ_l G[l] G[l]^H
        R_f = np.einsum("lzi,lki->zk", G_all, G_all.conj()) / (L * cols)

        # Diagonal loading for conditioning
        load = self.eps * np.real(np.trace(R_f)) / Z
        R_f += load * np.eye(Z, dtype=complex)

        return np.linalg.inv(R_f)

    def _steering_matrix(self, V: int) -> np.ndarray:
        """Construct steering matrix A of shape (K·V, Z).

        A[l, k] = exp(j · k · ω_l),  ω_l = 2π·l / (K·V)
        """
        K, Z = self.K, self.Z
        l_vals = np.arange(1, K * V + 1)  # (K*V,)
        omega = 2.0 * np.pi * l_vals / (K * V)  # (K*V,)
        k_vals = np.arange(Z)  # (Z,)
        return np.exp(1j * np.outer(omega, k_vals))  # (K*V, Z)

    def _capon(self, R_f_inv: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Compute Capon pseudospectrum P[l] = 1 / (a^H R_f⁻¹ a).

        Parameters
        ----------
        R_f_inv : complex ndarray (Z, Z)
        A       : complex ndarray (K*V, Z)  — steering matrix

        Returns
        -------
        P : real ndarray (K*V,)  — pseudospectrum (linear power)
        """
        # a^H R_f_inv a = einsum('li,ij,lj->l', A.conj(), R_f_inv, A)
        # Computed as two matrix products to avoid O((KV)² Z) cost.
        tmp = A.conj() @ R_f_inv  # (K*V, Z)
        denom = np.real(np.einsum("li,li->l", tmp, A))
        return 1.0 / np.where(denom > 0, denom, np.finfo(float).tiny)

    def _process_pulse(self, range_profile: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Image one range profile.  Returns linear pseudospectrum (K*V,)."""
        G_ss = np.fft.fft(range_profile)  # cross-power spectrum
        R_f_inv = self._covariance(G_ss)
        return self._capon(R_f_inv, A)

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def fit(
        self,
        iq_cube: Union[np.ndarray, np.ndarray],
    ) -> EsImagingResult:
        """Run high-resolution Es layer imaging on a VIPIR IQ cube.

        Parameters
        ----------
        iq_cube : complex ndarray
            Shape ``(pulse_count, gate_count)`` or
            ``(pulse_count, gate_count, rx_count)``.
            The gate axis must contain pulse-compressed range bins (the
            cross-correlation output stored in RIQ files).

        Returns
        -------
        EsImagingResult
        """
        # ── normalise input shape ──────────────────────────────────────
        arr = np.asarray(iq_cube, dtype=complex)
        if arr.ndim == 2:
            # (pulse_count, gate_count) — already single channel
            cube = arr
        elif arr.ndim == 3:
            cube = arr[:, :, self.rx_index]
        else:
            raise ValueError(f"iq_cube must be 2-D or 3-D, got shape {arr.shape}.")

        n_pulses, V = cube.shape
        self._validate(V)

        # ── precompute height axes ─────────────────────────────────────
        gate_heights = self.gate_start_km + np.arange(V) * self.gate_spacing_km
        hr_heights = self.gate_start_km + np.arange(self.K * V) * (
            self.gate_spacing_km / self.K
        )

        # ── precompute steering matrix (same for all snapshots) ────────
        A = self._steering_matrix(V)

        # ── coherent integration & imaging ─────────────────────────────
        n_snapshots = max(1, n_pulses // self.n_coh)
        spectra = []

        for s in range(n_snapshots):
            i0 = s * self.n_coh
            i1 = min(i0 + self.n_coh, n_pulses)
            profile = cube[i0:i1, :].mean(axis=0)  # coherent integration
            P = self._process_pulse(profile, A)
            spectra.append(P)

        spectra_arr = np.stack(spectra, axis=0)  # (n_snapshots, K*V)

        # ── normalise to dB ────────────────────────────────────────────
        P_max = spectra_arr.max()
        if P_max <= 0:
            P_max = 1.0
        spectra_db = 10.0 * np.log10(spectra_arr / P_max + 1e-15)

        logger.info(
            f"EsCaponImager: V={V}  Z={self.Z}  K={self.K}  "
            f"snapshots={n_snapshots}  "
            f"Δr={self.gate_spacing_km/self.K:.3f} km  "
            f"h={hr_heights[0]:.1f}–{hr_heights[-1]:.1f} km"
        )

        return EsImagingResult(
            pseudospectrum_db=spectra_db,
            heights_km=hr_heights,
            gate_heights_km=gate_heights,
            n_subbands=self.Z,
            resolution_factor=self.K,
            coherent_integrations=self.n_coh,
            gate_spacing_km=self.gate_spacing_km,
        )
