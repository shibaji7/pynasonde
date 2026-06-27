"""nextyz.py — NeXtYZ 3-D electron density inversion for Dynasonde ionograms.

Physics-based implementation of:

    Zabotin, N. A., Wright, J. W., & Zhbankov, G. A. (2006).
    NeXtYZ: Three-dimensional electron density inversion for dynasonde
    ionograms. *Radio Science*, 41, RS6S32, doi:10.1029/2005RS003352.

The **Wedge-Stratified Ionosphere (WSI)** model represents the local
electron density as a stack of plasma-frequency wedges.  Each wedge is
bounded above by a **frame plane** whose orientation (nₓ, nᵧ) encodes
the ionospheric tilt at that height.  A Hamiltonian ray-tracer (eikonal /
method of characteristics with the full Appleton-Lassen refractive index)
propagates sounding signals through the WSI model.  Wedge parameters are
determined bottom-up in a least-squares loop that alternately minimises:

1. Group-range residual  ΔR'ᵢ₊₁ = √Σⱼ(ρ'ᵢ₊₁,ⱼ − R'ᵢ₊₁,ⱼ)²
2. Ground-return distance of the mean-direction ray (tilt constraint)

**NeXtYZ Lite** (default) — constant tilts per broad layer (E, F) derived
from mean angles of arrival; only hᵢ₊₁ is optimised per wedge.  ~6× faster.

**NeXtYZ Full** — (hᵢ₊₁, nₓᵢ₊₁, nᵧᵢ₊₁) solved per wedge with alternating
optimisation.

This module provides:

:class:`WedgePlane`
    Dataclass holding the solved parameters of one WSI wedge boundary.

:class:`NeXtYZResult`
    Output dataclass — fp(h) profile, tilt angles, height error bars.

:class:`NeXtYZInverter`
    Processor — runs the full inversion pipeline from an echo DataFrame.

Coordinate system
-----------------
Local Cartesian centred at the sounder::

    x = geographic East  (km)
    y = geographic North (km)
    z = vertical Up      (km)

The ODE independent variable τ (km) parameterises ray trajectories so
that **dr/dτ = group-slowness direction** (dimensionless).  Group range
R' = c·t (km) accumulated during integration.

Required DataFrame columns
--------------------------
xl_km, yl_km     — Dynasonde echolocation coordinates (km)
height_km        — Observed group range R' (km)
frequency_khz    — Sounding frequency (kHz)
mode             — Magnetoionic polarization "O" or "X"  (optional)
amplitude_db     — Echo amplitude in dB                  (optional)

Notes
-----
* Collisions are neglected (valid for E and F regions, paper §4).
* For the first wedge the Titheridge underlying-ionisation model is
  approximated by a linear ramp from 0 to fp_start below the lowest
  observed echo.  Full Titheridge model integration is future work.
* NeXtYZ replaces POLAN as the standard profile inversion algorithm for
  Dynasonde data.  It has been deployed operationally since March 2005.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar

# ── Physical constants ───────────────────────────────────────────────────────
_GYR_MHZ_PER_GAUSS = 2.80  # electron gyrofrequency, MHz gauss⁻¹
_FP2_TO_N_CM3 = 1.2399e4  # N_cm3 = fp_mhz² × this

# ── WSI defaults (paper §5) ──────────────────────────────────────────────────
_FP_STEP_MHZ = 0.05  # standard plasma-frequency step
_MIN_ECHOES = 10  # minimum echoes per wedge
_MAX_ECHOES = 50  # maximum echoes used per wedge
_TAU_MAX_KM = 2500.0  # ODE integration path limit
_H_BASE_KM = 80.0  # base of ionospheric model (km)
_MAX_STEP_KM = 5.0  # ODE max step size


# ===========================================================================
# WSI frame-plane geometry
# ===========================================================================


def _nz(nx: float, ny: float) -> float:
    """Vertical component of a unit frame-plane normal vector (always ≥ 0)."""
    return float(np.sqrt(max(1.0 - nx**2 - ny**2, 1e-12)))


def _plane_signed_dist(r: np.ndarray, h: float, nx: float, ny: float) -> float:
    """Signed distance from point r to a frame plane (h, nx, ny).

    Positive on the side in the direction of the outward normal
    (above the plane when nz > 0).
    """
    return float(nx * r[0] + ny * r[1] + _nz(nx, ny) * (r[2] - h))


def _wedge_rho(
    r: np.ndarray,
    h_lo: float,
    nx_lo: float,
    ny_lo: float,
    h_hi: float,
    nx_hi: float,
    ny_hi: float,
) -> float:
    """Fractional position ρ ∈ [0, 1] within a WSI wedge.

    ρ = l_lo / (l_lo + l_hi) where l_lo, l_hi are the perpendicular
    distances from r to the lower and upper frame planes respectively
    (variable ρ defined below eq. 10 in the paper).
    """
    l_lo = _plane_signed_dist(r, h_lo, nx_lo, ny_lo)
    l_hi = -_plane_signed_dist(r, h_hi, nx_hi, ny_hi)
    total = l_lo + l_hi
    if total < 1e-9:
        return 0.5
    return float(np.clip(l_lo / total, 0.0, 1.0))


def _wedge_grad_rho(
    r: np.ndarray,
    h_lo: float,
    nx_lo: float,
    ny_lo: float,
    h_hi: float,
    nx_hi: float,
    ny_hi: float,
) -> np.ndarray:
    """Spatial gradient ∂ρ/∂r within a WSI wedge (3-vector, km⁻¹)."""
    l_lo = _plane_signed_dist(r, h_lo, nx_lo, ny_lo)
    l_hi = -_plane_signed_dist(r, h_hi, nx_hi, ny_hi)
    total = l_lo + l_hi
    if total < 1e-9:
        return np.zeros(3)
    n_lo = np.array([nx_lo, ny_lo, _nz(nx_lo, ny_lo)])  # ∂l_lo/∂r
    n_hi = np.array([nx_hi, ny_hi, _nz(nx_hi, ny_hi)])  # ∂(-l_hi)/∂r reversed
    # ρ = A/(A+B) → ∂ρ/∂r = (n_lo·B − l_lo·(n_lo−n_hi)) / (A+B)²
    grad = (n_lo * l_hi + l_lo * n_hi) / (total**2)
    return grad


# ===========================================================================
# Appleton-Lassen refractive index (collisionless)
# ===========================================================================


def _appleton_lassen_n2(X: float, Y_L: float, Y_T2: float, polarization: str) -> float:
    """Collisionless Appleton-Lassen refractive index squared.

    Follows Budden (1985) as cited in paper §4::

        n² = 1 − 2X(1−X) / [2(1−X) − Y_T² ± D]

        D = √(Y_T⁴ + 4·Y_L²·(1−X)²)

    O-mode: upper sign (+ D); X-mode: lower sign (− D).

    Parameters
    ----------
    X : fp²/f² (plasma frequency² / sounding frequency²).
    Y_L : Longitudinal magneto-ionic parameter Y·cosα.  Y = fH/f.
    Y_T2 : Y_T² = Y² − Y_L² (transverse term squared).
    polarization : ``"O"`` or ``"X"``.

    Returns
    -------
    float — n² ≥ 0 (clipped at reflection condition).
    """
    if X >= 1.0 - 1e-9:
        return 0.0
    D = float(np.sqrt(max(Y_T2**2 + 4.0 * Y_L**2 * (1.0 - X) ** 2, 0.0)))
    numer = 2.0 * X * (1.0 - X)
    denom = (
        (2.0 * (1.0 - X) - Y_T2 + D)
        if polarization == "O"
        else (2.0 * (1.0 - X) - Y_T2 - D)
    )
    if abs(denom) < 1e-12:
        return 0.0
    return float(max(1.0 - numer / denom, 0.0))


def _d_n2_dX_numerical(
    X: float, Y_L: float, Y_T2: float, polarization: str, eps: float = 1e-5
) -> float:
    """∂n²/∂X via central finite difference."""
    return (
        _appleton_lassen_n2(X + eps, Y_L, Y_T2, polarization)
        - _appleton_lassen_n2(X - eps, Y_L, Y_T2, polarization)
    ) / (2.0 * eps)


# ===========================================================================
# Output dataclasses
# ===========================================================================


@dataclass
class WedgePlane:
    """Solved parameters for one WSI wedge boundary.

    Parameters
    ----------
    fp_lo_mhz, fp_hi_mhz:
        Plasma-frequency bounds of the wedge (MHz).
    h_upper_km:
        Vertical height of the upper frame plane (km).
    nx, ny:
        Horizontal components of the upper frame plane's unit normal
        vector.  ``nz = sqrt(1 − nx² − ny²)``.
    residual_km:
        RMS group-range residual ΔR'ᵢ₊₁ converted to height (km).
    n_echoes:
        Number of echoes used in this wedge's optimisation.
    """

    fp_lo_mhz: float
    fp_hi_mhz: float
    h_upper_km: float
    nx: float = 0.0
    ny: float = 0.0
    residual_km: float = np.nan
    n_echoes: int = 0


@dataclass
class NeXtYZResult:
    """Output of the NeXtYZ inversion.

    Parameters
    ----------
    wedges:
        List of solved :class:`WedgePlane` objects (bottom to top).
    method:
        ``"Lite"`` or ``"Full"``.
    fp_profile_mhz:
        Plasma frequency at each solved frame plane (MHz).
    h_true_km:
        True height at each frame plane (km).
    h_errors_km:
        Real-height error estimates Δhᵢ₊₁ from paper §7.
    tilt_meridional_deg:
        Frame-plane tilt from vertical in the meridional (x–z) plane (degrees).
        Computed as arctan2(nₓ, nz) where nz = √(1 − nₓ² − nᵧ²).
    tilt_zonal_deg:
        Frame-plane tilt from vertical in the zonal (y–z) plane (degrees).
        Computed as arctan2(nᵧ, nz).
    converged:
        Boolean per wedge (True when ΔR' < 10 km).
    """

    wedges: List[WedgePlane]
    method: str
    fp_profile_mhz: np.ndarray = field(default_factory=lambda: np.array([]))
    h_true_km: np.ndarray = field(default_factory=lambda: np.array([]))
    h_errors_km: np.ndarray = field(default_factory=lambda: np.array([]))
    tilt_meridional_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    tilt_zonal_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: List[bool] = field(default_factory=list)

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Return the profile as a DataFrame (one row per wedge boundary)."""
        if not self.wedges:
            return pd.DataFrame()
        rows = []
        n = len(self.wedges)
        for k, w in enumerate(self.wedges):
            rows.append(
                dict(
                    fp_lo_mhz=w.fp_lo_mhz,
                    fp_hi_mhz=w.fp_hi_mhz,
                    h_upper_km=(
                        self.h_true_km[k] if k < len(self.h_true_km) else w.h_upper_km
                    ),
                    h_error_km=(
                        self.h_errors_km[k] if k < len(self.h_errors_km) else np.nan
                    ),
                    fp_mhz=(
                        self.fp_profile_mhz[k]
                        if k < len(self.fp_profile_mhz)
                        else np.nan
                    ),
                    tilt_meridional_deg=(
                        self.tilt_meridional_deg[k]
                        if k < len(self.tilt_meridional_deg)
                        else 0.0
                    ),
                    tilt_zonal_deg=(
                        self.tilt_zonal_deg[k] if k < len(self.tilt_zonal_deg) else 0.0
                    ),
                    nx=w.nx,
                    ny=w.ny,
                    residual_km=w.residual_km,
                    n_echoes=w.n_echoes,
                    converged=self.converged[k] if k < len(self.converged) else False,
                )
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """One-line text summary."""
        n = len(self.wedges)
        if n == 0:
            return "NeXtYZResult: no wedges solved."
        h_top = self.h_true_km[-1] if len(self.h_true_km) else np.nan
        fp_top = self.fp_profile_mhz[-1] if len(self.fp_profile_mhz) else np.nan
        return (
            f"NeXtYZResult [{self.method}]: {n} wedges  "
            f"h_top={h_top:.0f} km  foF2≈{fp_top:.2f} MHz  "
            f"mean_Δh={np.nanmean(self.h_errors_km):.1f} km"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot the fp(h) profile with error bars and tilt angles.

        When ``ax`` is ``None`` a new two-panel figure is created
        (profile left, tilt angles right, sharing the height axis).

        Parameters
        ----------
        ax:
            Optional existing axes for the profile panel only.

        Returns
        -------
        matplotlib.axes.Axes — profile axes.
        """
        if ax is None:
            fig, (ax_prof, ax_tilt) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        else:
            ax_prof = ax
            ax_tilt = None

        if len(self.h_true_km) > 0:
            ax_prof.errorbar(
                self.fp_profile_mhz,
                self.h_true_km,
                xerr=None,
                yerr=self.h_errors_km,
                fmt="o-",
                color="steelblue",
                ecolor="lightblue",
                capsize=3,
                label=f"NeXtYZ {self.method}",
            )
        ax_prof.set_xlabel("Plasma frequency (MHz)")
        ax_prof.set_ylabel("True height (km)")
        ax_prof.set_title(f"NeXtYZ {self.method}  —  fp(h)")
        ax_prof.legend()
        ax_prof.grid(True, alpha=0.3)

        if ax_tilt is not None and len(self.h_true_km) > 0:
            ax_tilt.plot(
                self.tilt_meridional_deg,
                self.h_true_km,
                "r-o",
                ms=4,
                label="Meridional (Θₓ)",
            )
            ax_tilt.plot(
                self.tilt_zonal_deg,
                self.h_true_km,
                "b-o",
                ms=4,
                label="Zonal (Θᵧ)",
            )
            ax_tilt.axvline(0, color="k", lw=0.5)
            ax_tilt.set_xlabel("Tilt angle (°)")
            ax_tilt.set_title("Layer tilts")
            ax_tilt.legend()
            ax_tilt.grid(True, alpha=0.3)

        return ax_prof


# ===========================================================================
# Main inversion class
# ===========================================================================


class NeXtYZInverter:
    """NeXtYZ 3-D electron density inversion for Dynasonde ionograms.

    Implements the WSI model and Hamiltonian ray-tracing inversion of
    Zabotin, Wright & Zhbankov (2006) Radio Sci. 41, RS6S32.

    Parameters
    ----------
    dip_angle_deg:
        Geomagnetic dip angle at the station (degrees, positive downward
        in the northern hemisphere).
    declination_deg:
        Geomagnetic declination (degrees, east positive).
    B_gauss:
        Geomagnetic field magnitude (gauss).  Electron gyrofrequency
        fH = 2.80 × B_gauss (MHz).
    fp_step_mhz:
        Plasma-frequency step per wedge (MHz).  Default 0.05 (paper §5).
    min_echoes:
        Minimum echoes required to solve a wedge.  Default 10 (paper §5).
    max_echoes:
        Maximum echoes used per wedge (top-amplitude selected).
        Default 50 (paper §5).
    mode:
        ``"Lite"`` — constant tilts per layer, hᵢ₊₁ only (fast).
        ``"Full"`` — (hᵢ₊₁, nₓᵢ₊₁, nᵧᵢ₊₁) per wedge.
    fp_start_mhz:
        Lowest plasma frequency to begin inversion (MHz).  Default 0.5.
    xl_col, yl_col, height_col, freq_col, mode_col, amp_col:
        Column names in the input echo DataFrame.

    Examples
    --------
    >>> inv = NeXtYZInverter(
    ...     dip_angle_deg=70.0,
    ...     declination_deg=-5.0,
    ...     B_gauss=0.55,
    ...     mode="Lite",
    ... )
    >>> result = inv.fit(echo_df)
    >>> print(result.summary())
    ScaledParameters: foE=… MHz …

    Notes
    -----
    Input DataFrame must contain echolocation columns ``xl_km``, ``yl_km``
    (km) in addition to the standard ``height_km`` (group range R') and
    ``frequency_khz``.  These come from the Dynasonde seven-parameter set:
    XL, YL are the echolocation coordinates of the group path vector endpoint
    (Paul et al. 1974).
    """

    def __init__(
        self,
        dip_angle_deg: float,
        declination_deg: float,
        B_gauss: float,
        fp_step_mhz: float = _FP_STEP_MHZ,
        min_echoes: int = _MIN_ECHOES,
        max_echoes: int = _MAX_ECHOES,
        mode: str = "Lite",
        fp_start_mhz: float = 0.5,
        xl_col: str = "xl_km",
        yl_col: str = "yl_km",
        height_col: str = "height_km",
        freq_col: str = "frequency_khz",
        mode_col: str = "mode",
        amp_col: str = "amplitude_db",
    ) -> None:
        self.dip_deg = dip_angle_deg
        self.dec_deg = declination_deg
        self.B_gauss = B_gauss
        self.fH_mhz = _GYR_MHZ_PER_GAUSS * B_gauss
        self.fp_step = fp_step_mhz
        self.min_echo = min_echoes
        self.max_echo = max_echoes
        self.mode = mode
        self.fp_start = fp_start_mhz

        # DataFrame column names
        self._xl = xl_col
        self._yl = yl_col
        self._h = height_col
        self._f = freq_col
        self._m = mode_col
        self._a = amp_col

        # Geomagnetic field unit vector in (East, North, Up) frame.
        # B points downward in NH → negative Up component.
        dip = np.radians(dip_angle_deg)
        dec = np.radians(declination_deg)
        self._B_hat = np.array(
            [
                np.cos(dip) * np.sin(dec),  # East
                np.cos(dip) * np.cos(dec),  # North
                -np.sin(dip),  # Up (negative = downward)
            ]
        )

    # ------------------------------------------------------------------
    # Magneto-ionic helpers
    # ------------------------------------------------------------------

    def _YL_YT2(self, p: np.ndarray, f_mhz: float) -> Tuple[float, float]:
        """Longitudinal Y_L and transverse Y_T² for wave-vector direction p."""
        Y = self.fH_mhz / f_mhz
        norm = np.linalg.norm(p)
        p_hat = p / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        Y_L = float(Y * np.dot(p_hat, self._B_hat))
        Y_T2 = float(max(Y**2 - Y_L**2, 0.0))
        return Y_L, Y_T2

    def _n2(self, X: float, p: np.ndarray, f_mhz: float, pol: str) -> float:
        """Appleton-Lassen n² at position r (via X) and ray direction p."""
        Y_L, Y_T2 = self._YL_YT2(p, f_mhz)
        return _appleton_lassen_n2(X, Y_L, Y_T2, pol)

    def _dn2_dX(self, X: float, p: np.ndarray, f_mhz: float, pol: str) -> float:
        """∂n²/∂X (numerical central difference)."""
        Y_L, Y_T2 = self._YL_YT2(p, f_mhz)
        return _d_n2_dX_numerical(X, Y_L, Y_T2, pol)

    def _grad_n2_p(self, X: float, p: np.ndarray, f_mhz: float, pol: str) -> np.ndarray:
        """∂n²/∂p — finite difference over wave-vector direction (3-vector)."""
        eps = 1e-5
        grad = np.zeros(3)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = eps
            YLp, YT2p = self._YL_YT2(p + dp, f_mhz)
            YLm, YT2m = self._YL_YT2(p - dp, f_mhz)
            n2p = _appleton_lassen_n2(X, YLp, YT2p, pol)
            n2m = _appleton_lassen_n2(X, YLm, YT2m, pol)
            grad[i] = (n2p - n2m) / (2.0 * eps)
        return grad

    # ------------------------------------------------------------------
    # WSI density model
    # ------------------------------------------------------------------

    def _wsi_X_and_grad(
        self,
        r: np.ndarray,
        f_mhz: float,
        wedges: List[WedgePlane],
        h_lo_base: float = _H_BASE_KM,
        nx_lo_base: float = 0.0,
        ny_lo_base: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        """Return (X, ∂X/∂r) at position r in the current WSI model.

        Uses the vertical coordinate z to locate the wedge, then
        computes ρ from the full tilted frame-plane geometry.

        Parameters
        ----------
        r : (3,) position in km.
        f_mhz : sounding frequency.
        wedges : solved wedges below the current integration wedge.
        h_lo_base, nx_lo_base, ny_lo_base : parameters of the lowest frame plane.

        Returns
        -------
        (X, grad_X) — X dimensionless; grad_X in km⁻¹.
        """
        z = r[2]
        f2 = f_mhz**2

        if not wedges or z <= h_lo_base:
            return 0.0, np.zeros(3)

        # Build frame-plane list: [base, wedge0_top, wedge1_top, ...]
        planes_h = [h_lo_base] + [w.h_upper_km for w in wedges]
        planes_nx = [nx_lo_base] + [w.nx for w in wedges]
        planes_ny = [ny_lo_base] + [w.ny for w in wedges]
        fp_vals = [wedges[0].fp_lo_mhz] + [w.fp_hi_mhz for w in wedges]

        # Find wedge by z (valid for small tilts)
        idx = None
        for k in range(len(wedges)):
            if planes_h[k] <= z <= planes_h[k + 1]:
                idx = k
                break

        if idx is None:
            # Above all wedges: return top density
            return float(fp_vals[-1] ** 2 / f2), np.zeros(3)

        h_lo = planes_h[idx]
        nx_lo = planes_nx[idx]
        ny_lo = planes_ny[idx]
        h_hi = planes_h[idx + 1]
        nx_hi = planes_nx[idx + 1]
        ny_hi = planes_ny[idx + 1]
        fp_lo = fp_vals[idx]
        fp_hi = fp_vals[idx + 1]

        rho = _wedge_rho(r, h_lo, nx_lo, ny_lo, h_hi, nx_hi, ny_hi)
        grad_rho = _wedge_grad_rho(r, h_lo, nx_lo, ny_lo, h_hi, nx_hi, ny_hi)

        fp2_r = fp_lo**2 + rho * (fp_hi**2 - fp_lo**2)
        X = fp2_r / f2
        grad_X = (fp_hi**2 - fp_lo**2) / f2 * grad_rho

        return float(X), grad_X

    # ------------------------------------------------------------------
    # Hamiltonian ray tracing (eikonal / method of characteristics)
    # ------------------------------------------------------------------

    def _ray_ode(
        self,
        tau: float,
        state: np.ndarray,
        f_mhz: float,
        pol: str,
        wedges: List[WedgePlane],
        h_lo_base: float,
        nx_lo_base: float,
        ny_lo_base: float,
    ) -> np.ndarray:
        """Hamilton's equations for the ray (Zabotin et al. 2006, eq. 2).

        State vector: [x, y, z, pₓ, pᵧ, p_z, c·t]  (km, dimensionless, km).

        Equations::

            dr/dτ  = p − ½ ∂n²/∂p          (group-velocity direction)
            dp/dτ  = ½ ∂n²/∂r               (wave-vector refraction)
            c dt/dτ = n² + ½ ω ∂n²/∂ω      (group path accumulation)
                     = n² − X ∂n²/∂X        (using ω ∂n²/∂ω = −2X ∂n²/∂X)
        """
        r = state[:3]
        p = state[3:6]

        X, grad_X = self._wsi_X_and_grad(
            r, f_mhz, wedges, h_lo_base, nx_lo_base, ny_lo_base
        )
        n2_val = self._n2(X, p, f_mhz, pol)

        # ∂n²/∂r via chain rule through X(r)
        dn2_dX = self._dn2_dX(X, p, f_mhz, pol)
        grad_n2r = dn2_dX * grad_X

        # ∂n²/∂p (finite difference)
        grad_n2p = self._grad_n2_p(X, p, f_mhz, pol)

        # ω ∂n²/∂ω = −2X · (∂n²/∂X)
        omega_dn2_domega = -2.0 * X * dn2_dX

        dr_dtau = p - 0.5 * grad_n2p
        dp_dtau = 0.5 * grad_n2r
        dct_dtau = n2_val + 0.5 * omega_dn2_domega

        return np.concatenate([dr_dtau, dp_dtau, [dct_dtau]])

    def _trace_ray(
        self,
        xl_km: float,
        yl_km: float,
        r_prime_obs_km: float,
        f_mhz: float,
        pol: str,
        wedges: List[WedgePlane],
        h_trial_upper: float,
        nx_trial: float,
        ny_trial: float,
        h_lo_base: float = _H_BASE_KM,
        nx_lo_base: float = 0.0,
        ny_lo_base: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Trace one ray from the sounder to its turning point.

        The echolocation vector (xl, yl, R') defines the ray's direction-of-
        arrival at the ground, which serves as the initial condition for the
        Hamiltonian ODE (paper §4 final paragraph and §5 para 31).

        Only the upward half is integrated — the downward branch is assumed
        symmetric (paper §4, para 29).  Predicted group range = c·t
        accumulated to the turning point.

        Parameters
        ----------
        xl_km, yl_km : Dynasonde echolocation coordinates (km).
        r_prime_obs_km : Observed group range R' (km).
        f_mhz : Sounding frequency (MHz).
        pol : ``"O"`` or ``"X"``.
        wedges : Already-solved WSI wedges (frozen).
        h_trial_upper, nx_trial, ny_trial : Current trial upper frame plane.
        h_lo_base, nx_lo_base, ny_lo_base : Lowest frame plane.

        Returns
        -------
        (predicted_R_prime_km, x_turn_km, y_turn_km)
            ``NaN`` values on integration failure.
        """
        # Initial wave vector direction from echolocation coordinates.
        # Dynasonde definition (Paul et al. 1974):
        #   xl_km = R' · sin(θ) · sin(φ)   (East component of group path)
        #   yl_km = R' · sin(θ) · cos(φ)   (North component of group path)
        # so the initial unit wave-vector is directly (xl/R', yl/R', cos θ).
        # Do NOT use arctan2(horiz, R') — that gives arctan(sin θ) ≠ θ.
        r_inv = 1.0 / max(r_prime_obs_km, 1e-6)
        px0 = float(xl_km) * r_inv
        py0 = float(yl_km) * r_inv
        pz0 = float(np.sqrt(max(1.0 - px0**2 - py0**2, 0.0)))

        # Append the trial wedge temporarily so the ODE sees the current stack
        trial = WedgePlane(
            fp_lo_mhz=wedges[-1].fp_hi_mhz if wedges else self.fp_start,
            fp_hi_mhz=(
                (wedges[-1].fp_hi_mhz + self.fp_step)
                if wedges
                else self.fp_start + self.fp_step
            ),
            h_upper_km=h_trial_upper,
            nx=nx_trial,
            ny=ny_trial,
        )
        full_wedges = list(wedges) + [trial]

        state0 = np.array([0.0, 0.0, 0.0, px0, py0, pz0, 0.0])

        # Stopping event: ray z exceeds observed R' × 1.1 (generous ceiling)
        def z_ceiling(tau, s, *a):
            return s[2] - r_prime_obs_km * 1.1

        z_ceiling.terminal = True
        z_ceiling.direction = 1.0

        def below_ground(tau, s, *a):
            return s[2]

        below_ground.terminal = True
        below_ground.direction = -1.0

        try:
            sol = solve_ivp(
                self._ray_ode,
                t_span=(0.0, _TAU_MAX_KM),
                y0=state0,
                method="RK45",
                args=(f_mhz, pol, full_wedges, h_lo_base, nx_lo_base, ny_lo_base),
                events=[z_ceiling, below_ground],
                max_step=_MAX_STEP_KM,
                rtol=1e-4,
                atol=1e-6,
            )
            x_turn = float(sol.y[0, -1])
            y_turn = float(sol.y[1, -1])
            ct_turn = float(sol.y[6, -1])  # group path (km)
            return ct_turn, x_turn, y_turn

        except Exception as exc:
            logger.debug(f"Ray trace failed (f={f_mhz:.2f} MHz, pol={pol}): {exc}")
            return np.nan, np.nan, np.nan

    # ------------------------------------------------------------------
    # Per-wedge optimisation helpers
    # ------------------------------------------------------------------

    def _select_echoes(
        self, df: pd.DataFrame, fp_lo: float, fp_hi: float
    ) -> pd.DataFrame:
        """Select echoes reflected within [fp_lo, fp_hi] MHz.

        Filters by frequency band, then selects up to ``max_echoes``
        highest-amplitude echoes (paper §5 para 30).
        """
        mask = df[self._f].between(fp_lo * 1e3, fp_hi * 1e3)
        sub = df[mask].copy()
        if self._a in sub.columns and len(sub) > self.max_echo:
            sub = sub.nlargest(self.max_echo, self._a)
        return sub

    def _group_range_residual(
        self,
        h_upper: float,
        nx: float,
        ny: float,
        echoes: pd.DataFrame,
        f_mhz: float,
        wedges: List[WedgePlane],
        h_lo_base: float,
        nx_lo_base: float,
        ny_lo_base: float,
    ) -> float:
        """Compute ΔR'ᵢ₊₁ = √(Σⱼ(ρ'pred − R'obs)²/N) for a trial wedge.

        This is the first residual component of the optimisation
        (paper §5 eq. for ΔR').
        """
        sq_sum = 0.0
        n_ok = 0
        for _, row in echoes.iterrows():
            pol = str(row.get(self._m, "O"))
            xl = float(row[self._xl])
            yl = float(row[self._yl])
            r_obs = float(row[self._h])
            r_pred, _, _ = self._trace_ray(
                xl,
                yl,
                r_obs,
                f_mhz,
                pol,
                wedges,
                h_upper,
                nx,
                ny,
                h_lo_base,
                nx_lo_base,
                ny_lo_base,
            )
            if not np.isnan(r_pred):
                sq_sum += (r_pred - r_obs) ** 2
                n_ok += 1
        if n_ok == 0:
            return 1e6
        return float(np.sqrt(sq_sum / n_ok))

    def _return_point_dist(
        self,
        nx: float,
        ny: float,
        mean_theta: float,
        mean_phi: float,
        h_upper: float,
        f_mhz: float,
        pol: str,
        wedges: List[WedgePlane],
        h_lo_base: float,
        nx_lo_base: float,
        ny_lo_base: float,
    ) -> float:
        """Distance of the ground-return point from the sounder origin.

        A ray launched at the wedge's mean arrival angles (Θ, Φ) should
        return to (0, 0) if the tilt is correct (paper §5 para 33,
        second minimisation condition).
        """
        # Synthesise echolocation for the average-direction ray
        r_approx = h_upper / max(np.cos(mean_theta), 0.05)
        xl_approx = r_approx * np.sin(mean_theta) * np.sin(mean_phi)
        yl_approx = r_approx * np.sin(mean_theta) * np.cos(mean_phi)
        _, x_t, y_t = self._trace_ray(
            xl_approx,
            yl_approx,
            r_approx,
            f_mhz,
            pol,
            wedges,
            h_upper,
            nx,
            ny,
            h_lo_base,
            nx_lo_base,
            ny_lo_base,
        )
        if np.isnan(x_t):
            return 1e6
        return float(np.sqrt(x_t**2 + y_t**2))

    # ------------------------------------------------------------------
    # Wedge solvers
    # ------------------------------------------------------------------

    def _solve_wedge_lite(
        self,
        echoes: pd.DataFrame,
        wedges: List[WedgePlane],
        fp_lo: float,
        fp_hi: float,
        fixed_nx: float,
        fixed_ny: float,
        h_lo_base: float,
        nx_lo_base: float,
        ny_lo_base: float,
    ) -> WedgePlane:
        """NeXtYZ Lite: optimise h_upper only (tilts held fixed).

        Fast scalar minimisation of the group-range residual ΔR'
        (paper §9).
        """
        f_mhz = (fp_lo + fp_hi) / 2.0
        h_init = (
            float(echoes[self._h].median())
            if len(echoes) > 0
            else (wedges[-1].h_upper_km + 20.0 if wedges else 100.0)
        )
        h_min = (wedges[-1].h_upper_km + 0.5) if wedges else _H_BASE_KM
        h_max = h_min + 300.0

        def obj(h):
            return self._group_range_residual(
                h,
                fixed_nx,
                fixed_ny,
                echoes,
                f_mhz,
                wedges,
                h_lo_base,
                nx_lo_base,
                ny_lo_base,
            )

        result = minimize_scalar(
            obj,
            bounds=(h_min, h_max),
            method="bounded",
            options={"xatol": 0.5, "maxiter": 30},
        )
        h_opt = float(result.x)
        res = float(result.fun)
        logger.debug(
            f"Lite [{fp_lo:.2f}–{fp_hi:.2f} MHz]  "
            f"h={h_opt:.1f} km  ΔR'={res:.2f} km  n={len(echoes)}"
        )
        return WedgePlane(fp_lo, fp_hi, h_opt, fixed_nx, fixed_ny, res, len(echoes))

    def _solve_wedge_full(
        self,
        echoes: pd.DataFrame,
        wedges: List[WedgePlane],
        fp_lo: float,
        fp_hi: float,
        h_lo_base: float,
        nx_lo_base: float,
        ny_lo_base: float,
        n_alternate: int = 4,
    ) -> WedgePlane:
        """NeXtYZ Full: optimise (h, nₓ, nᵧ) per wedge with alternating minimisation.

        Paper §5 para 34: the two residual components are minimised
        **alternately** at successive steps — not combined.
        """
        f_mhz = (fp_lo + fp_hi) / 2.0
        h_min = (wedges[-1].h_upper_km + 0.5) if wedges else _H_BASE_KM

        # Initial guesses
        nx_cur = wedges[-1].nx if wedges else 0.0
        ny_cur = wedges[-1].ny if wedges else 0.0
        h_cur = float(echoes[self._h].median()) if len(echoes) > 0 else h_min + 20.0

        # Mean angles of arrival for the return-point constraint.
        # Dynasonde XL = R'·sinθ·sinφ, YL = R'·sinθ·cosφ, so
        # sinθ = sqrt(xl²+yl²)/R'.  Use arcsin, not arctan2(horiz, R')
        # which gives arctan(sinθ) ≠ θ.
        horiz = np.sqrt(echoes[self._xl] ** 2 + echoes[self._yl] ** 2)
        sin_theta = np.clip(
            horiz / echoes[self._h].replace(0, np.nan).fillna(1e-6), 0.0, 1.0
        )
        mean_theta = float(np.arcsin(sin_theta).mean())
        mean_phi = float(np.arctan2(echoes[self._xl], echoes[self._yl]).mean())

        for _ in range(n_alternate):
            # ── Step A: optimise h (group-range residual) ───────────────
            def obj_h(h):
                return self._group_range_residual(
                    h,
                    nx_cur,
                    ny_cur,
                    echoes,
                    f_mhz,
                    wedges,
                    h_lo_base,
                    nx_lo_base,
                    ny_lo_base,
                )

            res_h = minimize_scalar(
                obj_h,
                bounds=(h_min, h_min + 300.0),
                method="bounded",
                options={"xatol": 0.5, "maxiter": 20},
            )
            h_cur = float(res_h.x)

            # ── Step B: optimise (nₓ, nᵧ) (return-point distance) ──────
            def obj_tilt(params):
                nx_, ny_ = params
                if nx_**2 + ny_**2 >= 1.0:
                    return 1e6
                return self._return_point_dist(
                    nx_,
                    ny_,
                    mean_theta,
                    mean_phi,
                    h_cur,
                    f_mhz,
                    "O",
                    wedges,
                    h_lo_base,
                    nx_lo_base,
                    ny_lo_base,
                )

            res_t = minimize(
                obj_tilt,
                x0=[nx_cur, ny_cur],
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 0.1, "maxiter": 60},
            )
            nx_cur, ny_cur = float(res_t.x[0]), float(res_t.x[1])

            # Enforce valid unit normal
            mag = np.sqrt(nx_cur**2 + ny_cur**2)
            if mag >= 1.0:
                nx_cur /= mag * 1.05
                ny_cur /= mag * 1.05

        res_final = self._group_range_residual(
            h_cur,
            nx_cur,
            ny_cur,
            echoes,
            f_mhz,
            wedges,
            h_lo_base,
            nx_lo_base,
            ny_lo_base,
        )
        logger.debug(
            f"Full [{fp_lo:.2f}–{fp_hi:.2f} MHz]  "
            f"h={h_cur:.1f} km  nx={nx_cur:.4f}  ny={ny_cur:.4f}  "
            f"ΔR'={res_final:.2f} km  n={len(echoes)}"
        )
        return WedgePlane(fp_lo, fp_hi, h_cur, nx_cur, ny_cur, res_final, len(echoes))

    # ------------------------------------------------------------------
    # Error estimate  (paper §7)
    # ------------------------------------------------------------------

    @staticmethod
    def _height_error(
        residual_km: float, h_true_km: float, mean_R_prime_km: float
    ) -> float:
        """Convert ΔR'ᵢ₊₁ to a real-height error estimate Δhᵢ₊₁.

        From paper §7::

            Δhᵢ₊₁ = ΔR'ᵢ₊₁ × (hᵢ₊₁ − 80 km) / (⟨R'ⱼ⟩ᵢ₊₁ − 80 km)
        """
        denom = mean_R_prime_km - _H_BASE_KM
        if denom < 1e-3:
            return residual_km
        return float(residual_km * (h_true_km - _H_BASE_KM) / denom)

    # ------------------------------------------------------------------
    # Tilt estimate for Lite mode (from mean arrival angles)
    # ------------------------------------------------------------------

    @staticmethod
    def _tilt_from_echoes(
        echoes: pd.DataFrame,
        xl_col: str,
        yl_col: str,
        h_col: str,
        scale: float = 0.1,
    ) -> Tuple[float, float]:
        """Estimate constant (nx, ny) for a layer from mean arrival angles.

        Uses the approximation that the tilt normal points opposite to the
        mean horizontal direction of arrival, scaled by ``scale`` to keep
        the resulting tilt small (appropriate for NeXtYZ Lite).

        Returns
        -------
        (nx, ny)
        """
        # Correct zenith angle from Dynasonde XL/YL: sinθ = sqrt(xl²+yl²)/R'.
        # arctan2(horiz, R') gives arctan(sinθ) ≠ θ for non-small angles.
        horiz = np.sqrt(echoes[xl_col] ** 2 + echoes[yl_col] ** 2)
        sin_theta = np.clip(
            horiz / echoes[h_col].replace(0, np.nan).fillna(1e-6), 0.0, 1.0
        )
        theta = np.arcsin(sin_theta)
        phi = np.arctan2(echoes[xl_col], echoes[yl_col])
        mt = float(theta.mean())
        mp = float(phi.mean())
        nx = float(-np.sin(mt) * np.sin(mp)) * scale
        ny = float(-np.sin(mt) * np.cos(mp)) * scale
        mag = np.sqrt(nx**2 + ny**2)
        if mag >= 1.0:
            nx /= mag * 1.05
            ny /= mag * 1.05
        return nx, ny

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> NeXtYZResult:
        """Run the NeXtYZ inversion.

        Parameters
        ----------
        df:
            Echo DataFrame — must contain ``xl_km``, ``yl_km``,
            ``height_km`` (group range R'), ``frequency_khz``.
            Optionally: ``mode`` (O/X) and ``amplitude_db``.

        Returns
        -------
        NeXtYZResult
        """
        # Validate required columns
        for col in [self._xl, self._yl, self._h, self._f]:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in DataFrame.")

        if self._m not in df.columns:
            df = df.copy()
            df[self._m] = "O"

        # Frequency bounds
        fp_lo = max(self.fp_start, float(df[self._f].min()) / 1e3)
        fp_hi = float(df[self._f].max()) / 1e3

        # ── Lite: estimate constant tilts per broad layer ───────────────
        nx_E = ny_E = nx_F = ny_F = 0.0
        if self.mode == "Lite":
            e_mask = df[self._h].between(90.0, 160.0)
            if e_mask.sum() >= self.min_echo:
                nx_E, ny_E = self._tilt_from_echoes(
                    df[e_mask], self._xl, self._yl, self._h
                )
            f_mask = df[self._h] > 160.0
            if f_mask.sum() >= self.min_echo:
                nx_F, ny_F = self._tilt_from_echoes(
                    df[f_mask], self._xl, self._yl, self._h
                )

        # ── WSI stack base ──────────────────────────────────────────────
        h_lo_base = _H_BASE_KM
        nx_lo_base = 0.0
        ny_lo_base = 0.0

        # ── Bottom-up loop over fp wedges ───────────────────────────────
        wedges: List[WedgePlane] = []
        fp_cur = fp_lo

        n_planned = int((fp_hi - fp_lo) / self.fp_step)
        logger.info(
            f"NeXtYZ {self.mode}: {fp_lo:.2f}–{fp_hi:.2f} MHz  "
            f"~{n_planned} wedges  fH={self.fH_mhz:.2f} MHz"
        )

        while fp_cur + self.fp_step <= fp_hi + 1e-6:
            fp_next = round(fp_cur + self.fp_step, 6)

            echoes = self._select_echoes(df, fp_cur, fp_next)

            # Adaptive step widening when too few echoes (paper §5 para 20)
            if len(echoes) < self.min_echo:
                fp_next = round(fp_cur + 2.0 * self.fp_step, 6)
                echoes = self._select_echoes(df, fp_cur, fp_next)
                if len(echoes) < self.min_echo:
                    logger.debug(
                        f"Skip wedge {fp_cur:.2f}–{fp_next:.2f} MHz "
                        f"({len(echoes)} echoes < {self.min_echo})"
                    )
                    fp_cur = fp_next
                    continue

            # Assign tilt for Lite
            mean_h = float(echoes[self._h].mean())
            if self.mode == "Lite":
                fixed_nx = nx_E if mean_h < 160.0 else nx_F
                fixed_ny = ny_E if mean_h < 160.0 else ny_F
                wedge = self._solve_wedge_lite(
                    echoes,
                    wedges,
                    fp_cur,
                    fp_next,
                    fixed_nx,
                    fixed_ny,
                    h_lo_base,
                    nx_lo_base,
                    ny_lo_base,
                )
            else:
                wedge = self._solve_wedge_full(
                    echoes,
                    wedges,
                    fp_cur,
                    fp_next,
                    h_lo_base,
                    nx_lo_base,
                    ny_lo_base,
                )

            wedges.append(wedge)
            fp_cur = fp_next

        if not wedges:
            logger.warning(
                "NeXtYZ: no wedges solved. "
                "Check column names and that xl_km/yl_km are present."
            )
            return NeXtYZResult(wedges=[], method=self.mode)

        # ── Assemble output arrays ──────────────────────────────────────
        fp_arr = np.array([w.fp_hi_mhz for w in wedges])
        h_arr = np.array([w.h_upper_km for w in wedges])
        nx_arr = np.array([w.nx for w in wedges])
        ny_arr = np.array([w.ny for w in wedges])
        res_arr = np.array([w.residual_km for w in wedges])

        # Height errors — paper §7
        h_err = np.zeros(len(wedges))
        for k, w in enumerate(wedges):
            band = self._select_echoes(df, w.fp_lo_mhz, w.fp_hi_mhz)
            mean_Rp = float(band[self._h].mean()) if len(band) > 0 else h_arr[k]
            h_err[k] = self._height_error(res_arr[k], h_arr[k], mean_Rp)

        # Tilt angles in degrees.
        # Frame-plane normal is (nx, ny, nz) with nz = sqrt(1−nx²−ny²).
        # Tilt from vertical = arctan(n_horiz / nz), not arctan(n_horiz).
        nz_arr = np.sqrt(np.maximum(1.0 - nx_arr**2 - ny_arr**2, 0.0))
        tilt_mer = np.degrees(np.arctan2(nx_arr, nz_arr))  # meridional (Θₓ)
        tilt_zon = np.degrees(np.arctan2(ny_arr, nz_arr))  # zonal      (Θᵧ)

        converged = [float(w.residual_km) < 10.0 for w in wedges]

        logger.info(
            f"NeXtYZ {self.mode}: {len(wedges)} wedges solved  "
            f"mean ΔR'={np.nanmean(res_arr):.1f} km  "
            f"mean Δh={np.nanmean(h_err):.1f} km  "
            f"converged={sum(converged)}/{len(wedges)}"
        )

        return NeXtYZResult(
            wedges=wedges,
            method=self.mode,
            fp_profile_mhz=fp_arr,
            h_true_km=h_arr,
            h_errors_km=h_err,
            tilt_meridional_deg=tilt_mer,
            tilt_zonal_deg=tilt_zon,
            converged=converged,
        )
