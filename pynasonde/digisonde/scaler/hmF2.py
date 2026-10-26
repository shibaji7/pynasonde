"""hmF2 and M(3000)F2 helper methods."""

from __future__ import annotations

import numpy as np
from loguru import logger


class ScaleHmF2:
    """
    A unified suite to process raw ionogram data points.
    It can calculate M(3000)F2 from trace arrays and estimate
    the true peak height (hmF2) using various E-layer strategies.
    """

    @staticmethod
    def calculate_m3000f2(
        frequencies: list[float] | np.ndarray,
        virtual_heights: list[float] | np.ndarray,
        fof2: float,
    ) -> float:
        """
        Calculates M(3000)F2 using Paul's analytical method based on
        the geometry of a single-hop 3000 km radio transmission path.

        Args:
            frequencies: list/array of trace frequency probe points (MHz)
            virtual_heights: list/array of trace virtual heights h' (km)
            fof2: F2 layer critical frequency (MHz)
        """
        if len(frequencies) != len(virtual_heights) or len(frequencies) == 0:
            raise ValueError(
                "Frequencies and virtual heights arrays must match in size and not be empty."
            )
        if fof2 <= 0:
            raise ValueError("foF2 must be a positive value.")

        d_half = 1500.0  # Half of 3000 km path [km]
        m_factors = []

        for f, hp in zip(frequencies, virtual_heights):
            if hp <= 0:
                continue
            # Flat-Earth secant law: sec(I) = sqrt(1 + (d/2 / h')^2)
            # This is the standard formula used by URSI/CCIR ionosonde software.
            # Spherical-Earth corrections are <1% for h' > 200 km and d = 3000 km
            # and do not improve accuracy given the uncertainty in virtual heights.
            k_factor = np.sqrt(1.0 + (d_half / hp) ** 2)

            # MUF at this trace point for a 3000 km path = f × k_factor
            # M(3000)F2 at this point = MUF / foF2
            m_factors.append(k_factor * (f / fof2))

        if not m_factors:
            raise ValueError("No valid height points found to compute M(3000)F2.")

        # M(3000)F2 = max over all trace points (tangent-point approximation)
        return max(m_factors)

    @staticmethod
    def shimazaki(m3000f2: float) -> float:
        """Classic Shimazaki (1955) formula for hmF2."""
        if m3000f2 <= 0:
            raise ValueError("M(3000)F2 must be greater than 0.")
        return (1490.0 / m3000f2) - 176.0

    @staticmethod
    def bilitza(m3000f2: float, fof2: float, foe: float) -> float:
        """Standard Bilitza (1979) formula with E-layer underlying ionization."""
        if m3000f2 <= 0 or fof2 <= 0 or foe <= 0:
            raise ValueError("Frequencies and propagation factors must be positive.")

        # Singularity in delta_m is at r = 1.4; clamp just above it.
        # Using 1.7 as the lower bound (original code) was too conservative
        # and biased hmF2 low when 1.4 < foF2/foE < 1.7.
        r = max(1.41, fof2 / foe)
        delta_m = 0.18 / (r - 1.4) + 0.096 * (r - 1.4)
        delta_m = max(-0.1, min(delta_m, 0.4))

        return (1490.0 / (m3000f2 + delta_m)) - 176.0

    @classmethod
    def constant_ratio_fallback(cls, m3000f2: float, fof2: float) -> float:
        """Bilitza calculation using the standard daytime fallback ratio (3.3)."""
        effective_foe = fof2 / 3.3
        return cls.bilitza(m3000f2, fof2, effective_foe)

    @classmethod
    def zenith_estimate_fallback(
        cls, m3000f2: float, fof2: float, sza_degrees: float, r12: float
    ) -> float:
        """Bilitza calculation predicting foE via provided Solar Zenith Angle and R12."""
        sza_rad = np.radians(sza_degrees)
        cos_sza = np.cos(sza_rad)

        if cos_sza <= 0:  # Night context
            return cls.shimazaki(m3000f2)

        # Chapman foE formula (Bilitza 2001):
        #   foE = 3.31 × (1 + 0.008×R12)^0.25 × cos(χ)^0.25
        # Original code used 2.25 / 0.015 which underestimated foE by ~15-30%.
        predicted_foe = 3.31 * ((1.0 + 0.008 * r12) ** 0.25) * (cos_sza**0.25)
        return cls.bilitza(m3000f2, fof2, predicted_foe)

    # ------------------------------------------------------------------
    # Abel inversion
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_f_trace(
        f_F: np.ndarray,
        hp_F: np.ndarray,
        foF2: float,
        poly_deg: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Fit a robust polynomial to the F2 trace to suppress noisy points.

        Uses iteratively re-weighted least squares (Theil-Sen is unavailable in
        numpy, so we do one pass of median-residual outlier rejection before the
        final polyfit).  Degree-3 captures the typical accelerating rise of h'(f)
        toward foF2 without over-fitting sparse night traces.

        Returns (f_clean, hp_clean, poly_coeffs) where ``poly_coeffs`` can be
        evaluated with ``np.polyval`` for extrapolation below f_min_F.
        """
        if len(f_F) < poly_deg + 1:
            # Too few points — fall back to constant at minimum height
            c = (
                np.polyfit(f_F, hp_F, min(1, len(f_F) - 1))
                if len(f_F) >= 2
                else [0, hp_F[0] if len(hp_F) else 200.0]
            )
            return f_F, hp_F, c

        c = np.polyfit(f_F, hp_F, poly_deg)
        residuals = hp_F - np.polyval(c, f_F)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        thr = max(3.0 * mad, 20.0)  # at least ±20 km tolerance
        keep = np.abs(residuals) <= thr

        if keep.sum() >= poly_deg + 1:
            c = np.polyfit(f_F[keep], hp_F[keep], poly_deg)
            return f_F[keep], hp_F[keep], c
        return f_F, hp_F, c

    @staticmethod
    def _build_hp_profile(
        f_F: np.ndarray,
        hp_F: np.ndarray,
        foF2: float,
        f_E: np.ndarray | None,
        hp_E: np.ndarray | None,
        foE: float | None,
        hpE: float | None,
        sza_deg: float = 90.0,
        r12: float = 100.0,
        n_e_min: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble a continuous h'(f) profile from f=0 to just below foF2.

        The F2 trace is **polynomial-smoothed** before use so that individual
        noisy Viterbi points do not produce large swings in the Abel integral.
        The same polynomial is used to extrapolate below f_min_F (night, no E),
        replacing the unstable "bottom-20%-slope" method.

        Regions filled in priority order:

        1. **F trace** (f_min_F … foF2): smooth polynomial evaluated on grid.
        2. **E-F valley** (E-layer top … f_min_F): linear bridge.
        3. **E layer** (0 … E-layer top):
           - ≥ n_e_min E-trace points  →  smoothed E trace interpolation.
           - foE + hpE known, sparse   →  QP model h'(f) = hpE/√(1−(f/foE)²).
           - Day, no E detected        →  Chapman-predicted foE + QP model.
           - Night, no E              →  polynomial extrapolated from F trace,
                                         floored at 90 km.

        Returns (f_grid, hp_grid) on a 512-point uniform grid 0 … 0.999·foF2.
        """
        N_GRID = 512
        f_grid = np.linspace(0.0, 0.999 * foF2, N_GRID)

        # Sort F trace, keep only valid points below foF2
        idx = np.argsort(f_F)
        f_F = f_F[idx]
        hp_F = hp_F[idx]
        mask_F = (f_F > 0) & (f_F < foF2) & np.isfinite(hp_F)
        f_F, hp_F = f_F[mask_F], hp_F[mask_F]

        hp_grid = np.full(N_GRID, np.nan)
        if len(f_F) < 2:
            hp_grid[:] = 200.0
            return f_grid, hp_grid

        f_min_F = float(f_F[0])

        # ── Polynomial smooth of the F trace ──────────────────────────────────
        poly_deg = min(3, len(f_F) - 1)
        _, _, f_poly = ScaleHmF2._smooth_f_trace(f_F, hp_F, foF2, poly_deg)

        # Evaluate the smooth polynomial over the entire F-region on the grid
        in_F = f_grid >= f_min_F
        hp_smooth_F = np.polyval(f_poly, f_grid[in_F])
        # Enforce monotonically non-decreasing h'(f) (physical constraint)
        hp_smooth_F = np.maximum.accumulate(hp_smooth_F)
        hp_smooth_F = np.clip(hp_smooth_F, float(hp_F.min()), None)
        hp_grid[in_F] = hp_smooth_F

        # ── E-layer strategy selection ────────────────────────────────────────
        cos_sza = np.cos(np.radians(sza_deg))

        has_good_E = (
            f_E is not None
            and hp_E is not None
            and len(f_E) >= n_e_min
            and foE is not None
            and foE > 0
        )
        has_sparse_E = (
            not has_good_E
            and foE is not None
            and foE > 0
            and hpE is not None
            and hpE > 0
        )
        is_day = cos_sza > 0

        # ── Case 1: good E trace ──────────────────────────────────────────────
        if has_good_E:
            idx_e = np.argsort(f_E)
            f_E_s = f_E[idx_e]
            hp_E_s = hp_E[idx_e]
            m_e = (f_E_s > 0) & (f_E_s < foE) & np.isfinite(hp_E_s)
            f_E_s, hp_E_s = f_E_s[m_e], hp_E_s[m_e]

            if len(f_E_s) >= 2:
                # Polynomial-smooth the E trace too
                _, _, e_poly = ScaleHmF2._smooth_f_trace(
                    f_E_s, hp_E_s, foE, min(2, len(f_E_s) - 1)
                )
                f_top_E = float(f_E_s[-1])
                in_E = f_grid <= f_top_E
                hp_e = np.polyval(e_poly, f_grid[in_E])
                hp_e = np.clip(hp_e, 90.0, None)
                hp_grid[in_E] = hp_e

                # Valley: linear bridge
                in_valley = (f_grid > f_top_E) & (f_grid < f_min_F)
                if in_valley.any():
                    hp_grid[in_valley] = np.interp(
                        f_grid[in_valley],
                        [f_top_E, f_min_F],
                        [float(np.polyval(e_poly, f_top_E)), float(hp_smooth_F[0])],
                    )
            else:
                has_good_E = False

        # ── Case 2: sparse E or daytime-predicted E (QP model) ───────────────
        if not has_good_E and (has_sparse_E or is_day):
            if has_sparse_E:
                foe_use, hpe_use = float(foE), float(hpE)
            else:
                foe_use = 3.31 * ((1.0 + 0.008 * r12) ** 0.25) * (cos_sza**0.25)
                hpe_use = 110.0

            f_qp = f_grid[f_grid < foe_use * 0.98]
            if len(f_qp):
                denom = np.sqrt(np.maximum(1.0 - (f_qp / foe_use) ** 2, 1e-6))
                hp_qp = np.clip(hpe_use / denom, hpe_use, hpe_use * 5.0)
                hp_grid[f_grid < foe_use * 0.98] = hp_qp

            f_top_qp = foe_use * 0.98
            hp_qp_top = float(hpe_use / np.sqrt(max(1.0 - 0.98**2, 1e-6)))
            in_valley = (f_grid >= f_top_qp) & (f_grid < f_min_F)
            if in_valley.any():
                hp_grid[in_valley] = np.interp(
                    f_grid[in_valley],
                    [f_top_qp, f_min_F],
                    [hp_qp_top, float(hp_smooth_F[0])],
                )

        # ── Case 3: night, no E — polynomial extrapolation from F trace ───────
        # Use the same polynomial fitted to the F trace evaluated below f_min_F.
        # This is stable because it uses ALL trace points (not just bottom 20%).
        if not has_good_E and not has_sparse_E and not is_day:
            below = f_grid < f_min_F
            hp_extrap = np.polyval(f_poly, f_grid[below])
            # Clamp: extrapolation cannot exceed f_min_F height and must be ≥90 km
            hp_extrap = np.clip(hp_extrap, 90.0, float(hp_smooth_F[0]))
            hp_grid[below] = hp_extrap

        # ── Fill any residual NaN with nearest-neighbour ──────────────────────
        nan_mask = ~np.isfinite(hp_grid)
        if nan_mask.any() and (~nan_mask).any():
            hp_grid[: np.argmax(~nan_mask)] = hp_grid[np.argmax(~nan_mask)]
            nans = np.where(nan_mask)[0]
            valid_idx = np.where(~nan_mask)[0]
            hp_grid[nans] = np.interp(nans, valid_idx, hp_grid[valid_idx])

        hp_grid = np.clip(hp_grid, 90.0, None)
        return f_grid, hp_grid

    @classmethod
    def abel_inversion(
        cls,
        f_F: np.ndarray,
        hp_F: np.ndarray,
        foF2: float,
        f_E: np.ndarray | None = None,
        hp_E: np.ndarray | None = None,
        foE: float | None = None,
        hpE: float | None = None,
        sza_deg: float = 90.0,
        r12: float = 100.0,
        n_e_min: int = 5,
        n_quad: int = 64,
    ) -> float:
        """True height of F2 peak via numerical Abel inversion.

        Evaluates

            hmF2 = (2/π) ∫₀^{foF2}  h'(f) / √(foF2²−f²)  df

        after assembling a continuous h'(f) profile via ``_build_hp_profile``
        (E+F trace filling with tiered fallbacks — see that method's docstring).

        The singularity at f → foF2 is removed with the substitution
        f = foF2 · sin θ, giving

            hmF2 = (2/π) ∫₀^{π/2}  h'(foF2 · sin θ)  dθ

        which is integrated by Gauss-Legendre quadrature.

        Args:
            f_F: F2 Viterbi trace frequencies (MHz).
            hp_F: F2 Viterbi trace virtual heights (km).
            foF2: F2 critical frequency (MHz).
            f_E: E-layer Viterbi trace frequencies (MHz); None if absent.
            hp_E: E-layer Viterbi trace virtual heights (km); None if absent.
            foE: E-layer critical frequency scaled by ``IonoscaleResult``; None if absent.
            hpE: E-layer virtual height scaled by ``IonoscaleResult``; None if absent.
            sza_deg: Solar zenith angle [°] — controls night/day fallback branch.
            r12: 12-month smoothed sunspot number — used in Chapman foE model.
            n_e_min: Minimum E-trace points required for "good E trace" branch.
            n_quad: Gauss-Legendre quadrature order (64 is sufficient).

        Returns:
            hmF2 in km, or ``np.nan`` if insufficient data.
        """
        if foF2 <= 0 or len(f_F) < 3:
            return np.nan

        f_E_arr = np.asarray(f_E, dtype=float) if f_E is not None else None
        hp_E_arr = np.asarray(hp_E, dtype=float) if hp_E is not None else None

        try:
            f_grid, hp_grid = cls._build_hp_profile(
                np.asarray(f_F, dtype=float),
                np.asarray(hp_F, dtype=float),
                foF2,
                f_E_arr,
                hp_E_arr,
                foE,
                hpE,
                sza_deg,
                r12,
                n_e_min,
            )
        except Exception as exc:
            logger.warning(f"abel_inversion: _build_hp_profile failed — {exc}")
            return np.nan

        from scipy.interpolate import interp1d

        hp_interp = interp1d(
            f_grid,
            hp_grid,
            kind="linear",
            bounds_error=False,
            fill_value=(hp_grid[0], hp_grid[-1]),
        )

        # Gauss-Legendre quadrature on θ ∈ [0, π/2]
        # f = foF2·sin θ  →  singularity-free integrand = h'(foF2·sinθ)
        from numpy.polynomial.legendre import leggauss

        xi, wi = leggauss(n_quad)
        theta = (xi + 1.0) * (np.pi / 4.0)  # map [-1,1] → [0, π/2]
        w = wi * (np.pi / 4.0)
        f_samp = foF2 * np.sin(theta)

        hmF2 = float((2.0 / np.pi) * np.dot(w, hp_interp(f_samp)))
        return max(hmF2, 90.0)
