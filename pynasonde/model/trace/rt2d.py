# hf_raytrace_2d.py
# ---------------------------------------------------------------------
# 2-D HF ray tracing in an inhomogeneous ionosphere n(x,y) from Ne(x,y)
# No magnetic field, no collisions: n^2 = 1 - (f_p/f)^2
# Uses Hamiltonian ray equations in 2D with a simple RK4 integrator.
# Units:
#   x, y in km; Ne in m^-3; frequency f in Hz; c in km/s internally.
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace

import numpy as np
from loguru import logger
from scipy import constants

from pynasonde.model.trace.ionosphere import IRI, IonosphereModels
from pynasonde.model.trace.plottrace import PlotRays


# ======================== Helper: Interpolator ========================
class Bilinear2D:
    """
    Bilinear interpolator for a rectilinear (x,y) grid.
    Grid:
      x [Nx] (km, ascending), y [Ny] (km, ascending)
      F [Ny, Nx] values (e.g., Ne in m^-3) at (x[i], y[j]) => F[j, i]
    """

    def __init__(self, x_km: np.ndarray, y_km: np.ndarray, F: np.ndarray):
        x = np.asarray(x_km, float)
        y = np.asarray(y_km, float)
        F = np.asarray(F, float)
        assert F.shape == (y.size, x.size), "F must be [Ny, Nx]"
        assert np.all(np.diff(x) > 0) and np.all(
            np.diff(y) > 0
        ), "x,y must be strictly increasing"
        self.x = x
        self.y = y
        self.F = F
        self.dx = np.diff(x).mean()
        self.dy = np.diff(y).mean()

    def inside(self, x: float, y: float) -> bool:
        return (self.x[0] <= x <= self.x[-1]) and (self.y[0] <= y <= self.y[-1])

    def __call__(self, xq: float, yq: float) -> float:
        """Interpolate value at (xq,yq). Out of bounds returns 0."""
        if not self.inside(xq, yq):
            return 0.0
        # find i such that x[i] <= xq <= x[i+1]
        i = np.searchsorted(self.x, xq) - 1
        j = np.searchsorted(self.y, yq) - 1
        i = np.clip(i, 0, self.x.size - 2)
        j = np.clip(j, 0, self.y.size - 2)
        x0, x1 = self.x[i], self.x[i + 1]
        y0, y1 = self.y[j], self.y[j + 1]
        tx = (xq - x0) / (x1 - x0)
        ty = (yq - y0) / (y1 - y0)
        f00 = self.F[j, i]
        f10 = self.F[j, i + 1]
        f01 = self.F[j + 1, i]
        f11 = self.F[j + 1, i + 1]
        return (
            (1 - tx) * (1 - ty) * f00
            + tx * (1 - ty) * f10
            + (1 - tx) * ty * f01
            + tx * ty * f11
        )

    def grad_central(
        self, xq: float, yq: float, h: float | None = None
    ) -> tuple[float, float]:
        """
        Central-difference gradient of F at (xq,yq) using bilinear values.
        h: step (km); default uses grid-spacing average.
        Returns (dF/dx, dF/dy) with F units per km.
        """
        if h is None:
            h = 0.5 * (self.dx + self.dy)
        fxp = self(xq + h, yq)
        fxm = self(xq - h, yq)
        fyp = self(xq, yq + h)
        fym = self(xq, yq - h)
        dfdx = (fxp - fxm) / (2 * h)
        dfdy = (fyp - fym) / (2 * h)
        return dfdx, dfdy


# ======================== Plasma / refractive index ====================
def plasma_freq_hz(ne_m3: float | np.ndarray) -> np.ndarray:
    """f_p [Hz] from electron density ne [m^-3]."""
    omega_p = np.sqrt(
        ne_m3 * constants.e * constants.e / (constants.epsilon_0 * constants.m_e)
    )
    return omega_p / (2 * np.pi)


def n2_from_ne(
    f_hz: float, ne_m3: float | np.ndarray, n2_floor: float = 1e-8
) -> np.ndarray:
    """n^2 = 1 - (f_p/f)^2 with a small positive floor to avoid NaNs."""
    fp = plasma_freq_hz(ne_m3)
    n2 = 1.0 - (fp / f_hz) ** 2
    return np.maximum(n2, n2_floor)


def n_and_grad(f_hz: float, ne_interp: Bilinear2D, x: float, y: float):
    """Return n(x,y) and ∇n using finite-diff on n^2 to be robust near cutoff."""
    ne = ne_interp(x, y)
    n2 = n2_from_ne(f_hz, ne)
    n = np.sqrt(n2)

    # step for finite-diff (km)
    h = 0.5 * (ne_interp.dx + ne_interp.dy)
    n2_xp = n2_from_ne(f_hz, ne_interp(x + h, y))
    n2_xm = n2_from_ne(f_hz, ne_interp(x - h, y))
    n2_yp = n2_from_ne(f_hz, ne_interp(x, y + h))
    n2_ym = n2_from_ne(f_hz, ne_interp(x, y - h))

    # ∂n/∂x = (1/(2n)) ∂(n^2)/∂x  (and same for y). Guard n>0.
    if n > 0:
        dn_dx = 0.5 * (n2_xp - n2_xm) / (2 * h) / n
        dn_dy = 0.5 * (n2_yp - n2_ym) / (2 * h) / n
    else:
        dn_dx = dn_dy = 0.0
    return float(n), float(dn_dx), float(dn_dy)


@dataclass
class RayConfig:
    f_MHz: float
    el0_deg: float
    x0_km: float = 0.0
    y0_km: float = 0.0
    s_max_km: float = 4000.0  # TOTAL path length to integrate (km)
    ds_km: float = 0.5  # step in km along the ray
    y_ground_km: float = 0.0
    y_max_km: float = 1200.0
    x_max_km: float = 6000.0
    n2_floor: float = 1e-2  # treat n^2 below this as evanescent
    keep_every: int = 1


class RayTracer2D:
    def __init__(self, x_km, y_km, Ne_m3):
        self.ne = Bilinear2D(x_km, y_km, Ne_m3)

    @staticmethod
    def _rk4_step_arc(f_hz, ne_interp, r, T, ds):
        """One RK4 step for arc-length system: dr/ds=T; dT/ds=(∇n - (∇n·T)T)/n."""

        def rhs(r, T):
            x, y = r
            n, dn_dx, dn_dy = n_and_grad(f_hz, ne_interp, x, y)
            # stop forcing if near cutoff to avoid singular accel
            if n <= 1e-6:
                a = np.array([0.0, 0.0])
            else:
                grad = np.array([dn_dx, dn_dy])
                a = (grad - (grad @ T) * T) / n  # curvature vector
            return T, a

        # k1
        k1_r, k1_T = rhs(r, T)
        # k2
        k2_r, k2_T = rhs(r + 0.5 * ds * k1_r, T + 0.5 * ds * k1_T)
        # k3
        k3_r, k3_T = rhs(r + 0.5 * ds * k2_r, T + 0.5 * ds * k2_T)
        # k4
        k4_r, k4_T = rhs(r + ds * k3_r, T + ds * k3_T)

        r_new = r + (ds / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        T_new = T + (ds / 6.0) * (k1_T + 2 * k2_T + 2 * k3_T + k4_T)
        # renormalize T to unit length (controls drift)
        nrm = np.hypot(T_new[0], T_new[1])
        if nrm > 0:
            T_new /= nrm
        return r_new, T_new

    def trace(self, cfg: RayConfig):
        f_hz = cfg.f_MHz * 1e6

        # initial position & unit tangent from elevation (CW-from-north not needed in 2D)
        r = np.array([cfg.x0_km, cfg.y0_km], float)
        el = np.deg2rad(cfg.el0_deg)
        T = np.array([np.cos(el), np.sin(el)], float)  # (x,y) components; |T|=1

        s_vals, xs, ys, ns = [], [], [], []
        reason = "max_s_reached"
        steps = int(np.ceil(cfg.s_max_km / cfg.ds_km))
        last_above = True

        for i in range(steps):
            if i % cfg.keep_every == 0:
                n_here, *_ = n_and_grad(f_hz, self.ne, r[0], r[1])
                s_vals.append(i * cfg.ds_km)
                xs.append(r[0])
                ys.append(r[1])
                ns.append(n_here)

            # termination checks BEFORE step
            if not self.ne.inside(r[0], r[1]):
                reason = "out_of_bounds"
                break
            if abs(r[0]) > cfg.x_max_km or r[1] > cfg.y_max_km:
                reason = "domain_limit"
                break
            n2_here = n2_from_ne(f_hz, self.ne(r[0], r[1]))
            if n2_here < cfg.n2_floor:
                reason = "evanescent"
                break
            if i > 0 and last_above and (r[1] <= cfg.y_ground_km):
                reason = "ground_hit"
                break
            last_above = r[1] > cfg.y_ground_km - 1e-6

            # advance one arc-length step
            r, T = self._rk4_step_arc(f_hz, self.ne, r, T, cfg.ds_km)

        # Double check for theta near start point
        if np.round(ys[-1], 1) == 0.0:
            reason = "ground_hit"
        return {
            "s_km": np.asarray(s_vals),
            "x_km": np.asarray(xs),
            "y_km": np.asarray(ys),
            "n": np.asarray(ns),
            "f_MHz": cfg.f_MHz,
            "el0_deg": cfg.el0_deg,
            "reason": reason,
        }

    def run_single_ray(self, args: tuple):
        (
            f_MHz,
            el0_deg,
            x0_km,
            y0_km,
            s_max_km,
            ds_km,
            y_max_km,
            x_max_km,
            keep_every,
        ) = args
        logger.info(f"Running simulations: f {f_MHz} e {el0_deg}")
        cfg = RayConfig(
            f_MHz=f_MHz,
            el0_deg=el0_deg,
            x0_km=x0_km,
            y0_km=y0_km,
            s_max_km=s_max_km,
            ds_km=ds_km,
            y_max_km=y_max_km,
            x_max_km=x_max_km,
            keep_every=keep_every,
        )
        out = self.trace(cfg)
        logger.warning(f"Termination: {out['reason']}")
        logger.info(
            f"Max height (km): {out['y_km'].max() if out['y_km'].size else None}"
        )
        logger.info(
            f"Ground range (km): {np.abs(out['x_km']).max() if out['x_km'].size else None}"
        )
        return SimpleNamespace(**out)

    def run_all_rays(
        self,
        frequencies,
        el_angles,
        x0_km,
        y0_km,
        s_max_km,
        ds_km,
        y_max_km,
        x_max_km,
        keep_every,
    ):
        tasks = [
            (f, el, x0_km, y0_km, s_max_km, ds_km, y_max_km, x_max_km, keep_every)
            for f in frequencies
            for el in el_angles
        ]
        # use all CPUs, or specify processes=N
        with Pool(processes=cpu_count() - 3) as pool:
            self.outputs = pool.map(self.run_single_ray, tasks)
        return self.outputs

    def homing_in_to_locate_roots(
        self,
        outputs=[],
        ground_height_precision: int = 2,
        homing_error_km: float = 10,
    ):
        self.homing_roots = []
        outputs = outputs if len(outputs) > 0 else self.outputs
        for o in outputs:
            homing = (
                (o.reason == "ground_hit")
                and (np.round(o.y_km[-1], ground_height_precision) == 0.0)
                and (np.abs(np.abs(o.x_km[-1]) - o.x_km[0]) <= homing_error_km)
            )
            if homing:
                self.homing_roots.append(o.el0_deg)
        return self.homing_roots

    def plot_fan(
        self,
        X,
        Z,
        Ne,
        outputs=[],
        homing_roots=[],
        figure_file_name=None,
        kind="pf",
        close=False,
        text=None,
    ):
        outputs = outputs if len(outputs) > 0 else self.outputs
        homing_roots = homing_roots if len(homing_roots) > 0 else self.homing_roots
        rp = PlotRays()
        rp.set_density(X, Z, Ne, plasma_freq_hz(Ne) / 1e6)
        rp.lay_rays(
            outputs,
            kind=kind,
            ped_angles=homing_roots,
            text=text,
        )
        if figure_file_name:
            rp.save(figure_file_name)
        if close:
            rp.close()
            return 0
        else:
            return rp


# ========================== Examples =======================
# Build a demo Ne(x,y) field: background + Chapman bump centered at x
def ray_trace_2d_ionosphereic_bump(
    x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
    hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
    NmF2: float = 1e12,  # peak density [m^-3]
    hmF2: float = 300.0,  # F2 peak height [km]
    nmf2_funct=lambda dx: (1.0 + 0.15 * np.exp(-(((dx + 30) / 100) ** 2))),
    hmf2_funct=lambda dx: (30.0 * np.exp(-(((dx + 30) / 100) ** 2))),
    H_scale: float = 50.0,  # scale height [km]
    Ne_floor: float = 2e10,
    frequencies: np.ndarray = np.asarray([8]),
    el_angles: np.ndarray = np.arange(50, 130, 5),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,  # allow enough total path
    ds_km=0.05,  # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
):
    X, Z, Ne = IonosphereModels.create_chapman_ionosphere_bump(
        x, hs, NmF2, hmF2, nmf2_funct, hmf2_funct, H_scale, Ne_floor
    )
    rt = RayTracer2D(x, hs, Ne)
    outputs = rt.run_all_rays(
        frequencies,
        el_angles,
        x0_km,
        y0_km,
        s_max_km,
        ds_km,
        y_max_km,
        x_max_km,
        keep_every,
    )
    rp = rt.plot_fan(X, Z, Ne, figure_file_name=figure_file_name)
    return X, Z, Ne, outputs


# Build a demo Ne(x,y) field: background + Chapman tilt centered
def ray_trace_2d_ionosphereic_tilt(
    x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
    hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
    NmF2: float = 1e12,  # peak density [m^-3]
    hmF2: float = 300.0,  # F2 peak height [km]
    H_scale: float = 50.0,  # scale height [km]
    Ne_floor: float = 2e10,
    hmf2_tilt_funct=lambda dx: (-0.1 * dx),
    frequencies: np.ndarray = np.asarray([8]),
    el_angles: np.ndarray = np.arange(50, 130, 5),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,  # allow enough total path
    ds_km=0.01,  # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
):
    X, Z, Ne = IonosphereModels.chapman_with_tilted_hmf2(
        x, hs, NmF2, hmF2, H_scale, Ne_floor, hmf2_tilt_funct
    )
    rt = RayTracer2D(x, hs, Ne)
    outputs = rt.run_all_rays(
        frequencies,
        el_angles,
        x0_km,
        y0_km,
        s_max_km,
        ds_km,
        y_max_km,
        x_max_km,
        keep_every,
    )
    rp = rt.plot_fan(X, Z, Ne, figure_file_name=figure_file_name)
    return X, Z, Ne, outputs


# Build a demo Ne(x,y) field: Chapman background + Wave front
def ray_trace_2d_ionosphereic_wave_front(
    x: np.ndarray = np.linspace(-500, 500, 2001),  # horizontal distance [km]
    hs: np.ndarray = np.linspace(0, 1000, 2001),  # altitude [km]
    layer_names: np.ndarray = np.asarray(["E", "F1", "F2"]),
    layer_heights: np.ndarray = np.asarray([110.0, 180.0, 300.0]),
    layer_base_ne: np.ndarray = np.asarray([1e11, 4.0e11, 11.0e11]),
    layer_scales: np.ndarray = np.asarray([10.0, 25.0, 50.0]),
    Ne_floor: float = 2e10,
    x_params: np.ndarray = np.asarray([-62, 93, 127]),
    d_params: np.ndarray = np.asarray([0.4, 0.15]),
    frequencies: np.ndarray = np.asarray([8.3]),
    el_angles: np.ndarray = np.arange(50, 110, 0.5),
    homing_roots: np.ndarray = np.asarray([]),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,  # allow enough total path
    ds_km=0.05,  # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
    ground_location_precision=6,
):
    el_angles = np.concatenate((el_angles, homing_roots))
    X, Z, Ne, alpha_X, Nex = IonosphereModels.cusp_function_alpha(
        x,
        hs,
        layer_names,
        layer_heights,
        layer_base_ne,
        layer_scales,
        Ne_floor,
        x_params,
        d_params,
    )
    rt = RayTracer2D(x, hs, Nex)
    outputs = rt.run_all_rays(
        frequencies,
        el_angles,
        x0_km,
        y0_km,
        s_max_km,
        ds_km,
        y_max_km,
        x_max_km,
        keep_every,
    )
    homing_roots = rt.homing_in_to_locate_roots(
        outputs, ground_location_precision=ground_location_precision
    )
    rp = rt.plot_fan(X, Z, Nex, figure_file_name=figure_file_name)
    return X, Z, Ne, outputs


class TIDModel:
    def __init__(
        self,
        lambda_h,
        period,
        I_deg,
        az_wave_deg=0.0,
        az_Bh_deg=0.0,
        Ub_ref=15.0,
        z0_ref=250e3,
        kzi_override=None,
        gamma=1.4,
        g=9.80665,
    ):
        """
        Initialize a gravity-wave–driven TID model (Hooke 1968 form).

        Parameters
        ----------
        lambda_h : float
            Horizontal wavelength (m).
        period : float
            Wave period (s).
        I_deg : float
            Geomagnetic inclination (deg).
        az_wave_deg : float
            Azimuth of wave propagation (deg; 0°=east, 90°=north).
        az_Bh_deg : float
            Azimuth of horizontal projection of B-field (deg).
        Ub_ref : float
            Neutral velocity along B at z0_ref (m/s).
        z0_ref : float
            Reference altitude for Ub_ref (m).
        kzi_override : float or None
            Optional override for dissipation term k_zi (1/m).
        gamma : float
            Ratio of specific heats for dry air.
        g : float
            Gravitational acceleration (m/s^2).
        """
        self.lambda_h = lambda_h
        self.period = period
        self.I_deg = I_deg
        self.az_wave_deg = az_wave_deg
        self.az_Bh_deg = az_Bh_deg
        self.Ub_ref = Ub_ref
        self.z0_ref = z0_ref
        self.kzi_override = kzi_override
        self.g = g
        self.gamma = gamma
        return

    @staticmethod
    def brunt_vaisala_and_acoustic(H, gamma=1.4, g=9.80665):
        """Return Brunt–Väisälä frequency, sound speed, acoustic cutoff."""
        N_B = (gamma - 1) * g / (gamma * H)
        c = np.sqrt(gamma * g * H)
        omega_a = c / (2.0 * H)
        return N_B, c, omega_a

    @staticmethod
    def vertical_wavenumber(
        kh, omega, H, omega_a=None, N_B=None, c=None, gamma=1.4, g=9.80665
    ):
        """Dispersion relation for vertical wavenumber."""
        if N_B is None or c is None or omega_a is None:
            N_B, c, omega_a = TIDModel.brunt_vaisala_and_acoustic(H, gamma=gamma, g=g)
        tiny = 1e-12
        term1 = (
            kh**2
            * ((omega**2) / (N_B**2 + tiny))
            * (1.0 / (np.maximum(omega**2, tiny) - 1.0 + tiny))
        )
        term2 = (omega**2 - omega_a**2) / (c**2 + tiny)
        return term1 + term2

    @staticmethod
    def make_k_components(lambda_h, az_wave_deg):
        """Return kh, kx, ky for horizontal wavevector."""
        kh = 2.0 * np.pi / lambda_h
        az = np.deg2rad(az_wave_deg)
        kx = kh * np.cos(az)
        ky = kh * np.sin(az)
        return kh, kx, ky

    @staticmethod
    def k_parallel_to_B(kx, ky, m_r, I_deg, az_Bh_deg=0.0):
        """Return component of k along geomagnetic field."""
        I = np.deg2rad(I_deg)
        azb = np.deg2rad(az_Bh_deg)
        bx = np.cos(I) * np.cos(azb)
        by = np.cos(I) * np.sin(azb)
        bz = -np.sin(I)
        return kx * bx + ky * by + m_r * bz

    def compute(self, x, z, t, Ne_bg, dNe_dz, H):
        """
        Compute TID perturbation.

        Parameters
        ----------
        x, z, t : array_like
            Grids of horizontal, vertical, and time coordinates (m, m, s).
        Ne_bg : callable or array_like
            Background electron density profile function of z.
        dNe_dz : callable or array_like
            Gradient of Ne_bg.
        H : float or callable
            Scale height (m) or function of z.

        Returns
        -------
        Ne_prime : ndarray
            Complex perturbation electron density.
        Ne_prime_abs : ndarray
            Amplitude of perturbation.
        Phi : ndarray
            Phase term (rad).
        """
        x = np.asarray(x)
        z = np.asarray(z)
        t = np.asarray(t)
        X, Z, T = np.meshgrid(x, z, t, indexing="xy")

        Ne = Ne_bg(Z) if callable(Ne_bg) else np.asarray(Ne_bg)
        dNez = dNe_dz(Z) if callable(dNe_dz) else np.asarray(dNe_dz)
        Hval = H(Z) if callable(H) else np.asarray(H)
        z0_ref = self.z0_ref or float(np.nanmean(z))

        omega = 2.0 * np.pi / self.period
        kh, kx, ky = self.make_k_components(self.lambda_h, self.az_wave_deg)

        m2 = self.vertical_wavenumber(kh, omega, Hval)
        m_r = np.real(np.where(m2 >= 0.0, np.sqrt(m2), 0.0))
        kzi = (1.0 / (2.0 * Hval)) if (self.kzi_override is None) else self.kzi_override

        kbr = self.k_parallel_to_B(kx, ky, m_r, self.I_deg, az_Bh_deg=self.az_Bh_deg)

        I = np.deg2rad(self.I_deg)
        tiny = 1e-12
        geom_term = (1.0 / np.maximum(Ne, tiny)) * dNez + kzi
        arctan_term = np.arctan((np.sin(I) * geom_term) / np.maximum(kbr, tiny))
        Phi = (omega * T) - (kx * X + ky * 0 * X + m_r * Z) + arctan_term

        growth = np.exp(kzi * (Z - z0_ref))
        mag_factor = np.sqrt(
            (geom_term**2) + ((kbr / np.maximum(np.sin(I), tiny)) ** 2)
        )
        Ne_prime_abs = Ne * self.Ub_ref * growth * (np.sin(I) / omega) * mag_factor

        Ne_prime = Ne_prime_abs * np.exp(1j * Phi)
        return Ne_prime, Ne_prime_abs, Phi
