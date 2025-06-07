from typing import List

import numpy as np
from loguru import logger

from pynasonde.model.absorption.constants import pconst
from pynasonde.model.polan.datasets import SimulationOutputs, Trace


def ne2f(ne: np.array, in_mhz=True):
    """
    Convert electron density (m^-3) to frequency in MHz or rad/s.
    """
    ne = np.asarray(ne)
    omega_p = np.sqrt((ne * pconst["q_e"] ** 2) / (pconst["eps0"] * pconst["m_e"]))
    if in_mhz:
        omega_p = omega_p / (2 * np.pi) / 1e6
    return omega_p


def f2ne(f: np.array, in_cc=False):
    """
    Convert plsama frequency (in MHz) to electron density in /cc or /m^3
    """
    f = np.asarray(f)
    ne = (
        (pconst["eps0"] * pconst["m_e"])
        * (f * (2 * np.pi) * 1e6) ** 2
        / pconst["q_e"] ** 2
    )
    if in_cc:
        ne = ne * 1e-6
    return ne


def chapman_ionosphere(
    nbins: int,
    h_step: float,
    regions: List[str],
    Nps: List[float],
    hps: List[float],
    scale_hs: List[float],
):
    """
    Calculates electron density for a Chapman layer.

    Returns:
        float or numpy.ndarray: Electron density at the given altitude (electrons/m^3).
    """
    h = np.arange(nbins) * h_step
    Nh = np.zeros(nbins)
    for region, Np, hp, scale_height in zip(regions, Nps, hps, scale_hs):
        logger.info(f"Chapman Region {region}, Np:{Np}, hp:{hp}, Hs: {scale_height}")
        z = (h - hp) / scale_height
        Nh += Np * np.exp(0.5 * (1 - z - np.exp(-z)))  # Chapman function
    return (h, ne2f(Nh))


def parabolic_ionosphere(
    nbins: int,
    h_step: float,
    regions: List[str],
    ds: List[float],
    Nps: List[float],
    hps: List[float],
):
    """
    Creates parabolic layers of the ionosphere using the definition:

    f(h) = fp * (1 - ((h - hp) / d) ** 2) for |h - hp| <= d, else 0;

    where:
        - d is the thickness of the ionospheric parabolic layer,
        - fp is the peak plasma frequency,
        - hp is the height of the peak plasma frequency.

    The final ionosphere is the sum of all regions described above.
    """
    h = np.arange(nbins) * h_step
    fh = np.zeros(nbins)
    for region, d, Np, hp in zip(regions, ds, Nps, hps):
        logger.info(f"Parabolic Region {region}, Np:{Np}, hp:{hp}, Ds: {d}")
        fp = ne2f(Np)
        f = np.zeros(nbins)
        mask = np.abs(h - hp) < d
        f[mask] = (fp * (1 - ((h - hp) / d) ** 2))[mask]
        fh += f
    return (h, fh)


def generate_random_samples(hp_bounds, np_bounds, hd_bounds, n_samples=100):
    """
    Generate random 3D samples within the specified bounds.
    Returns:
        np.ndarray: Array of shape (n_samples, 3)
    """
    hp_samples = np.random.uniform(hp_bounds[0], hp_bounds[1], n_samples)
    np_samples = np.random.uniform(np_bounds[0], np_bounds[1], n_samples)
    hd_samples = np.random.uniform(hd_bounds[0], hd_bounds[1], n_samples)

    return np.vstack((hp_samples, np_samples, hd_samples)).T


def get_Np_bounds_from_fv(trace_fv: np.array, up: float = 0.0, down: float = 0.0):
    return [f2ne(np.max(trace_fv) - down), f2ne(np.max(trace_fv) + up)]


def get_hp_bounds_from_ht(trace_ht: np.array, up: float = 0.0, down: float = 0.0):
    return [np.mean(trace_ht) - down, np.mean(trace_ht) + up]


def get_hp_bounds_from_scale_h(trace_ht: np.array, up: float = 0.0, down: float = 0.0):
    return [
        ((np.max(trace_ht) - np.min(trace_ht)) / 2) - down,
        ((np.max(trace_ht) - np.min(trace_ht)) / 2) + up,
    ]
