from typing import List

import numpy as np

from pynasonde.model.absorption.constants import pconst


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
        z = (h - hp) / scale_height
        Nh += Np * np.exp(0.5 * (1 - z - np.exp(-z)))  # Chapman function
    return (h, ne2f(Nh))


def parabolic_ionosphere(
    nbins: int,
    h_step: float,
    regions: List[str],
    ds: List[float],
    fps: List[float],
    hps: List[float],
):
    """
    A method to create parabolic layers of ionosphere using defination
    f(h) = fo(1-((h-ho)/d)**2) for |h-ho|<=d, else 0; where d is the thickness of
    the ionospheric parabolic layer.

    Final ionosphere is sum of regions described above.
    """
    h = np.arange(nbins) * h_step
    fh = np.zeros(nbins)
    for region, d, fp, hp in zip(regions, ds, fps, hps):
        f = np.zeros(nbins)
        mask = np.abs(h - hp) < d
        f[mask] = (fp * (1 - ((h - hp) / d) ** 2))[mask]
        fh += f
    return (h, fh)
