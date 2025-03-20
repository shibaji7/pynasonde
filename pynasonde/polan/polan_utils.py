import numpy as np


def chapman_layer(
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
    z = (altitude - peak_altitude) / scale_height
    density = peak_density * np.exp(0.5 * (1 - z - np.exp(-z)))  # Chapman function
    return density


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
