import datetime as dt

import iricore
import numpy as np
from geopy.distance import distance as geo_distance
from geopy.point import Point


def calculate_ne_from_plasma_frequency(fp):
    """
    Calculates the electron number density (Ne) from the angular plasma frequency.

    Args:
        fp (float): The plasma frequency in per second (/s).

    Returns:
        float: The electron number density in cubic meters (m^-3).
    """
    # Physical constants (in SI units)
    e = 1.602176634e-19  # Elementary charge (C)
    m_e = 9.1093837015e-31  # Electron mass (kg)
    epsilon_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)

    omega_p = 2 * np.pi * fp
    # Calculate electron number density (ne)
    ne = (omega_p**2 * epsilon_0 * m_e) / (e**2)

    return ne


class IRI(object):

    def __init__(self, event: dt.datetime, iri_version: int = 20):
        self.event = event
        self.iri_version = iri_version
        return

    def create_iri_2D_based_on_central_latlon(
        self,
        lat0_deg: float,
        lon0_deg: float,
        x_axis_pan_km: float = 300,
        dx: float = 1.0,
        unit: float = 1.0,  # change to 1e-6 if need /cc, as is /cm
        alts: np.ndarray = np.arange(100, 500, 1),
    ):
        x_axis = np.arange(-x_axis_pan_km, x_axis_pan_km, dx)
        y_axis = np.zeros_like(x_axis)
        self.alts = np.asarray(alts)
        x, y = np.broadcast_arrays(x_axis, y_axis)
        dist_km = np.hypot(x, y)
        bearing_deg = np.degrees(np.arctan2(x, y))

        # Flatten for simple geopy loop
        dist_flat = dist_km.ravel()
        bear_flat = bearing_deg.ravel()

        lat_flat = np.empty_like(dist_flat)
        lon_flat = np.empty_like(dist_flat)

        start = Point(lat0_deg, lon0_deg)

        for i, (d, brg) in enumerate(zip(dist_flat, bear_flat)):
            if np.isnan(d) or np.isnan(brg):
                lat_flat[i] = np.nan
                lon_flat[i] = np.nan
            else:
                dest = geo_distance(kilometers=float(d)).destination(start, float(brg))
                lat_flat[i] = dest.latitude
                lon_flat[i] = dest.longitude

        # Reshape and normalize lon to [-180, 180)
        self.lats = lat_flat.reshape(dist_km.shape)
        self.lons = ((lon_flat.reshape(dist_km.shape) + 180.0) % 360.0) - 180.0

        self.edens = np.zeros((len(self.alts), len(self.lats)))
        alt_range = [alts[0], alts[-1], alts[1] - alts[0]]

        for i in range(len(self.lats)):
            iriout = iricore.iri(
                self.event,
                alt_range,
                self.lats[i],
                self.lons[i],
                self.iri_version,
            )
            self.edens[:, i] = iriout.edens * unit
        # return density in /cm
        return self.edens

    def fetch_dataset(
        self,
        lats: np.array,
        lons: np.array,
        alts: np.array,
        unit: float = 1.0,  # change to 1e-6 if need /cc, as is /cm
    ):
        self.lats, self.alts, self.lons = (lats, alts, lons)
        self.edens, self.itemp, self.etemp = (
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
        )
        alt_range = [alts[0], alts[-1], alts[1] - alts[0]]
        for i in range(len(self.lats)):
            for j in range(len(self.lons)):
                iriout = iricore.iri(
                    self.event,
                    alt_range,
                    self.lats[i],
                    self.lons[j],
                    self.iri_version,
                )
                self.edens[:, i, j] = iriout.edens * unit
        # return density in /cm, and K
        return self.edens, self.itemp, self.etemp

    def iri_1D(
        self,
        lat: float,
        lon: float,
        alts: np.array,
        event: dt.datetime = None,
        unit: float = 1.0,
        # change to 1e-6 if need /cc, as is /cm
        **kwargs,
    ):
        event = event if event else self.event
        alt_range = [alts[0], alts[-1], alts[1] - alts[0]]
        iriout = iricore.iri(
            self.event,
            alt_range,
            lat,
            lon,
            version=self.iri_version,
            **kwargs,
        )
        edens = iriout.edens * unit
        # return density in /cm
        return edens


class IonosphereModels:

    @classmethod
    def chapman(cls, x, hs, NmF2=1e12, hmF2=300.0, scale_H=50.0, ne_floor=2e10):
        X, Z = np.meshgrid(x, hs)
        z = (Z - hmF2) / scale_H
        Ne = NmF2 * np.exp(0.5 * (1.0 - z - np.exp(-z))) + ne_floor
        return X, Z, Ne

    # background low density + F2-like bump that varies with x
    @classmethod
    def create_chapman_ionosphere_bump(
        cls,
        x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
        hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
        NmF2: float = 1e12,  # peak density [m^-3]
        hmF2: float = 300.0,  # F2 peak height [km]
        nmf2_funct=lambda dx: (1.0 + 0.15 * np.exp(-(((dx - 600) / 400) ** 2))),
        hmf2_funct=lambda dx: (30.0 * np.exp(-(((dx - 600) / 600) ** 2))),
        H_scale: float = 50.0,  # scale height [km]
        Ne_floor: float = 2e10,
    ):
        """
        Generate a 2D Chapman ionosphere with a localized horizontal bump.

        Parameters
        ----------
        x : array
            Horizontal axis [km]
        hs : array
            Altitude axis [km]
        NmF2 : float
            Peak electron density [m^-3] (background, scaled by bump function)
        hmF2 : float
            Peak height [km]
        nmf2_funct : callable
            Horizontal scaling function for NmF2 (default: Gaussian bump)

        Returns
        -------
        X : 2D array [Ny, Nx]
            Horizontal grid [km]
        Z : 2D array [Ny, Nx]
            Altitude grid [km]
        Ne : 2D array [Ny, Nx]
            Electron density [m^-3]
        """

        # Meshgrid: X along horizontal, Hs vertical
        X, Z = np.meshgrid(x, hs)

        # Apply horizontal scaling to NmF2 and hmF2
        NmF2_mod = NmF2 * nmf2_funct(X)
        hmF2_mod = hmF2 + hmf2_funct(X)

        # Chapman function
        z = (Z - hmF2_mod) / H_scale
        Ne = NmF2_mod * np.exp(0.5 * (1.0 - z - np.exp(-z))) + Ne_floor  # floor density

        return X, Z, Ne

    @staticmethod
    def default_obscuration_profile(
        dx: np.ndarray,
        peak_km: float = 0.0,
        half_width_km: float = 300.0,
    ) -> np.ndarray:
        """
        Normalized triangular obscuration profile with a configurable peak.

        Parameters
        ----------
        dx : array-like
            Horizontal coordinate(s) [km].
        peak_km : float
            Location of the maximum obscuration [km].
        half_width_km : float
            Distance from the peak where the obscuration tapers to zero [km].

        Returns
        -------
        np.ndarray
            Values between 0 and 1.
        """
        half_width_km = max(float(half_width_km), np.finfo(float).eps)
        profile = 1.0 - np.abs(np.asarray(dx, dtype=float) - peak_km) / half_width_km
        return np.clip(profile, 0.0, 1.0)

    @classmethod
    def n_layers(
        cls,
        x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
        hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
        layer_names: np.ndarray = np.asarray(["E", "F1", "F2"]),
        layer_heights: np.ndarray = np.asarray([110.0, 220.0, 300.0]),
        layer_base_ne: np.ndarray = np.asarray([1e11, 4.0e11, 8.0e11]),
        layer_scales: np.ndarray = np.asarray([10.0, 25.0, 50.0]),
        Ne_floor: float = 2e10,
    ):
        X, Z = np.meshgrid(x, hs)
        Ne = np.zeros_like(Z, dtype=float) + Ne_floor
        for name, ht, ne_base, hs in zip(
            layer_names, layer_heights, layer_base_ne, layer_scales
        ):
            z = (Z - ht) / hs
            Ne += ne_base * np.exp(0.5 * (1 - z - np.exp(-z)))
        return X, Z, Ne

    @classmethod
    def chapman_with_tilted_hmf2(
        cls,
        x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
        hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
        NmF2: float = 1e12,  # peak density [m^-3]
        hmF2: float = 300.0,  # F2 peak height [km]
        H_scale: float = 50.0,  # scale height [km]
        Ne_floor: float = 2e10,
        hmf2_tilt_funct=lambda dx: (0.05 * dx),
    ):
        """Introduce a horizontal gradient/tilt in hmF2"""
        X, Z = np.meshgrid(x, hs)
        hmF2 = hmF2 + hmf2_tilt_funct(X)
        z = (Z - hmF2) / H_scale
        Ne = NmF2 * np.exp(0.5 * (1 - z - np.exp(-z))) + Ne_floor
        return X, Z, Ne

    @classmethod
    def chapman_with_grading_obscuration(
        cls,
        x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
        hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
        NmF2: float = 1e12,  # peak density [m^-3]
        hmF2: float = 300.0,  # F2 peak height [km]
        H_scale: float = 50.0,  # scale height [km]
        Ne_floor: float = 2e10,
        obs_tilt_funct=None,
        obscuration_peak_km: float = 0.0,
        obscuration_half_width_km: float = 300.0,
        obscuration_depth: float = 1.0,
    ):
        """
        Introduce a Chapman layer with a horizontal obscuration gradient.

        Parameters
        ----------
        x, hs : array-like
            Horizontal and vertical grids [km].
        NmF2 : float
            Background peak electron density [m^-3].
        hmF2 : float
            F2 peak height [km].
        H_scale : float
            Scale height [km].
        Ne_floor : float
            Minimum electron density [m^-3].
        obs_tilt_funct : callable, optional
            Callable returning values in [0, 1] describing the obscuration profile.
        obscuration_peak_km : float
            Peak location for the default obscuration profile [km].
        obscuration_half_width_km : float
            Half-width for the default obscuration profile [km].
        obscuration_depth : float
            Fractional reduction applied at full obscuration (0=no change, 1=full).
        """
        X, Z = np.meshgrid(x, hs)
        z = (Z - hmF2) / H_scale
        base_profile = np.exp(0.5 * (1.0 - z - np.exp(-z)))

        if obs_tilt_funct is None:
            obscuration = cls.default_obscuration_profile(
                X,
                peak_km=obscuration_peak_km,
                half_width_km=obscuration_half_width_km,
            )
        else:
            obscuration = np.clip(np.asarray(obs_tilt_funct(X), dtype=float), 0.0, 1.0)

        depth = np.clip(obscuration_depth, 0.0, 1.0)
        NmF2_mod = NmF2 * (1.0 - depth * obscuration)

        Ne = NmF2_mod * base_profile + Ne_floor
        return X, Z, Ne

    @classmethod
    def cusp_function_alpha(
        cls,
        x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
        hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
        layer_names: np.ndarray = np.asarray(["E", "F1", "F2"]),
        layer_heights: np.ndarray = np.asarray([110.0, 220.0, 300.0]),
        layer_base_ne: np.ndarray = np.asarray([1e11, 4.0e11, 8.0e11]),
        layer_scales: np.ndarray = np.asarray([10.0, 25.0, 50.0]),
        Ne_floor: float = 2e10,
        x_params: np.ndarray = np.asarray([-93, 62, 127]),
        d_params: np.ndarray = np.asarray([0.09, 0.04]),
    ):
        X, Z, Ne = IonosphereModels.n_layers(
            x, hs, layer_names, layer_heights, layer_base_ne, layer_scales, Ne_floor
        )
        alpha_X = np.zeros_like(X)
        alpha_X[X <= x_params[0]] = 1
        alpha_X = np.where(
            (X > x_params[0]) & (X <= x_params[1]),
            1
            - d_params[0]
            * np.sin((np.pi / 2) * (X - x_params[0]) / (x_params[1] - x_params[0]))
            ** 2,
            alpha_X,
        )
        alpha_X = np.where(
            (X > x_params[1]) & (X <= x_params[2]),
            1
            - d_params[0]
            + (
                (d_params[0] - d_params[1])
                * np.sin((np.pi / 2) * (X - x_params[1]) / (x_params[2] - x_params[1]))
                ** 2
            ),
            alpha_X,
        )
        alpha_X[X > x_params[2]] = 1 - d_params[1]
        return X, Z, Ne, alpha_X, Ne * alpha_X

    @classmethod
    def cusp_function_tids(
        cls,
        x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
        hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
        NmF2: float = 1e12,
        hmF2: float = 300.0,
        H_scale: float = 50.0,
        Ne_floor: float = 2e10,
        A=0.1,
        kx=2 * np.pi / 200,
        kz=2 * np.pi / 300,
        phi=0,
    ):
        X, Z = np.meshgrid(x, hs)
        z = (Z - hmF2) / H_scale
        Ne = NmF2 * np.exp(0.5 * (1 - z - np.exp(-z))) + Ne_floor
        factor = A * np.sin(kx * X + kz * Z + phi)
        ndNe = (1.0 + factor) * Ne
        return X, Z, Ne, ndNe
