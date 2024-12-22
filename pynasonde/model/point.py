#!/usr/bin/env python

"""point.py: create one point in the ionosphere to load along the altitude."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "chakras4@erau.edu"
__status__ = "Research"

import datetime as dt
from dataclasses import dataclass

import numpy as np

from pynasonde.model.absorption.collisions import CollisionProfiles
from pynasonde.model.absorption.dispersion_relations import AbsorptionProfiles
from pynasonde.model.igrf import IGRF
from pynasonde.model.iri import IRI
from pynasonde.model.msise import MSISE


@dataclass
class Point:
    """
    Altitude profiles of the following items
    """

    date: dt.datetime = None  # Datetime of the event / point
    lat: np.float64 = np.nan  # Latitude of the point
    lon: np.float64 = np.nan  # Longitude of the point
    alts: np.array = None  # Height along lat/lon
    eden: np.array = None  # Electron density profile
    B_north: np.array = None  # Magnetic field [North]
    B_east: np.array = None  # Magnetic field [East]
    B_down: np.array = None  # Magnetic field [Down]
    B_tot: np.array = None  # Total magnetic field
    B_incl: np.array = None  # Magnetic field inclination angle
    B_decl: np.array = None  # Magnetic field declination angle
    # MSISE datasets
    He: np.array = None  # Helium
    H: np.array = None  # Hydrogen
    O: np.array = None  # Atomic Oxygen
    Ar: np.array = None  # Argon
    N: np.array = None  # Atomic Nitrogen
    N2: np.array = None  # Molecular Nitrogen
    O2: np.array = None  # Molecular Oxygen
    rho: np.array = None  # Density in g/cc
    AnomO: np.array = None  # Anomalous Oxygen
    Texo: np.array = None  # Exo temperature K
    Talt: np.array = None  # Neutral temperature K
    Ap: np.float64 = np.nan
    f107: np.float64 = np.nan
    f107a: np.float64 = np.nan
    # Collision and absorption profiles
    absorption_profile: AbsorptionProfiles = None
    collision_profiles: CollisionProfiles = None

    def _load_profile_(
        self,
        edensity_model: str = "iri20",
        mag_model: str = "igrf",
        neutral_density_model: str = "msise",
    ):
        """
        Load based on the model types
        """
        if "iri" in edensity_model:
            self.iri = IRI(self.date, int(edensity_model.replace("iri", "")))
            self.eden = self.iri.fetch_dataset(
                np.array([self.lat]), np.array([self.lon]), self.alts
            )
        if "igrf" in mag_model:
            self.igrf = IGRF(self.date)
            (
                self.B_north,
                self.B_east,
                self.B_down,
                self.B_tot,
                self.B_incl,
                self.B_decl,
            ) = self.igrf.fetch_dataset(
                np.array([self.lat]), np.array([self.lon]), self.alts
            )
        if neutral_density_model == "msise":
            self.msise = MSISE(self.date)
            (
                self.He,
                self.H,
                self.O,
                self.Ar,
                self.N,
                self.N2,
                self.O2,
                self.rho,
                self.AnoO,
                self.Texo,
                self.Talt,
                self.Ap,
                self.f107,
                self.f107a,
            ) = self.msise.fetch_dataset(
                np.array([self.lat]), np.array([self.lon]), self.alts
            )
        return
