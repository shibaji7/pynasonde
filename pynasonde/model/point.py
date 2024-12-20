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

from pynasonde.model.igrf import IGRF
from pynasonde.model.iri import IRI


@dataclass
class Point:
    """
    Altitude profiles of the following items
    """

    date: dt.datetime = None  # Datetime of the event / point
    lat: np.float64 = np.nan  # Latitude of the point
    lon: np.float64 = np.nan  # Longitude of the point
    alts: np.array = np.arange(50, 500)  # Height along a
    eden: np.array = None  # Electron density profile
    B_north: np.array = None  # Magnetic field [North]
    B_east: np.array = None  # Magnetic field [East]
    B_down: np.array = None  # Magnetic field [Down]
    B: np.array = None  # Total magnetic field
    B_incl: np.array = None  # Magnetic field inclination angle
    B_decl: np.array = None  # Magnetic field declination angle

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
            iri = IRI(self.date, int(edensity_model.replace("iri", "")))
            self.eden = iri.fetch_dataset(
                np.array([self.lat]), np.array([self.lon]), self.alts
            )
        if "igrf" in mag_model:
            igrf = IGRF(self.date)
            self.B_north, self.B_east, self.B_down, self.B, self.B_incl, self.B_decl = (
                igrf.fetch_dataset(
                    np.array([self.lat]), np.array([self.lon]), self.alts
                )
            )
        return
