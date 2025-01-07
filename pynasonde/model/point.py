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

import copy
import datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.model.absorption.collisions import CalculateCollision, CollisionProfiles
from pynasonde.model.absorption.dispersion_relations import (
    AbsorptionProfiles,
    CalculateAbsorption,
)
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
    edens: np.array = None  # Electron density profile
    etemp: np.array = None  # Electron Temperature
    itemp: np.array = None  # Ion Temperature
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
    absorption_profiles: List[AbsorptionProfiles] = None
    collision_profile: CollisionProfiles = None

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
            logger.info(f"Loading IRI for {self.date} at {self.lat}/{self.lon}")
            self.iri = IRI(self.date, int(edensity_model.replace("iri", "")))
            self.edens, self.itemp, self.etemp = self.iri.fetch_dataset(
                np.array([self.lat]), np.array([self.lon]), self.alts
            )
        if "igrf" in mag_model:
            logger.info(f"Loading IGRF for {self.date} at {self.lat}/{self.lon}")
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
            logger.info(f"Loading MSISE00 for {self.date} at {self.lat}/{self.lon}")
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
                self.total_n_density,
            ) = self.msise.fetch_dataset(
                np.array([self.lat]), np.array([self.lon]), self.alts
            )
        return

    def calculate_collision_freqs(self):
        col = CalculateCollision(
            self.alts,
            self.edens,
            self.total_n_density,
            self.Texo,
            self.Texo,
            self.Talt,
        )
        col.calculate_FT_collision_frequency()
        col.calculate_SN_en_collision_frequency(
            self.N2, self.O2, self.O, self.He, self.H
        )
        # col.calculate_SN_ei_collision_frequency()
        self.collision_profile = copy.copy(col.cp)
        return

    def calculate_absorptions(self, f_sweep: np.array = np.linspace(2, 4, 5)):
        if self.collision_profile is None:
            self.calculate_collision_freqs()
        self.absorption_profiles, self.f_sweep = [], f_sweep
        for fo in f_sweep:
            caa = CalculateAbsorption(
                self.B_tot, self.collision_profile, self.edens, fo * 1e6
            )
            caa.estimate_AH()
            # caa.estimate_SW()
            self.absorption_profiles.append(copy.copy(caa.abs_profiles))
        return

    def get_absoption_profiles(
        self,
        fo: float = 2,
        disp_equation_kind: str = "ah",
        freq_kind: str = "av_cc",
        mode: str = "O",
        do_pandas: bool = True,
    ):
        profile = self.absorption_profiles[self.f_sweep.tolist().index(fo)]
        absorption = getattr(
            getattr(getattr(profile, disp_equation_kind), freq_kind), mode
        )
        if do_pandas:
            df = pd.DataFrame()
            df["alts"], df["absorption"] = self.alts, absorption.ravel()
            return df
        else:
            return absorption

    def find_ionogram_trace_max_height(
        self,
        f_sweep: List[float],
        disp_equation_kind: str = "ah",
        freq_kind: str = "av_cc",
        mode: str = "O",
        power_limit: float = 1.0,
        height_limit: float = 250.0,
    ):
        max_ret_heights = []
        for fo in f_sweep:
            df = self.get_absoption_profiles(fo, disp_equation_kind, freq_kind, mode)
            df["cumsum_abs"] = df.absorption.cumsum()
            print(fo, df.absorption.argmax())
            df = df[(df.absorption <= power_limit) & (df.alts <= height_limit)]
            max_ret_heights.append(df.alts.max())
        df = pd.DataFrame()
        df["max_ret_heights"], df["f_sweep"] = max_ret_heights, f_sweep
        df.max_ret_heights.replace(height_limit, np.nan, inplace=True)
        return df
