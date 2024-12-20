#!/usr/bin/env python

"""constant.py: constant instatntiate all the constants."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "chakras4@erau.edu"
__status__ = "Research"

import numpy as np

"""
Physical constants
"""
pconst = {
    "boltz": 1.38066e-23,  # Boltzmann constant  in Jule K^-1
    "h": 6.626e-34,  # Planks constant  in ergs s
    "c": 2.9979e08,  # in m s^-1
    "avo": 6.023e23,  # avogadro's number
    "Re": 6371.0e3,
    "amu": 1.6605e-27,
    "q_e": 1.602e-19,  # Electron charge in C
    "m_e": 9.109e-31,  # Electron mass in kg
    "g": 9.81,  # Gravitational acceleration on the surface of the Earth
    "eps0": 1e-9 / (36 * np.pi),
    "R": 8.31,  # J mol^-1 K^-1
}

mass = {
    "O3": 48.0,
    "O2": 32.0,
    "O": 16.0,
    "N2": 28.0,
    "AR": 40.0,
    "Na": 23.0,
    "He": 4.0,
    "NO": 30.0,
    "N4s": 14.0,
    "N2d": 14.0,
    "CH4": 16.0,
    "H2": 2.0,
    "CO": 28.0,
    "CO2": 44.0,
    "H2O": 18.0,
    "Hox": 1.0,
    "H": 1.0,
}
