"""Instrument adapters for NN-POLAN.

Each adapter converts instrument-native data into the canonical
h_virtual_km array on F_GRID_MHZ, ready for NNInversion.invert().

    from pynasonde.nn_inversion.adapters import VipirAdapter, DigisondeAdapter
    from pynasonde.nn_inversion.adapters.base import resample_trace
"""

from pynasonde.nn_inversion.adapters.base import resample_trace
from pynasonde.nn_inversion.adapters.digisonde import DigisondeAdapter
from pynasonde.nn_inversion.adapters.vipir import VipirAdapter

__all__ = ["resample_trace", "VipirAdapter", "DigisondeAdapter"]
