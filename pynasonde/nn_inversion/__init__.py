"""NN-POLAN: physics-informed neural network true-height inversion.

Instrument-agnostic inversion of virtual-height ionogram traces h'(f) into
true-height electron density profiles Ne(h).  Works with any ionosonde that
produces a virtual-height trace — VIPIR RIQ or Digisonde RSF.

Public API
----------
    from pynasonde.nn_inversion import NNInversion, F_GRID_MHZ, H_GRID_KM

    inv = NNInversion.from_weights("path/to/weights.npz")
    ne, h = inv.invert(h_virtual_km, cond)

Adapters
--------
    from pynasonde.nn_inversion.adapters import VipirAdapter, DigisondeAdapter
"""

from pynasonde.nn_inversion.forward_model import (
    F_GRID_MHZ,
    H_GRID_KM,
    find_foF2,
    forward_batch,
    forward_scalar,
    observable_f_grid,
)
from pynasonde.nn_inversion.inversion_nn import NNInversion

__all__ = [
    "NNInversion",
    "H_GRID_KM",
    "F_GRID_MHZ",
    "forward_batch",
    "forward_scalar",
    "find_foF2",
    "observable_f_grid",
]
