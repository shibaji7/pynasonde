"""NN-POLAN: physics-informed neural network true-height inversion.

Public API
----------
    from pynasonde.vipir.analysis.nn_inversion import NNInversion

    inv = NNInversion.from_weights("path/to/weights.npz")
    ne, h = inv.invert(h_virtual_km, cond)
"""

from pynasonde.vipir.analysis.nn_inversion.forward_model import (
    F_GRID_MHZ,
    H_GRID_KM,
    find_foF2,
    forward_batch,
    forward_scalar,
    observable_f_grid,
)
from pynasonde.vipir.analysis.nn_inversion.inversion_nn import NNInversion

__all__ = [
    "NNInversion",
    "H_GRID_KM",
    "F_GRID_MHZ",
    "forward_batch",
    "forward_scalar",
    "find_foF2",
    "observable_f_grid",
]
