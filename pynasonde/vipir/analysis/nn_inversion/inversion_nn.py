"""Public API for NN-POLAN ionospheric inversion.

Drop-in complement to classical POLAN.  Uses the trained NN-POLAN model
(loaded from a .npz weight file) to invert a virtual-height ionogram trace
into a true-height electron density profile N(h).

Zero runtime dependencies beyond numpy — no PyTorch, no iricore.

Typical usage
-------------
    from pynasonde.vipir.analysis.nn_inversion.inversion_nn import NNInversion
    import numpy as np

    inv = NNInversion.from_weights("weights/WI937.npz")

    # Single ionogram
    ne, h = inv.invert(
        h_virtual_km=trace_km,       # (N_f,) on F_GRID_MHZ
        cond=np.array([37.9, -75.5, 180, 12.0, 2.0, 130.0]),
    )
    # ne : (N_h,) cm⁻³   h : (N_h,) km

    # Batch
    ne_batch, h_batch = inv.invert(
        h_virtual_km=traces_km,      # (B, N_f)
        cond=conds,                  # (B, 6)
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pynasonde.vipir.analysis.nn_inversion.forward_model import (
    F_GRID_MHZ,
    H_GRID_KM,
    find_foF2,
    forward_batch,
)
from pynasonde.vipir.analysis.nn_inversion.network import NNPolanNumpy


class NNInversion:
    """NN-POLAN true-height inversion.

    Parameters
    ----------
    model : NNPolanNumpy
        Loaded inference model.
    """

    def __init__(self, model: NNPolanNumpy) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_weights(cls, npz_path: str | Path) -> "NNInversion":
        """Load from a .npz weight file exported by export_weights.py."""
        return cls(NNPolanNumpy(Path(npz_path)))

    # ------------------------------------------------------------------
    # Inversion
    # ------------------------------------------------------------------

    def invert(
        self,
        h_virtual_km: np.ndarray,
        cond: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert virtual-height trace(s) to Ne profile(s).

        Parameters
        ----------
        h_virtual_km : (N_f,) or (B, N_f) — virtual heights [km]
                       NaN-padded entries are interpolated before inference.
        cond         : (6,) or (B, 6)     — [lat_deg, lon_deg, doy, ut_h, Kp, F10.7_sfu]

        Returns
        -------
        ne_cm3 : (N_h,) or (B, N_h) — electron density [cm⁻³]
        h_km   : (N_h,)              — height grid [km]  (same for all profiles)
        """
        h_virtual_km = np.asarray(h_virtual_km, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float32)
        squeeze = h_virtual_km.ndim == 1

        if squeeze:
            h_virtual_km = h_virtual_km[np.newaxis]
            cond = cond[np.newaxis]

        h_virtual_km = _fill_nan_trace(h_virtual_km)
        ne_cm3 = self._model.predict(h_virtual_km, cond)

        if squeeze:
            ne_cm3 = ne_cm3[0]

        return ne_cm3, H_GRID_KM.copy()

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def profile_summary(
        self,
        ne_cm3: np.ndarray,
        h_km: np.ndarray | None = None,
    ) -> dict:
        """Compute ionospheric scalars from a Ne profile.

        Parameters
        ----------
        ne_cm3 : (N_h,) electron density [cm⁻³]
        h_km   : (N_h,) height grid [km] — defaults to H_GRID_KM

        Returns
        -------
        dict with keys: foF2_MHz, hmF2_km, NmF2_cm3, foE_MHz, hmE_km
        """
        if h_km is None:
            h_km = H_GRID_KM

        from pynasonde.vipir.analysis.nn_inversion.forward_model import ne_to_fp

        fp = ne_to_fp(ne_cm3.astype(np.float64))  # MHz

        foF2 = float(fp.max())
        hmF2 = float(h_km[np.argmax(fp)])
        NmF2 = float(ne_cm3[np.argmax(fp)])

        # E-layer: look for local peak between 90–150 km
        e_mask = (h_km >= 90.0) & (h_km <= 150.0)
        if e_mask.sum() > 0:
            e_fp = fp[e_mask]
            e_h = h_km[e_mask]
            foE = float(e_fp.max())
            hmE = float(e_h[e_fp.argmax()])
        else:
            foE = hmE = float("nan")

        return dict(foF2_MHz=foF2, hmF2_km=hmF2, NmF2_cm3=NmF2, foE_MHz=foE, hmE_km=hmE)

    def reconstruct_trace(self, ne_cm3: np.ndarray) -> np.ndarray:
        """Run forward model on predicted Ne → reconstructed h'(f).

        Useful for computing residual between observed and predicted trace.

        Parameters
        ----------
        ne_cm3 : (N_h,) or (B, N_h)

        Returns
        -------
        h_virtual_km : (N_f,) or (B, N_f)
        """
        squeeze = ne_cm3.ndim == 1
        if squeeze:
            ne_cm3 = ne_cm3[np.newaxis]
        hv = forward_batch(ne_cm3)
        return hv[0] if squeeze else hv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_nan_trace(hv: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN entries in virtual-height trace(s).

    hv : (B, N_f)
    """
    hv = hv.copy()
    for b in range(hv.shape[0]):
        row = hv[b]
        nans = np.isnan(row)
        if nans.all():
            hv[b] = 200.0  # fallback neutral value
            continue
        if nans.any():
            idx = np.arange(len(row))
            good = ~nans
            hv[b] = np.interp(idx, idx[good], row[good])
    return hv
