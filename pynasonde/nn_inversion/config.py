"""Central configuration loader for the NN-POLAN inversion pipeline.

All tuneable constants live in ``pynasonde/config.toml`` under the
``[nn_inversion]`` section.  Import this module instead of scattering
magic numbers across synthetic_data.py / trainer_stage1.py / physics_loss.py
/ architecture.py.

Usage
-----
    from pynasonde.nn_inversion.config import NNCfg

    NNCfg.data.geo_chunk          # 4096
    NNCfg.model.latent_dim        # 256
    NNCfg.training.warmup_epochs  # 5
    NNCfg.normalisation.hv_std_km # 150.0
"""

from __future__ import annotations

import numpy as np

from pynasonde.vipir.ngi.utils import load_toml

# Load once at import time — same pattern used everywhere else in pynasonde
_cfg = load_toml()
NNCfg = _cfg.nn_inversion

# ---------------------------------------------------------------------------
# Derived arrays that are computed from the scalar grid params
# (kept here so synthetic_data.py doesn't repeat the meshgrid logic)
# ---------------------------------------------------------------------------

lat_step = NNCfg.data.lat_step_deg
lon_step = NNCfg.data.lon_step_deg

LAT_DEG: np.ndarray = np.arange(-90.0, 91.0, lat_step)
LON_DEG: np.ndarray = np.arange(-180.0, 180.0, lon_step)
YEARS: np.ndarray = np.arange(
    int(NNCfg.data.year_start), int(NNCfg.data.year_end) + 1, dtype=int
)
DOY: np.ndarray = np.arange(1, 367, int(NNCfg.data.doy_step), dtype=int)[
    : int(NNCfg.data.doy_n)
]
UT_H: np.ndarray = np.arange(0, 24, int(NNCfg.data.ut_step_h), dtype=int)

# Flat (lat, lon) mesh — shape (N_G,), order matches PyIRI alon/alat args
_lats, _lons = np.meshgrid(LAT_DEG, LON_DEG, indexing="ij")
ALAT_FLAT: np.ndarray = _lats.ravel().astype(np.float64)
ALON_FLAT: np.ndarray = _lons.ravel().astype(np.float64)

# Conditioning normalisation as numpy arrays (for trainer_stage1.py)
COND_MEAN: np.ndarray = np.array(NNCfg.normalisation.cond_mean, dtype=np.float32)
COND_STD: np.ndarray = np.array(NNCfg.normalisation.cond_std, dtype=np.float32)
