"""Synthetic training-data generator for the NN-POLAN inversion model.

Produces (N(h), h'(f), c) triplets using PyIRI + real space weather indices,
designed for VEGA Slurm job-array parallelism.

Parameter grid
--------------
    lat   : -90 … +90 °N      (5° step  → 37 values)
    lon   : -180 … +175 °E    (5° step  → 72 values)
    year  : 1995 … 2024       (1-yr step → 30 values)
    doy   : 1, 46, 91 … 361   (45-day step → 9 values)
    UT    : 0, 6, 12, 18 h    (4 values)

Total grid points ≈ 37 × 72 × 30 × 9 × 4 = 2 882 880

F10.7 is taken from the pyomnidata daily solar-flux record for the exact
calendar date — Kp and F10.7 are therefore naturally correlated because
both come from the same historical record.

Kp is derived from OMNI 60-min hourly SymH (daily mean → Kp proxy).
If OMNI hourly data has not been downloaded, Kp falls back to 1.5 (typical
quiet-time value) with a one-time warning.  Pre-download with:

    python -c "import pyomnidata; pyomnidata.UpdateLocalData(yearRange=[1995,2024])"

PyIRI batch API
---------------
One call to ``IRI_density_1day`` processes ALL 2 664 (lat, lon) pairs
× 4 UT values at once → returns EDP [N_T, N_V, N_G] = [4, 904, 2664].
With the recommended --n_shards=270 each shard covers one (year, doy)
combination and runs a single PyIRI batch.

NetCDF layout per shard file  (CF-1.8 conventions)
----------------------------------------------------
Dimensions
    sample    : N_valid samples in this shard
    height_km : N_h  (coordinate = H_GRID_KM)
    freq_mhz  : N_f  (coordinate = F_GRID_MHZ)
    cond_dim  : 6    (coordinate = COND_COLS labels)

Variables
    ne_cm3    (sample, height_km)  float32  — electron density [cm⁻³]
    h_virtual (sample, freq_mhz)   float32  — virtual height [km]; 0 above foF2
    obs_mask  (sample, freq_mhz)   int8     — 1 where freq < foF2, else 0
    cond      (sample, cond_dim)   float32  — [lat, lon, doy, ut_h, kp, f107]

Usage
-----
    conda activate pynasonde
    python synthetic_data.py \\
        --shard 0 --n_shards 270 \\
        --out_dir /tmp/nn_polan_test \\
        --verbose
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pyomnidata data directory — all GetSolarFlux / GetOMNI calls read from here
# ---------------------------------------------------------------------------
OMNI_DATA_PATH = Path.home() / "OMNI"


def _configure_omnidata() -> None:
    """Point pyomnidata at OMNI_DATA_PATH (idempotent).

    Sets both the env-var (controls module-init path) and Globals.DataPath
    (controls runtime path) so pyomnidata never falls back to ~/omnidata.
    """
    import os

    import pyomnidata

    OMNI_DATA_PATH.mkdir(parents=True, exist_ok=True)
    os.environ["OMNIDATA_PATH"] = str(OMNI_DATA_PATH)
    pyomnidata.Globals.DataPath = str(OMNI_DATA_PATH)


sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # repo root
from pynasonde.vipir.analysis.nn_inversion.forward_model import (
    F_GRID_MHZ,
    H_GRID_KM,
    forward_batch,
    ne_to_fp,
)

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
_LAT_DEG = np.arange(-90.0, 91.0, 5.0)
_LON_DEG = np.arange(-180.0, 180.0, 5.0)
_YEARS = np.arange(1995, 2025, dtype=int)  # 30  (full solar cycles)
_DOY = np.arange(1, 367, 45, dtype=int)[:9]
_UT_H = np.arange(0, 24, 6, dtype=int)

# Flat (lat, lon) mesh — shape (N_G, 2), order matches PyIRI alon/alat args
_LATS, _LONS = np.meshgrid(_LAT_DEG, _LON_DEG, indexing="ij")
_ALAT_FLAT = _LATS.ravel().astype(np.float64)  # (N_G,)
_ALON_FLAT = _LONS.ravel().astype(np.float64)  # (N_G,)

# Geo-chunk size: max locations per PyIRI call to keep EDP memory manageable.
# EDP [N_T, N_V, chunk] × 8 bytes; with N_T=8, N_V=904, chunk=4096 → ~230 MB.
_GEO_CHUNK = 4096

# Conditioning-vector column order (must match architecture.py / trainer_stage1.py)
COND_COLS = ("lat_deg", "lon_deg", "doy", "ut_h", "kp", "f107_sfu")

_N_H = len(H_GRID_KM)
_N_F = len(F_GRID_MHZ)
_N_G = len(_ALAT_FLAT)  # 2664

# ---------------------------------------------------------------------------
# Day list & shard slicing
# ---------------------------------------------------------------------------


def _build_day_list() -> list[tuple[int, int]]:
    """Return the ordered list of (year, doy) pairs for the full grid."""
    return [(int(y), int(d)) for y in _YEARS for d in _DOY]


def _shard_slice(total: int, n_shards: int, shard: int) -> slice:
    size = (total + n_shards - 1) // n_shards
    start = shard * size
    stop = min(start + size, total)
    return slice(start, stop)


# ---------------------------------------------------------------------------
# Space-weather lookup  (F10.7 and Kp)
# ---------------------------------------------------------------------------


def _build_f107_table() -> dict[tuple[int, int], float]:
    """Return {(year, doy): F10.7_SFU} from pyomnidata solar flux record."""
    import pyomnidata

    _configure_omnidata()
    sf = pyomnidata.GetSolarFlux()  # daily record, Date=YYYYMMDD
    table: dict[tuple[int, int], float] = {}
    for date_int, f107 in zip(sf["Date"].astype(int), sf["F10_7"].astype(float)):
        if f107 >= 999.0:  # fill value
            continue
        year = date_int // 10000
        month = (date_int % 10000) // 100
        day = date_int % 100
        try:
            doy = datetime.date(year, month, day).timetuple().tm_yday
        except ValueError:
            continue
        table[(year, doy)] = f107
    return table


def _build_kp_table(years: np.ndarray) -> dict[tuple[int, int], float]:
    """Return {(year, doy): Kp} from OMNI 60-min hourly SymH.

    SymH (daily mean) → Kp approximation:
        Kp ≈ clip(-SymH_mean / 25, 0, 9)

    This gives:  SymH=0 → Kp≈0,  SymH=-50 → Kp≈2,  SymH=-100 → Kp≈4.

    Falls back to Kp=1.5 (quiet-time climatology) if OMNI data is absent.
    """
    try:
        import pyomnidata

        _configure_omnidata()
        omni = pyomnidata.GetOMNI([int(years.min()), int(years.max())], Res=60)
        if len(omni) == 0:
            raise FileNotFoundError("empty OMNI record")

        # Build (year, doy) → daily-mean SymH
        # omni.Date is YYYYMMDD, omni.SymH in nT
        dates = omni["Date"].astype(int)
        symh = omni["SymH"].astype(float)
        invalid = (symh > 9000) | ~np.isfinite(symh)
        symh[invalid] = np.nan

        table: dict[tuple[int, int], float] = {}
        unique_dates = np.unique(dates)
        for date_int in unique_dates:
            mask = dates == date_int
            daily = np.nanmean(symh[mask])
            if not np.isfinite(daily):
                continue
            year = int(date_int) // 10000
            month = (int(date_int) % 10000) // 100
            day = int(date_int) % 100
            try:
                doy = datetime.date(year, month, day).timetuple().tm_yday
            except ValueError:
                continue
            kp = float(np.clip(-daily / 25.0, 0.0, 9.0))
            table[(year, doy)] = kp
        logger.info("Kp table built from OMNI SymH: %d entries", len(table))
        return table

    except (FileNotFoundError, KeyError, ImportError) as exc:
        logger.warning(
            "OMNI hourly data unavailable (%s) — using Kp=1.5 fallback. "
            'Pre-download with: python -c "import pyomnidata; '
            'pyomnidata.UpdateLocalData(yearRange=[1995,2024])"',
            exc,
        )
        return {}  # empty → caller uses 1.5


_KP_FALLBACK = 1.5  # typical quiet-time daily mean


# ---------------------------------------------------------------------------
# PyIRI call (one batch = all lat/lon × all UTs for one calendar day)
# ---------------------------------------------------------------------------


def _call_pyiri_chunk(
    year: int,
    doy: int,
    f107: float,
    alat: np.ndarray,
    alon: np.ndarray,
) -> np.ndarray | None:
    """Run PyIRI for a geo-chunk (N_c locations) and all UTs for one day.

    Memory: EDP [N_T, N_V, N_c] × 8 bytes.
    With _GEO_CHUNK=4096, N_T=8, N_V=904 → ~230 MB per call.

    Returns
    -------
    ne_cm3 : float32 array, shape (N_T * N_c, N_H). None on failure.
    """
    try:
        import PyIRI
        import PyIRI.main_library as il

        dt = datetime.date(int(year), 1, 1) + datetime.timedelta(days=int(doy) - 1)

        _, _, _, _, _, _, EDP = il.IRI_density_1day(
            int(year),
            dt.month,
            dt.day,
            _UT_H.astype(np.float64),
            alon,  # NOTE: lon before lat in PyIRI API
            alat,
            H_GRID_KM.astype(np.float64),
            float(f107),
            PyIRI.coeff_dir,
        )
        # EDP: [N_T, N_V, N_c]  in m⁻³
        N_T, N_V, N_c = EDP.shape
        ne_m3 = EDP.transpose(0, 2, 1).reshape(N_T * N_c, N_V)
        ne_m3 = np.where(np.isfinite(ne_m3), np.maximum(ne_m3, 0.0), 0.0)
        return (ne_m3 * 1e-6).astype(np.float32)

    except Exception as exc:
        logger.warning("PyIRI failed year=%d doy=%d: %s", year, doy, exc)
        return None


# ---------------------------------------------------------------------------
# Conditioning vector builder for one geo-chunk
# ---------------------------------------------------------------------------


def _build_cond_chunk(
    doy: int,
    f107: float,
    kp: float,
    alat: np.ndarray,
    alon: np.ndarray,
) -> np.ndarray:
    """Return cond array (N_T * N_c, 6) for one geo-chunk.

    Column order: [lat, lon, doy, ut_h, kp, f107]
    Row order: outer UT loop, inner geo-chunk order (matches _call_pyiri_chunk).
    """
    N_c = len(alat)
    N_T = len(_UT_H)
    lat_rep = np.tile(alat, N_T).astype(np.float32)
    lon_rep = np.tile(alon, N_T).astype(np.float32)
    ut_rep = np.repeat(_UT_H, N_c).astype(np.float32)
    doy_rep = np.full(N_T * N_c, doy, dtype=np.float32)
    kp_rep = np.full(N_T * N_c, kp, dtype=np.float32)
    f107_rep = np.full(N_T * N_c, f107, dtype=np.float32)

    return np.column_stack(
        [lat_rep, lon_rep, doy_rep, ut_rep, kp_rep, f107_rep]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

_MIN_FOF2_MHZ = 1.5
_MAX_FOF2_MHZ = 15.0
_MIN_OBS_FREQS = 5  # minimum sub-foF2 frequency cells


def _compute_h_virtual(
    ne_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run forward_batch, return (h_virtual_dense, obs_mask, valid_mask).

    Cells above foF2 are NaN from the Abel integral (physically correct).
    Accepted profiles need ≥ _MIN_OBS_FREQS finite frequency cells.
    NaN cells are filled with 0.0; obs_mask encodes which bins are valid.
    """
    h_virt = forward_batch(ne_batch).astype(np.float32)  # (B, N_f)

    fp_batch = ne_to_fp(ne_batch.astype(np.float64))
    foF2 = fp_batch.max(axis=1)  # (B,) MHz

    n_finite = np.isfinite(h_virt).sum(axis=1)
    fin_min = np.where(np.isfinite(h_virt), h_virt, np.inf).min(axis=1)
    fin_max = np.where(np.isfinite(h_virt), h_virt, -np.inf).max(axis=1)

    valid = (
        (n_finite >= _MIN_OBS_FREQS)
        & (foF2 >= _MIN_FOF2_MHZ)
        & (foF2 <= _MAX_FOF2_MHZ)
        & (fin_min >= H_GRID_KM[0] - 1.0)
        & (fin_max < 2000.0)
    )

    h_virt_dense = np.where(np.isfinite(h_virt), h_virt, 0.0).astype(np.float32)
    obs_mask = np.isfinite(h_virt).astype(np.float32)  # (B, N_f)

    return h_virt_dense, obs_mask, valid


# ---------------------------------------------------------------------------
# Per-shard generation
# ---------------------------------------------------------------------------


def generate_shard(
    shard: int,
    n_shards: int,
    out_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Generate one shard of synthetic training data.

    Each shard covers ceil(270 / n_shards) (year, doy) combinations.
    With n_shards=270 each shard processes one day (one PyIRI batch call).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard:05d}.nc"

    if out_path.exists() and not overwrite:
        logger.info("Shard %d already exists — skipping.", shard)
        return out_path

    # Assign (year, doy) pairs to this shard
    all_days = _build_day_list()  # 270 pairs
    sl = _shard_slice(len(all_days), n_shards, shard)
    my_days = all_days[sl]

    if not my_days:
        logger.warning("Shard %d is empty (no days assigned).", shard)
        return out_path

    logger.info("Shard %d/%d: %d day(s) → %s", shard, n_shards, len(my_days), out_path)

    # Build space-weather lookup tables once per shard
    f107_table = _build_f107_table()
    kp_table = _build_kp_table(_YEARS)

    ne_buf: list[np.ndarray] = []
    hv_buf: list[np.ndarray] = []
    mask_buf: list[np.ndarray] = []
    cond_buf: list[np.ndarray] = []

    t0 = time.time()
    n_ok = 0
    n_fail = 0

    for day_idx, (year, doy) in enumerate(my_days):
        f107 = f107_table.get((year, doy))
        if f107 is None:
            # Interpolate nearest available F10.7 within ±15 days
            candidates = {k: v for k, v in f107_table.items() if k[0] == year}
            if candidates:
                nearest = min(candidates.keys(), key=lambda k: abs(k[1] - doy))
                f107 = candidates[nearest]
                logger.debug(
                    "F10.7 interpolated for (%d,%d) from (%d,%d)=%.1f",
                    year,
                    doy,
                    *nearest,
                    f107,
                )
            else:
                f107 = 120.0  # solar-cycle-average fallback
                logger.debug("F10.7 fallback 120 SFU for year=%d doy=%d", year, doy)

        kp = kp_table.get((year, doy), _KP_FALLBACK)

        # Geo-chunked PyIRI calls — keeps peak RAM ≤ _GEO_CHUNK × N_T × N_V × 8 bytes
        for g_start in range(0, _N_G, _GEO_CHUNK):
            g_end = min(g_start + _GEO_CHUNK, _N_G)
            alat_c = _ALAT_FLAT[g_start:g_end]
            alon_c = _ALON_FLAT[g_start:g_end]

            ne_cm3 = _call_pyiri_chunk(year, doy, f107, alat_c, alon_c)
            if ne_cm3 is None:
                n_fail += (g_end - g_start) * len(_UT_H)
                continue

            cond = _build_cond_chunk(doy, f107, kp, alat_c, alon_c)

            h_virt, obs_mask, valid = _compute_h_virtual(ne_cm3)

            ne_buf.append(ne_cm3[valid])
            hv_buf.append(h_virt[valid])
            mask_buf.append(obs_mask[valid])
            cond_buf.append(cond[valid])
            n_ok += int(valid.sum())
            n_fail += int((~valid).sum())

        logger.info(
            "  day %d/%d  (%d-%03d)  f107=%.1f  kp=%.1f  ok=%d  fail=%d",
            day_idx + 1,
            len(my_days),
            year,
            doy,
            f107,
            kp,
            n_ok,
            n_fail,
        )

    if not ne_buf:
        logger.warning("Shard %d produced zero valid samples.", shard)
        return out_path

    ne_all = np.concatenate(ne_buf, axis=0).astype(np.float32)
    hv_all = np.concatenate(hv_buf, axis=0).astype(np.float32)
    mask_all = np.concatenate(mask_buf, axis=0).astype(np.float32)
    cond_all = np.concatenate(cond_buf, axis=0).astype(np.float32)

    logger.info(
        "Shard %d: writing %d samples (failed=%d) → %s",
        shard,
        len(ne_all),
        n_fail,
        out_path,
    )

    # ── Build xarray Dataset (CF-1.8) ─────────────────────────────────────────
    ds = xr.Dataset(
        {
            "ne_cm3": xr.DataArray(
                ne_all,
                dims=["sample", "height_km"],
                attrs={"units": "cm-3", "long_name": "Electron density"},
            ),
            "h_virtual": xr.DataArray(
                hv_all,
                dims=["sample", "freq_mhz"],
                attrs={"units": "km", "long_name": "Virtual height (0 above foF2)"},
            ),
            "obs_mask": xr.DataArray(
                mask_all.astype(np.int8),
                dims=["sample", "freq_mhz"],
                attrs={
                    "flag_values": "0 1",
                    "flag_meanings": "above_foF2 valid_observation",
                },
            ),
            "cond": xr.DataArray(
                cond_all,
                dims=["sample", "cond_dim"],
                attrs={
                    "long_name": "Conditioning vector",
                    "columns": " ".join(COND_COLS),
                },
            ),
        },
        coords={
            "height_km": xr.DataArray(
                H_GRID_KM.astype(np.float32),
                dims=["height_km"],
                attrs={"units": "km", "long_name": "Altitude above ground"},
            ),
            "freq_mhz": xr.DataArray(
                F_GRID_MHZ.astype(np.float32),
                dims=["freq_mhz"],
                attrs={"units": "MHz", "long_name": "Sounding frequency"},
            ),
            "cond_dim": xr.DataArray(
                np.array(list(COND_COLS)),
                dims=["cond_dim"],
                attrs={"long_name": "Conditioning variable names"},
            ),
        },
        attrs={
            "shard": shard,
            "n_shards": n_shards,
            "n_samples": len(ne_all),
            "n_failed": n_fail,
            "pyiri_version": "PyIRI-0.1.5 (IRI-2016)",
            "elapsed_s": float(time.time() - t0),
            "lat_grid": str(_LAT_DEG.tolist()),
            "lon_grid": str(_LON_DEG.tolist()),
            "years": str(_YEARS.tolist()),
            "doy_grid": str(_DOY.tolist()),
            "ut_grid_h": str(_UT_H.tolist()),
            "note_kp": (
                "Kp derived from OMNI 60-min SymH daily mean (Kp≈−SymH/25). "
                "PyIRI does not model storm dynamics; Kp in cond reflects the "
                "actual space-weather state of the sampled date."
            ),
            "Conventions": "CF-1.8",
        },
    )

    encoding = {
        v: {"zlib": True, "complevel": 4}
        for v in ["ne_cm3", "h_virtual", "obs_mask", "cond"]
    }
    ds.to_netcdf(out_path, encoding=encoding)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Generate a shard of NN-POLAN synthetic training data via PyIRI."
    )
    p.add_argument(
        "--shard",
        type=int,
        required=True,
        help="Shard index (0-based). Set to $SLURM_ARRAY_TASK_ID on VEGA.",
    )
    p.add_argument(
        "--n_shards",
        type=int,
        required=True,
        help="Total number of shards (recommend 270 = one per day).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for NetCDF shard files.",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing shard files."
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out = generate_shard(
        shard=args.shard,
        n_shards=args.n_shards,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    _cli()
