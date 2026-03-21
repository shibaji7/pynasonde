"""RSF direction ionogram and daily directogram for KR835.

Produces two figures:
1. Direction-coded ionogram  – single ionogram, frequency vs height,
   colored by echo direction (mimics Figure 3-8).
2. Daily directogram         – all RSF files for 2023-10-14 stacked by time,
   X = ground distance D_i (km), Y = UT time (mimics Figure 3-12).
"""

import glob

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from pynasonde.digisonde.digi_plots import RsfIonogram
from pynasonde.digisonde.parsers.rsf import RsfExtractor

RSF_DIR = (
    "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/"
    "Digisonde Files/SKYWAVE_DPS4D_2023_10_14"
)
RSF_ONE = f"{RSF_DIR}/KR835_2023287000000.RSF"

# ── 1. Single ionogram: direction-coded plot ─────────────────────────────────
extractor = RsfExtractor(
    RSF_ONE, extract_time_from_name=True, extract_stn_from_name=True
)
extractor.extract()
df_one = extractor.to_pandas()

h = extractor.rsf_data.rsf_data_units[0].header
title_one = f"KR835  {h.date.strftime('%Y-%m-%d  %H:%M:%S')} UT"

r = RsfIonogram(figsize=(6, 5), font_size=10)
r.add_direction_ionogram(
    df_one,
    ylim=[80, 600],
    xlim=[1, 15],
    xticks=[1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
    text=title_one,
    lower_plimit=5,
    ms=1.0,
)
r.save("tmp/rsf_direction_ionogram_KR835.png")
r.save("docs/examples/figures/rsf_direction_ionogram_KR835.png")
r.close()

# ── 2. Daily directogram: all RSF files for the day ─────────────────────────
all_files = sorted(glob.glob(f"{RSF_DIR}/KR835_*.RSF"))
all_files = all_files[::4]
logger.info(f"Loading {len(all_files)} RSF files for daily directogram")

def _load_rsf(fpath: str) -> pd.DataFrame | None:
    """Parse one RSF file and return its DataFrame, or None on failure."""
    try:
        ex = RsfExtractor(fpath, extract_time_from_name=True)
        ex.extract()
        return ex.to_pandas()
    except Exception as e:
        logger.warning(f"Skipped {fpath}: {e}")
        return None


N_PROCS = 2   # tune to the number of available CPU cores
results = Parallel(n_jobs=N_PROCS, backend="loky")(
    delayed(_load_rsf)(fpath) for fpath in all_files
)

frames = [df for df in results if df is not None]

df_day = pd.concat(frames, ignore_index=True)
logger.info(f"Total records for daily directogram: {len(df_day)}")

r = RsfIonogram(figsize=(6, 8), font_size=10)
r.add_directogram(
    df_day,
    dlim=[-800, 800],
    lower_plimit=5,
    ms=1.5,
    text="KR835  2023-10-14",
)
r.save("tmp/rsf_directogram_KR835_daily.png")
r.save("docs/examples/figures/rsf_directogram_KR835_daily.png")
r.close()
