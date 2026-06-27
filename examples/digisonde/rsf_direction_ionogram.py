"""RSF direction ionogram and daily directogram for KR835.

Produces two figures:
1. Direction-coded ionogram  – single ionogram, frequency vs height,
   colored by echo direction (mimics Figure 3-8).
2. Daily directogram         – a memory-safe subset of RSF files stacked by time,
   X = ground distance D_i (km), Y = UT time (mimics Figure 3-12).

Set ``RSF_EXAMPLE_MAX_FILES=0`` to process the full day, and tune
``RSF_EXAMPLE_N_PROCS`` for local hardware. The default is intentionally
small so the example runs in documentation and CI environments.
"""

import glob
import os

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from pynasonde.digisonde.digi_plots import RsfIonogram
from pynasonde.digisonde.parsers.rsf import RsfExtractor

STN = "WS833"
RSF_DIR = (
    "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/"
    "Digisonde Files/WSMR_DPS4D_2023_10_14/"
)
all_files = sorted(glob.glob(f"{RSF_DIR}/{STN}_*.RSF"))
all_files.sort()
if not all_files:
    raise FileNotFoundError(f"No RSF files found under {RSF_DIR!r}")
RSF_ONE = all_files[0]

# ── 1. Single ionogram: direction-coded plot ─────────────────────────────────
extractor = RsfExtractor(
    RSF_ONE, extract_time_from_name=True, extract_stn_from_name=True
)
extractor.extract()
df_one = extractor.to_pandas()

h = extractor.rsf_data.rsf_data_units[0].header
title_one = f"{STN}  {h.date.strftime('%Y-%m-%d  %H:%M:%S')} UT"

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
all_files = all_files[::4]
max_files = int(os.environ.get("RSF_EXAMPLE_MAX_FILES", "6"))
if max_files > 0:
    all_files = all_files[:max_files]
logger.info(f"Loading {len(all_files)} RSF files for daily directogram")


def _load_rsf(fpath: str) -> pd.DataFrame | None:
    """Parse one RSF file and return its DataFrame, or None on failure."""
    try:
        ex = RsfExtractor(fpath, extract_time_from_name=True)
        ex.extract()
        df = ex.to_pandas()
        return df[df.amplitude >= 30]
    except Exception as e:
        logger.warning(f"Skipped {fpath}: {e}")
        return None


N_PROCS = int(os.environ.get("RSF_EXAMPLE_N_PROCS", "1"))
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
    lower_plimit=10,
    ms=0.5,
    text=f"{STN}  2023-10-14",
)
r.save("tmp/rsf_directogram_KR835_daily.png")
r.save("docs/examples/figures/rsf_directogram_KR835_daily.png")
r.close()
