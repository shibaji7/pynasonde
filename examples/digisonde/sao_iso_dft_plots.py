"""Daily isodensity contour + DFT Doppler waterfall/spectra for KR835.

Produces three figures:

1. Isodensity contour  – time vs virtual height colored by plasma frequency
   (mimics Digisonde-Isodensity.gif from the Digisonde website).
2. Doppler waterfall   – Doppler bin vs height, amplitude color, single DFT block.
3. Doppler spectra     – amplitude vs Doppler bin, one line per height.
"""

import glob

import pandas as pd
from loguru import logger

from pynasonde.digisonde.digi_plots import SaoSummaryPlots, SkySummaryPlots
from pynasonde.digisonde.parsers.dft import DftExtractor
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.digi_utils import setsize

SAO_DIR = (
    "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/"
    "Digisonde Files/SKYWAVE_DPS4D_2023_10_14"
)
DFT_FILE = f"{SAO_DIR}/KR835_2023287000915.DFT"
font_size = 18
setsize(font_size)

# ── 1. Daily isodensity contours from all SAO files ──────────────────────────
logger.info("Loading SAO files for isodensity contours…")
df_sao = SaoExtractor.load_SAO_files(
    folders=[SAO_DIR],
    ext="KR835_*.SAO",
    n_procs=8,
    func_name="height_profile",
)
df_sao = df_sao.reset_index(drop=True)
df_sao["datetime"] = pd.to_datetime(df_sao["datetime"])
logger.info(f"SAO total records: {len(df_sao)}")

p = SaoSummaryPlots(figsize=(10, 4), font_size=10)
p.add_isodensity_contours(
    df_sao,
    xparam="datetime",
    yparam="th",
    zparam="pf",
    ylim=[50, 500],
    fbins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    text="KR835  2023-10-14",
)
p.save("docs/examples/figures/sao_isodensity_KR835.png")
p.close()
logger.info("Saved docs/examples/figures/sao_isodensity_KR835.png")

# ── 2. DFT Doppler waterfall ─────────────────────────────────────────────────
logger.info(f"Loading DFT file: {DFT_FILE}")
dft = DftExtractor(DFT_FILE, extract_time_from_name=True, extract_stn_from_name=True)
dft.extract()
df_dft = dft.to_pandas()
logger.info(f"DFT total records: {len(df_dft)}")
title_dft = f"KR835  {dft.date.strftime('%Y-%m-%d  %H:%M:%S')} UT"

sk = SkySummaryPlots(figsize=(8, 5), font_size=font_size, subplot_kw={})
sk.plot_doppler_waterfall(
    df_dft,
    cmap="inferno",
    text=title_dft,
)
sk.save("docs/examples/figures/dft_doppler_waterfall_KR835.png")
sk.close()
logger.info("Saved docs/examples/figures/dft_doppler_waterfall_KR835.png")

# ── 3. DFT Doppler spectra ───────────────────────────────────────────────────
sk2 = SkySummaryPlots(figsize=(12, 6), font_size=font_size, subplot_kw={})
sk2.plot_doppler_spectra(
    df_dft,
    n_heights=8,
    cmap="viridis",
    text=title_dft,
)
sk2.save("docs/examples/figures/dft_doppler_spectra_KR835.png")
sk2.close()
logger.info("Saved docs/examples/figures/dft_doppler_spectra_KR835.png")
