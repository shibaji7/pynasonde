"""Read a Digisonde MMM (ModMax) file and produce ionogram PNGs.

Two figures are saved:

1. ``mmm_ionogram.png``     — combined O+X, pcolormesh, amplitude-coloured
2. ``mmm_ionogram_OX.png``  — side-by-side O-mode / X-mode pcolormesh

Data are binned onto a regular frequency × height grid; the maximum
amplitude across all Doppler channels is taken per cell (brightest-echo
convention, matching SAOExplorer display).

Polarisation (O/X) and Doppler channel are decoded per range bin from
the lower nibble of each data byte (DPS4D convention):
    bit 3    → polarisation: 0 = O,  1 = X
    bits 2-0 → Doppler bin 0-7

Usage (from repo root):
    conda run -n pynasonde python examples/digisonde/mmm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.digisonde.parsers.mmm import ModMaxExtractor

MMM_FILE = "/home/chakras4/Research/ERAUCodeBase/apep_eclipse/AU930_2017147000005.MMM"
OUT_DIR = Path("docs/examples/figures")
FONT_SIZE = 14
NOISE_FLOOR = 10   # dB — discard bins at or below this amplitude
VMIN, VMAX = 10, 84

setsize(FONT_SIZE)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load and parse ─────────────────────────────────────────────────────────────
logger.info(f"Loading: {MMM_FILE}")
ext = ModMaxExtractor(MMM_FILE, extract_time_from_name=True, extract_stn_from_name=True)
ext.extract()
df = ext.to_pandas()

if df.empty:
    logger.error("No data extracted — check file path and format.")
    sys.exit(1)

logger.info(f"Frequency range : {df.frequency_mhz.min():.2f} – {df.frequency_mhz.max():.2f} MHz")
logger.info(f"Height range    : {df.range_km.min():.0f} – {df.range_km.max():.0f} km")
logger.info(f"Polarisation    : {df.polarization.value_counts().to_dict()}")
logger.info(f"Doppler channels: {sorted(df.doppler_channel.unique())}")

df = df[df["amplitude_dB"] > NOISE_FLOOR]
logger.info(f"After noise filter (>{NOISE_FLOOR} dB): {len(df)} points")

dtime = df["datetime"].iloc[0]
title_base = f"AU930  {dtime.strftime('%Y DOY%j  %H:%M')} UT"

# ── Grid axes (from actual sounding values) ───────────────────────────────────
freq_bins   = np.sort(df["frequency_mhz"].unique())
height_bins = np.sort(df["range_km"].unique())
F, H = np.meshgrid(freq_bins, height_bins)


def _to_grid(sub, fill=np.nan):
    """Pivot sub-DataFrame to (height × frequency) max-amplitude grid.

    Args:
        fill: value for empty cells (use VMIN to render as noise floor colour).
    """
    grid = (
        sub.pivot_table(
            index="range_km", columns="frequency_mhz",
            values="amplitude_dB", aggfunc="max",
        )
        .reindex(index=height_bins, columns=freq_bins)
        .values
    )
    if fill is not np.nan:
        grid = np.where(np.isnan(grid), fill, grid)
    return grid


def _style(ax):
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 7, 10, 13])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(0.9, 14)
    ax.set_ylim(50, 600)
    ax.set_xlabel("Frequency (MHz)", fontsize=FONT_SIZE)
    ax.set_ylabel("Virtual Height (km)", fontsize=FONT_SIZE)


# ── Figure 1: combined O+X ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
pc = ax.pcolormesh(F, H, _to_grid(df),
                   cmap="plasma", vmin=VMIN, vmax=VMAX, shading="nearest")
_style(ax)
ax.set_title(f"MMM Ionogram (O+X) — {title_base}", fontsize=FONT_SIZE)
plt.colorbar(pc, ax=ax, label="Amplitude (dB)", pad=0.02)
out1 = OUT_DIR / "mmm_ionogram.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info(f"Saved: {out1}")

# ── Figure 2: O-mode / X-mode side by side ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.subplots_adjust(wspace=0.35)

for ax, (pol, cmap, label) in zip(axes, [
    ("O", "plasma", "O-mode"),
    ("X", "viridis", "X-mode"),
]):
    sub = df[df["polarization"] == pol]
    pc = ax.pcolormesh(F, H, _to_grid(sub, fill=VMIN),
                       cmap=cmap, vmin=VMIN, vmax=VMAX, shading="nearest")
    _style(ax)
    ax.set_title(f"{label}  ({len(sub):,} pts)", fontsize=FONT_SIZE)
    plt.colorbar(pc, ax=ax, label="Amplitude (dB)", pad=0.02)

fig.suptitle(f"MMM Ionogram — {title_base}", fontsize=FONT_SIZE + 1)
out2 = OUT_DIR / "mmm_ionogram_OX.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info(f"Saved: {out2}")
