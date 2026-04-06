"""Coherent echo filtering for a single VIPIR sounding (PL407).

Demonstrates :class:`~pynasonde.vipir.riq.parsers.filter.IonogramFilter`
applied to one ``PL407_2024058061501.RIQ`` sounding.  The script compares
the raw echo cloud with the filtered cloud side-by-side and prints
per-stage rejection counts.

PL407 differences from WI937
------------------------------
PL407 uses ``vipir_version=1`` / ``data_type=2`` (``configs[0]``).
With only 2 receivers (n_rx=2) the direction-finding fit requires
``min_rx_for_direction=3`` so XL, YL, and EP are all NaN.
Consequently:

- **Stage 2 (EP filter)** is a no-op — all NaN EP values are passed through.
- **Stage 4 (DBSCAN)** uses only ``frequency_khz``, ``height_km``,
  ``velocity_mps``, and ``amplitude_db`` (``residual_deg`` dropped because
  it is entirely NaN).
- **Stage 5 (RANSAC)** operates in (f, h) space only — unaffected.

Output figure
-------------
A 2×2 comparison figure saved to
``docs/examples/figures/ionogram_filter_pl407.png``:

    (A) Raw ionogram   — echo cloud before filtering
    (B) Filtered ionogram — echo cloud after all stages
    (C) V* vs height   — raw (grey) vs kept (coloured by rejection stage)
    (D) Per-stage rejection bar chart

Update ``fname`` to point to your local copy of the file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.vipir.riq.echo import EchoExtractor
from pynasonde.vipir.riq.parsers.filter import IonogramFilter
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

fname = "examples/data/PL407_2024058061501.RIQ"
font_size = 12
setsize(font_size)

# ---------------------------------------------------------------------------
# Step 1: Load and extract echoes (PL407 — configs[0])
# ---------------------------------------------------------------------------

riq = RiqDataset.create_from_file(
    fname,
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[0],  # version 1 / data_type 2
)

extractor = EchoExtractor(
    sct=riq.sct,
    pulsets=riq.pulsets,
    snr_threshold_db=3.0,
    min_height_km=60.0,
    max_height_km=1000.0,
    min_rx_for_direction=3,  # PL407 has n_rx=2 → XL/YL/EP will be NaN
    max_echoes_per_pulset=5,
)
extractor.extract()

df_raw = extractor.to_dataframe()
print(f"Raw echoes : {len(df_raw)}")
print(
    f"  XL valid : {df_raw['xl_km'].notna().sum()} "
    f"(n_rx=2 < min_rx_for_direction=3 → expected 0)"
)
print(f"  EP valid : {df_raw['residual_deg'].notna().sum()}")

# ---------------------------------------------------------------------------
# Step 2: Configure and run IonogramFilter
# ---------------------------------------------------------------------------

filt = IonogramFilter(
    # Stage 1: RFI — three complementary checks for PL407 (no EP/XL/YL)
    # (a) Height IQR: RFI scatters echoes across all heights
    rfi_enabled=True,
    rfi_height_iqr_km=300.0,
    rfi_min_echoes=3,
    # Stage 2: EP filter — NaN EP for PL407 (no-op, but keep enabled)
    ep_filter_enabled=True,
    ep_max_deg=90.0,
    # Stage 3: Multi-hop
    multihop_enabled=True,
    multihop_orders=(2, 3),
    multihop_height_tol_km=50.0,
    multihop_snr_margin_db=6.0,
    # Stage 4: DBSCAN — drop residual_deg (all NaN for PL407)
    dbscan_enabled=True,
    dbscan_eps=1.0,
    dbscan_min_samples=5,
    dbscan_features=(
        "frequency_khz",
        "height_km",
        "velocity_mps",
        "amplitude_db",
    ),
    # Stage 5: RANSAC
    ransac_enabled=True,
    ransac_residual_km=100.0,
    ransac_min_samples=10,
    ransac_n_iter=200,
    ransac_poly_degree=3,
    ransac_min_inlier_fraction=0.3,
    # Stage 6: temporal coherence — disabled for single sounding
    temporal_enabled=False,
)

df_clean = filt.filter(extractor)
print(f"Filtered echoes: {len(df_clean)}")
print()
print(filt.summary())

# ---------------------------------------------------------------------------
# Step 3: Build per-echo stage labels for panel (C)
# ---------------------------------------------------------------------------

df_raw_si = df_raw.copy()
df_raw_si["sounding_index"] = 0

rfi_mask = filt._stage_rfi(df_raw_si)
ep_mask = filt._stage_ep(df_raw_si)
mh_mask = filt._stage_multihop(df_raw_si, rfi_mask & ep_mask)
db_mask = filt._stage_dbscan(df_raw_si, rfi_mask & ep_mask & mh_mask)
rs_mask = filt._stage_ransac(df_raw_si, rfi_mask & ep_mask & mh_mask & db_mask)

stage_labels = np.full(len(df_raw_si), "kept", dtype=object)
stage_labels[~rs_mask] = "RANSAC"
stage_labels[~db_mask] = "DBSCAN"
stage_labels[~mh_mask] = "Multi-hop"
stage_labels[~ep_mask] = "EP"
stage_labels[~rfi_mask] = "RFI"

stage_colors = {
    "kept": "tab:blue",
    "RFI": "tab:red",
    "EP": "tab:orange",
    "Multi-hop": "tab:purple",
    "DBSCAN": "tab:brown",
    "RANSAC": "tab:cyan",
}

# ---------------------------------------------------------------------------
# Step 4: Plot 2×2 comparison
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
fig.suptitle(
    "PL407  2024-058  06:15 UT — IonogramFilter: single-sounding  (vipir_version=1)",
    fontsize=font_size + 1,
)

freq_mhz_raw = df_raw["frequency_khz"] / 1e3
freq_mhz_clean = df_clean["frequency_khz"] / 1e3

amp_vmin = df_raw["amplitude_db"].quantile(0.05)
amp_vmax = df_raw["amplitude_db"].quantile(0.95)

# ── (A) Raw ionogram ─────────────────────────────────────────────────────────
ax = axes[0, 0]
sc = ax.scatter(
    freq_mhz_raw,
    df_raw["height_km"],
    c=df_raw["amplitude_db"],
    cmap="plasma",
    s=3,
    vmin=amp_vmin,
    vmax=amp_vmax,
    rasterized=True,
)
fig.colorbar(sc, ax=ax, pad=0.02).set_label("Amplitude (dB)", fontsize=font_size)
ax.set_xlabel("Frequency (MHz)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title(f"(A) Raw echo cloud  [{len(df_raw)} echoes]", fontsize=font_size)
ax.set_ylim(50, 1000)
ax.grid(True, alpha=0.3)

# ── (B) Filtered ionogram ────────────────────────────────────────────────────
ax = axes[0, 1]
if len(df_clean):
    sc2 = ax.scatter(
        freq_mhz_clean,
        df_clean["height_km"],
        c=df_clean["amplitude_db"],
        cmap="plasma",
        s=3,
        vmin=amp_vmin,
        vmax=amp_vmax,
        rasterized=True,
    )
    fig.colorbar(sc2, ax=ax, pad=0.02).set_label("Amplitude (dB)", fontsize=font_size)
else:
    ax.text(
        0.5, 0.5, "No echoes survived", ha="center", va="center", transform=ax.transAxes
    )
ax.set_xlabel("Frequency (MHz)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title(f"(B) Filtered echo cloud  [{len(df_clean)} echoes]", fontsize=font_size)
ax.set_ylim(50, 1000)
ax.grid(True, alpha=0.3)

# ── (C) V* vs height coloured by rejection stage ─────────────────────────────
ax = axes[1, 0]
v_valid = df_raw_si["velocity_mps"].notna()
for label, color in stage_colors.items():
    mask = v_valid & (stage_labels == label)
    if mask.sum() == 0:
        continue
    ax.scatter(
        df_raw_si.loc[mask, "velocity_mps"],
        df_raw_si.loc[mask, "height_km"],
        color=color,
        s=3,
        alpha=0.6,
        label=f"{label} ({mask.sum()})",
        rasterized=True,
    )
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("V* — phase-path velocity (m/s)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title("(C) V* per echo, coloured by rejection stage", fontsize=font_size)
ax.set_ylim(50, 1000)
ax.legend(fontsize=font_size - 3, loc="upper right", markerscale=2)
ax.grid(True, alpha=0.3)

# ── (D) Per-stage rejection bar chart ────────────────────────────────────────
ax = axes[1, 1]
stage_order = ["RFI", "EP", "Multi-hop", "DBSCAN", "RANSAC"]
counts = [
    int(filt.stats.get(k, {}).get("rejected", 0))
    for k in ["rfi", "ep", "multihop", "dbscan", "ransac"]
]
colors = [stage_colors[s] for s in stage_order]
bars = ax.barh(stage_order, counts, color=colors, edgecolor="k", lw=0.5)
ax.bar_label(bars, padding=3, fontsize=font_size - 1)
ax.set_xlabel("Echoes rejected", fontsize=font_size)
ax.set_title(
    f"(D) Rejection per stage\n"
    f"({len(df_raw)} raw → {len(df_clean)} kept, "
    f"{100*len(df_clean)/max(len(df_raw),1):.0f}% retained)",
    fontsize=font_size,
)
ax.grid(True, axis="x", alpha=0.3)

# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------

out = "docs/examples/figures/ionogram_filter_pl407.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
