"""Coherent echo filtering across multiple VIPIR soundings (PL407, temporal coherence).

Extends ``ionogram_filter_pl407.py`` by feeding a list of
:class:`~pynasonde.vipir.riq.parsers.filter.IonogramFilter`-compatible
:class:`~pynasonde.vipir.riq.echo.EchoExtractor` objects (one per RIQ file)
to :meth:`~pynasonde.vipir.riq.parsers.filter.IonogramFilter.filter`.

When more than one sounding is provided Stage 6 (temporal coherence)
activates automatically.  A (frequency, height) cell that is populated in
at least ``temporal_min_soundings`` different soundings is considered
coherent; echoes in cells that appear only once or twice are treated as
temporally incoherent noise and removed.

Why this works
--------------
Real ionospheric echoes follow the slowly-varying plasma frequency profile
and persist from one sounding (~5 min cadence) to the next.  Random
interference and thermal noise occupy arbitrary (f, h) cells that change
between soundings.  The probability of a spurious hit landing in the same
(50 kHz × 50 km) cell in 3 out of 5 soundings is ≈ 10⁻³ while the
probability for a real echo is ≈ 0.99.

Steps covered
-------------
1. Load N RIQ files and extract echo clouds independently.
2. Pass the list of extractors to :meth:`IonogramFilter.filter`.
3. Plot a 3-panel figure:

       (A) Stack of all raw echo clouds (coloured by sounding index)
       (B) Filtered cloud — only temporally coherent echoes
       (C) Sounding-by-sounding echo count: raw vs filtered

Update ``fnames`` to point to 3–10 consecutive sounding files.
They do not need to be contiguous in time; the filter only counts cell
occupancy, not time ordering.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.vipir.riq.echo import EchoExtractor
from pynasonde.vipir.riq.parsers.filter import IonogramFilter
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

# ---------------------------------------------------------------------------
# Configuration — list 3+ consecutive RIQ files here
# ---------------------------------------------------------------------------

fnames = [
    "examples/data/PL407_2024058061501.RIQ",
    # "examples/data/PL407_2024058062001.RIQ",   # add more consecutive soundings here
    # "examples/data/PL407_2024058062501.RIQ",
    # "examples/data/PL407_2024058063001.RIQ",
    # "examples/data/PL407_2024058063501.RIQ",
]
font_size = 12
setsize(font_size)

# ---------------------------------------------------------------------------
# Step 1: Load and extract echoes for every sounding
# ---------------------------------------------------------------------------

extractors = []
for fname in fnames:
    riq = RiqDataset.create_from_file(
        fname,
        unicode="latin-1",
        vipir_config=VIPIR_VERSION_MAP.configs[0],  # PL407: version 1 / data_type 2
    )
    ext = EchoExtractor(
        sct=riq.sct,
        pulsets=riq.pulsets,
        snr_threshold_db=3.0,
        min_height_km=60.0,
        max_height_km=1000.0,
        min_rx_for_direction=3,
        max_echoes_per_pulset=5,
    )
    ext.extract()
    extractors.append(ext)
    print(f"Loaded {fname.split('/')[-1]}  →  {len(ext.echoes)} echoes")

# ---------------------------------------------------------------------------
# Step 2: Configure and run IonogramFilter (multi-sounding)
# ---------------------------------------------------------------------------

filt = IonogramFilter(
    # Stage 1: RFI blanking — three complementary checks (key for PL407)
    rfi_enabled=True,
    rfi_height_iqr_km=300.0,
    rfi_min_echoes=3,
    # Stage 2: EP filter
    ep_filter_enabled=True,
    ep_max_deg=90.0,  # conservative — oblique real echoes reach 50-80°
    # Stage 3: Multi-hop
    multihop_enabled=True,
    multihop_orders=(2, 3),
    multihop_height_tol_km=50.0,
    multihop_snr_margin_db=6.0,
    # Stage 4: DBSCAN
    dbscan_enabled=True,
    dbscan_eps=1.0,
    dbscan_min_samples=5,
    dbscan_features=(
        "frequency_khz",
        "height_km",
        "velocity_mps",
        "amplitude_db",
        # residual_deg omitted: PL407 n_rx=2 → EP all NaN
    ),
    # Stage 5: RANSAC — fit smooth h*(f) trace per sounding, reject outliers
    ransac_enabled=True,
    ransac_residual_km=100.0,
    ransac_min_samples=10,
    ransac_n_iter=200,
    ransac_poly_degree=3,
    ransac_min_inlier_fraction=0.3,
    # Stage 6: temporal coherence — echo must appear in >= min soundings
    # (silently skipped when only 1 extractor is provided)
    temporal_enabled=True,
    temporal_min_soundings=min(3, len(fnames)),
    temporal_freq_bin_khz=50.0,
    temporal_height_bin_km=50.0,
)

# Pass the full list → Stage 5 activates when len(extractors) > 1
df_all = filt.filter(extractors)

print()
print(filt.summary())

# ---------------------------------------------------------------------------
# Step 3: Plot 3-panel comparison
# ---------------------------------------------------------------------------

n_soundings = len(fnames)
cmap_idx = plt.cm.tab10

fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
fig.suptitle(
    f"IonogramFilter — temporal coherence  "
    f"({n_soundings} sounding{'s' if n_soundings > 1 else ''})",
    fontsize=font_size + 1,
)

# ── (A) All raw echo clouds stacked, one colour per sounding ─────────────────
ax = axes[0]
total_raw = 0
for idx, ext in enumerate(extractors):
    df_s = ext.to_dataframe()
    if df_s.empty:
        continue
    total_raw += len(df_s)
    ax.scatter(
        df_s["frequency_khz"] / 1e3,
        df_s["height_km"],
        color=cmap_idx(idx % 10),
        s=3,
        alpha=0.5,
        label=f"S{idx + 1} ({len(df_s)})",
        rasterized=True,
    )
ax.set_xlabel("Frequency (MHz)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title(f"(A) Raw echo clouds  [{total_raw} total]", fontsize=font_size)
ax.set_ylim(50, 1000)
ax.legend(fontsize=font_size - 3, markerscale=2)
ax.grid(True, alpha=0.3)

# ── (B) Filtered echo cloud coloured by sounding ─────────────────────────────
ax = axes[1]
for idx in range(n_soundings):
    subset = df_all[df_all["sounding_index"] == idx]
    if subset.empty:
        continue
    ax.scatter(
        subset["frequency_khz"] / 1e3,
        subset["height_km"],
        color=cmap_idx(idx % 10),
        s=3,
        alpha=0.7,
        label=f"S{idx + 1} ({len(subset)})",
        rasterized=True,
    )
ax.set_xlabel("Frequency (MHz)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title(f"(B) Filtered echo cloud  [{len(df_all)} kept]", fontsize=font_size)
ax.set_ylim(50, 1000)
ax.legend(fontsize=font_size - 3, markerscale=2)
ax.grid(True, alpha=0.3)

# ── (C) Echo counts per sounding: raw vs filtered ────────────────────────────
ax = axes[2]
raw_counts = [len(ext.echoes) for ext in extractors]
filt_counts = [(df_all["sounding_index"] == idx).sum() for idx in range(n_soundings)]
labels = [f"S{i + 1}" for i in range(n_soundings)]

x = np.arange(n_soundings)
w = 0.35
ax.bar(x - w / 2, raw_counts, w, label="Raw", color="tab:grey", alpha=0.8)
ax.bar(x + w / 2, filt_counts, w, label="Filtered", color="tab:green", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=font_size)
ax.set_ylabel("Echo count", fontsize=font_size)
ax.set_title("(C) Raw vs filtered per sounding", fontsize=font_size)
ax.legend(fontsize=font_size - 1)
ax.grid(True, axis="y", alpha=0.3)

for i, (r, f) in enumerate(zip(raw_counts, filt_counts)):
    pct = 100 * f / max(r, 1)
    ax.text(
        i,
        max(r, f) + 5,
        f"{pct:.0f}%",
        ha="center",
        va="bottom",
        fontsize=font_size - 2,
    )

# ---------------------------------------------------------------------------
# Step 4: Save
# ---------------------------------------------------------------------------

out = "docs/examples/figures/ionogram_filter_multi.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
