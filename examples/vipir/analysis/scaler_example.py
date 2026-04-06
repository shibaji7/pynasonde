"""Automatic ionogram parameter scaling using IonogramScaler.

Pipeline
--------
1. Load and filter echoes from WI937_2022233235902.RIQ.
2. Classify O/X modes via PolarizationClassifier.
3. Run IonogramScaler on the labelled DataFrame.
4. Print foE, foF2, h'F2, MUF(3000), M(3000)F2 with uncertainties.
5. Plot a bar chart of the key scaled parameters.

Expected output
---------------
Figure saved to ``docs/examples/figures/analysis_scaler.png``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from _load_data import WI937, load_echoes

from pynasonde.vipir.analysis import IonogramScaler, PolarizationClassifier

# ── 1. Load real data ─────────────────────────────────────────────────────────
df, station = load_echoes(WI937)
print(f"[{station}]  Filtered echoes: {len(df)}")

# ── 2. O/X labelling ──────────────────────────────────────────────────────────
ann = PolarizationClassifier(o_mode_sign=-1).fit(df).annotated_df

# ── 3. Scale parameters ───────────────────────────────────────────────────────
scaler = IonogramScaler(
    e_layer_height_range_km=(90, 160),
    f2_layer_height_range_km=(160, 800),
    n_bootstrap=300,
    min_echoes_for_layer=4,
)
params = scaler.fit(ann)

print(params.summary())
print("\nScaled parameters DataFrame:")
print(params.to_dataframe().to_string(index=False))
print(f"\nQuality flags: {params.quality_flags}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: ionogram with layer identifications overlaid
ax = axes[0]
freq_mhz = ann["frequency_khz"] / 1e3
colors = {"O": "steelblue", "X": "firebrick", "ambiguous": "grey"}
for mode, grp in ann.groupby("mode"):
    ax.scatter(
        freq_mhz[grp.index],
        grp["height_km"],
        c=colors.get(mode, "k"),
        s=4,
        alpha=0.4,
        label=mode,
    )

if not np.isnan(params.foE_mhz):
    ax.axvline(
        params.foE_mhz,
        color="orange",
        lw=1.5,
        ls="--",
        label=f"foE={params.foE_mhz:.2f} MHz",
    )
if not np.isnan(params.foF2_mhz):
    ax.axvline(
        params.foF2_mhz,
        color="green",
        lw=1.5,
        ls="--",
        label=f"foF2={params.foF2_mhz:.2f} MHz",
    )
if not np.isnan(params.h_prime_F2_km):
    ax.axhline(
        params.h_prime_F2_km,
        color="purple",
        lw=1,
        ls=":",
        label=f"h'F2={params.h_prime_F2_km:.0f} km",
    )

ax.set(
    xlabel="Frequency (MHz)",
    ylabel="Height (km)",
    title=f"Ionogram with scaled parameters  [{station}]",
    ylim=(60, 800),
)
ax.legend(fontsize=8, markerscale=3)
ax.grid(True, alpha=0.3)

# Right: parameter bar chart from ScaledParameters.plot()
params.plot(ax=axes[1])

fig.suptitle("IonogramScaler — foF2, foE, MUF(3000)")
fig.tight_layout()
out = "docs/examples/figures/analysis_scaler.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
