"""O/X wave-mode separation using PolarizationClassifier.

Pipeline
--------
1. Load and filter echoes from WI937_2022233235902.RIQ.
2. Run :class:`~pynasonde.vipir.analysis.PolarizationClassifier`.
3. Print summary and per-mode echo counts.
4. Plot PP vs height coloured by mode label.

Expected output
---------------
A scatter plot saved to ``docs/examples/figures/analysis_polarization.png``
showing O-mode (blue), X-mode (red), and ambiguous (grey) echoes vs height.
"""

import os

import matplotlib.pyplot as plt
from _load_data import WI937, load_echoes

from pynasonde.vipir.analysis import PolarizationClassifier

# ── 1. Load real data ─────────────────────────────────────────────────────────
df, station = load_echoes(WI937)
print(f"[{station}]  Filtered echoes: {len(df)}")

# ── 2. Classify O/X mode ──────────────────────────────────────────────────────
clf = PolarizationClassifier(
    o_mode_sign=-1,  # northern hemisphere: O-mode has negative PP
    pp_ambiguous_threshold_deg=30.0,
)
result = clf.fit(df)

print(result.summary())
print(result.to_dataframe().head())

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(f"PolarizationClassifier — O/X mode separation  [{station}]")

ann = result.annotated_df
colors = {"O": "steelblue", "X": "firebrick", "ambiguous": "grey", "unknown": "black"}

for mode, grp in ann.groupby("mode"):
    axes[0].scatter(
        grp["polarization_deg"],
        grp["height_km"],
        c=colors.get(mode, "black"),
        s=5,
        alpha=0.5,
        label=mode,
    )
axes[0].axvline(0, color="k", lw=0.8, ls="--")
axes[0].set(
    xlabel="PP (degrees)",
    ylabel="Height (km)",
    title="PP vs height (coloured by mode)",
    xlim=(-180, 180),
    ylim=(60, 800),
)
axes[0].legend(fontsize=9, markerscale=3)
axes[0].grid(True, alpha=0.3)

# Bar chart of mode counts
counts = ann["mode"].value_counts()
bar_colors = [colors.get(m, "k") for m in counts.index]
axes[1].barh(counts.index, counts.values, color=bar_colors, alpha=0.8)
axes[1].set(xlabel="Echo count", title="Mode counts")
axes[1].grid(True, axis="x", alpha=0.3)

fig.tight_layout()
out = "docs/examples/figures/analysis_polarization.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
