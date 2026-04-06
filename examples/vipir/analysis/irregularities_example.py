"""Irregularity spectral-index analysis using IrregularityAnalyzer.

Pipeline
--------
1. Load and filter echoes from both WI937 and PL407 RIQ files.
2. Classify O/X modes via PolarizationClassifier.
3. Run IrregularityAnalyzer on each labelled DataFrame.
4. Print spectral index α and anisotropy metrics for each sounding.
5. Plot the EP structure function D_EP(Δf) with power-law fit side-by-side.

Expected output
---------------
Figure saved to ``docs/examples/figures/analysis_irregularities.png``.

Physical interpretation
-----------------------
The spectral index α from D_EP(Δf) ∝ Δf^α characterises the electron-
density irregularity power spectrum.  Typical values:
    α ≈ 1.5 – 2.0  →  moderate irregularity (quiet to moderately disturbed)
    α < 1.0        →  strong small-scale irregularity (spread-F, scintillation)
    α > 2.5        →  smooth plasma with few small-scale structures
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from _load_data import PL407, WI937, load_echoes

from pynasonde.vipir.analysis import IrregularityAnalyzer, PolarizationClassifier

# ── 1. Load both soundings ────────────────────────────────────────────────────
df_wi, label_wi = load_echoes(WI937)
df_pl, label_pl = load_echoes(PL407)
print(f"[{label_wi}]  Filtered echoes: {len(df_wi)}")
print(f"[{label_pl}]  Filtered echoes: {len(df_pl)}")

# ── 2. O/X labelling ──────────────────────────────────────────────────────────
pol = PolarizationClassifier(o_mode_sign=-1)
ann_wi = pol.fit(df_wi).annotated_df
ann_pl = pol.fit(df_pl).annotated_df

# ── 3. Irregularity analysis ──────────────────────────────────────────────────
analyzer = IrregularityAnalyzer(min_pairs_for_fit=3)
res_wi = analyzer.fit(ann_wi)
res_pl = analyzer.fit(ann_pl)

for label, res in [(label_wi, res_wi), (label_pl, res_pl)]:
    print(f"\n=== {label} ===")
    print(res.summary())

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, res, label in zip(axes, [res_wi, res_pl], [label_wi, label_pl]):
    sf = res.structure_function
    valid = sf["D_EP_deg2"].notna() & (sf["D_EP_deg2"] > 0)
    if valid.any():
        ax.loglog(
            sf.loc[valid, "delta_f_mhz"],
            sf.loc[valid, "D_EP_deg2"],
            "o",
            ms=6,
            color="steelblue",
            label="D_EP observed",
        )
        lag_fit = np.linspace(
            sf.loc[valid, "delta_f_mhz"].min(), sf.loc[valid, "delta_f_mhz"].max(), 80
        )
        ax.loglog(
            lag_fit,
            res.amplitude_coeff * lag_fit**res.spectral_index,
            "-",
            lw=2,
            color="firebrick",
            label=f"Fit α={res.spectral_index:.3f}",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data for structure function",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set(
        xlabel="Frequency lag Δf (MHz)",
        ylabel="D_EP (deg²)",
        title=f"{label} — EP Structure Function",
    )
    if not np.isnan(res.anisotropy_ratio):
        ax.text(
            0.97,
            0.05,
            f"anisotropy ratio={res.anisotropy_ratio:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="grey"),
        )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

fig.suptitle("IrregularityAnalyzer — EP structure function and spectral index")
fig.tight_layout()
out = "docs/examples/figures/analysis_irregularities.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
