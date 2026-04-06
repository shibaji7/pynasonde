"""Spread-F detection using SpreadFAnalyzer.

Pipeline
--------
1. Load and filter echoes from both available RIQ files (WI937 and PL407).
2. Run :class:`~pynasonde.vipir.analysis.PolarizationClassifier` on each.
3. Run :class:`~pynasonde.vipir.analysis.SpreadFAnalyzer` on each.
4. Print classifications and compare height IQR / frequency spread metrics.
5. Plot side-by-side ionograms annotated with classification.

Expected output
---------------
Figure saved to ``docs/examples/figures/analysis_spread_f.png``.
"""

import os

import matplotlib.pyplot as plt
from _load_data import PL407, WI937, load_echoes

from pynasonde.vipir.analysis import PolarizationClassifier, SpreadFAnalyzer

# ── 1. Load both soundings ────────────────────────────────────────────────────
df_wi, label_wi = load_echoes(WI937)
df_pl, label_pl = load_echoes(PL407)
print(f"[{label_wi}]  Filtered echoes: {len(df_wi)}")
print(f"[{label_pl}]  Filtered echoes: {len(df_pl)}")

# ── 2. Polarization labelling ─────────────────────────────────────────────────
pol = PolarizationClassifier(o_mode_sign=-1)
ann_wi = pol.fit(df_wi).annotated_df
ann_pl = pol.fit(df_pl).annotated_df

# ── 3. Spread-F analysis ──────────────────────────────────────────────────────
analyzer = SpreadFAnalyzer(
    height_spread_threshold_km=40.0,
    freq_spread_threshold_mhz=0.5,
)
res_wi = analyzer.fit(ann_wi)
res_pl = analyzer.fit(ann_pl)

for label, res in [(label_wi, res_wi), (label_pl, res_pl)]:
    print(f"\n=== {label} ===")
    print(f"  Classification : {res.classification}")
    print(f"  Height IQR     : {res.height_iqr_km:.1f} km")
    print(f"  Freq spread    : {res.freq_spread_mhz:.2f} MHz")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, ann, res, label in zip(
    axes,
    [ann_wi, ann_pl],
    [res_wi, res_pl],
    [label_wi, label_pl],
):
    freq_mhz = ann["frequency_khz"] / 1e3
    sc = ax.scatter(
        freq_mhz,
        ann["height_km"],
        c=ann["amplitude_db"],
        cmap="plasma",
        s=4,
        alpha=0.6,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("A (dB)", fontsize=9)
    ax.set(
        xlabel="Frequency (MHz)",
        ylabel="Height (km)",
        title=f"{label}  —  [{res.classification}]",
        ylim=(60, 800),
    )
    ax.text(
        0.03,
        0.97,
        f"IQR={res.height_iqr_km:.0f} km\nΔf={res.freq_spread_mhz:.2f} MHz",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="grey"),
    )
    ax.grid(True, alpha=0.3)

fig.suptitle("SpreadFAnalyzer — WI937 vs PL407")
fig.tight_layout()
out = "docs/examples/figures/analysis_spread_f.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
