"""True-height inversion using TrueHeightInversion (lamination / Abel method).

Pipeline
--------
1. Load and filter echoes from WI937_2022233235902.RIQ.
2. Classify O/X modes via PolarizationClassifier.
3. Run TrueHeightInversion on the O-mode trace.
4. Print summary and EDP table.
5. Plot virtual h*(f) trace alongside the inverted true-height profile N(h).

Expected output
---------------
Figure saved to ``docs/examples/figures/analysis_inversion.png``.
"""

import os

import matplotlib.pyplot as plt
from _load_data import WI937, load_echoes

from pynasonde.vipir.analysis import PolarizationClassifier, TrueHeightInversion

# ── 1. Load real data ─────────────────────────────────────────────────────────
df, station = load_echoes(WI937)
print(f"[{station}]  Filtered echoes: {len(df)}")

# ── 2. O-mode labelling ───────────────────────────────────────────────────────
pol = PolarizationClassifier(o_mode_sign=-1)
ann_df = pol.fit(df).annotated_df
o_df = ann_df[ann_df["mode"] == "O"].copy()
print(f"O-mode echoes available: {len(o_df)}")

# ── 3. True-height inversion ──────────────────────────────────────────────────
inv = TrueHeightInversion(monotone_enforce=True, bin_width_mhz=5e-3)
edp = inv.fit_from_df(o_df)

print(edp.summary())
print("\nEDP table (first 10 rows):")
print(edp.to_dataframe().head(10).to_string(index=False))

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=False)

# Left: virtual h*(f) ionogram trace (O-mode)
ax = axes[0]
freq_mhz = o_df["frequency_khz"] / 1e3
ax.scatter(
    freq_mhz, o_df["height_km"], c="steelblue", s=5, alpha=0.5, label="O-mode echoes"
)
ax.set(
    xlabel="Frequency (MHz)",
    ylabel="Virtual height R' (km)",
    title=f"O-mode ionogram trace  [{station}]",
    ylim=(60, 800),
)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Right: true-height EDP
ax2 = axes[1]
if edp.n_layers > 0:
    ax2.plot(
        edp.plasma_freq_mhz,
        edp.true_height_km,
        "o-",
        color="steelblue",
        ms=5,
        label="fp(h) — true height",
    )
    ax2.plot(
        edp.plasma_freq_mhz,
        edp.virtual_height_km,
        "s--",
        color="grey",
        ms=4,
        alpha=0.6,
        label="h*(f) — virtual",
    )
    if not __import__("numpy").isnan(edp.hmF2_km):
        ax2.axhline(
            edp.hmF2_km,
            color="firebrick",
            lw=1,
            ls=":",
            label=f"hmF2={edp.hmF2_km:.0f} km",
        )
    if not __import__("numpy").isnan(edp.foF2_mhz):
        ax2.axvline(
            edp.foF2_mhz,
            color="firebrick",
            lw=1,
            ls=":",
            label=f"foF2={edp.foF2_mhz:.2f} MHz",
        )
else:
    ax2.text(
        0.5,
        0.5,
        "Insufficient O-mode echoes for inversion",
        ha="center",
        va="center",
        transform=ax2.transAxes,
    )
ax2.set(
    xlabel="Plasma frequency (MHz)",
    ylabel="Height (km)",
    title="True-height inversion (lamination)",
)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

fig.suptitle("TrueHeightInversion — Titheridge lamination method")
fig.tight_layout()
out = "docs/examples/figures/analysis_inversion.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
