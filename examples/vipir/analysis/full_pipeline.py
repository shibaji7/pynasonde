"""Full analysis pipeline — all six vipir/analysis modules chained together.

Demonstrates the complete post-trace-identification workflow using a real
VIPIR sounding (WI937_2022233235902.RIQ):

    PolarizationClassifier  →  label every echo O / X / ambiguous
    SpreadFAnalyzer         →  detect and classify spread-F
    TrueHeightInversion     →  convert virtual to true height (O-mode)
    IonogramScaler          →  derive foF2, foE, MUF(3000), M(3000)F2
    IrregularityAnalyzer    →  EP structure function and spectral index α
    NeXtYZInverter (Lite)   →  3-D WSI electron density profile + tilts

Pipeline
--------
1. Load and filter echoes from WI937_2022233235902.RIQ.
2. Run each analysis module in order.
3. Print one-line summaries for each result.
4. Produce a 3×2 diagnostic figure.

Expected output
---------------
Figure saved to ``docs/examples/figures/analysis_full_pipeline.png``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from _load_data import WI937, load_echoes

from pynasonde.vipir.analysis import (
    IonogramScaler,
    IrregularityAnalyzer,
    NeXtYZInverter,
    PolarizationClassifier,
    SpreadFAnalyzer,
    TrueHeightInversion,
)

# ── 1. Load real data ─────────────────────────────────────────────────────────
df, station = load_echoes(WI937)
print(f"[{station}]  Filtered echoes: {len(df)}")

# ── 2. PolarizationClassifier ─────────────────────────────────────────────────
pol_clf = PolarizationClassifier(o_mode_sign=-1, pp_ambiguous_threshold_deg=30.0)
pol_res = pol_clf.fit(df)
ann = pol_res.annotated_df
print(pol_res.summary())

# ── 3. SpreadFAnalyzer ────────────────────────────────────────────────────────
spread_res = SpreadFAnalyzer().fit(ann)
print(
    spread_res.summary()
    if hasattr(spread_res, "summary")
    else f"SpreadF: {spread_res.classification}"
)

# ── 4. TrueHeightInversion ────────────────────────────────────────────────────
edp = TrueHeightInversion(monotone_enforce=True).fit_from_df(ann[ann["mode"] == "O"])
print(edp.summary())

# ── 5. IonogramScaler ─────────────────────────────────────────────────────────
params = IonogramScaler(min_echoes_for_layer=4, n_bootstrap=200).fit(ann)
print(params.summary())

# ── 6. IrregularityAnalyzer ───────────────────────────────────────────────────
irreg = IrregularityAnalyzer(min_pairs_for_fit=3).fit(ann)
print(irreg.summary())

# ── 7. NeXtYZInverter (Lite) ──────────────────────────────────────────────────
nextyz_res = NeXtYZInverter(
    dip_angle_deg=66.0,
    declination_deg=11.0,
    B_gauss=0.55,
    fp_step_mhz=0.3,
    min_echoes=5,
    mode="Lite",
    fp_start_mhz=2.0,
).fit(ann)
print(nextyz_res.summary())

# ── 8. Diagnostic figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=True)
fig.suptitle(
    f"vipir/analysis — Full pipeline diagnostic\n({station})",
    fontsize=13,
)

freq_mhz = ann["frequency_khz"] / 1e3
mode_colors = {"O": "steelblue", "X": "firebrick", "ambiguous": "grey"}

# ── Panel A: Ionogram coloured by mode ────────────────────────────────────────
ax = axes[0, 0]
for mode, grp in ann.groupby("mode"):
    ax.scatter(
        freq_mhz[grp.index],
        grp["height_km"],
        c=mode_colors.get(mode, "k"),
        s=4,
        alpha=0.4,
        label=mode,
    )
ax.set(
    xlabel="Frequency (MHz)",
    ylabel="Height (km)",
    title="(A) Ionogram — O/X/ambiguous",
    ylim=(60, 800),
)
ax.legend(fontsize=8, markerscale=3)
ax.grid(True, alpha=0.3)

# ── Panel B: PP vs height ──────────────────────────────────────────────────────
ax = axes[0, 1]
for mode, grp in ann.groupby("mode"):
    ax.scatter(
        grp["polarization_deg"],
        grp["height_km"],
        c=mode_colors.get(mode, "k"),
        s=4,
        alpha=0.4,
        label=mode,
    )
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set(
    xlabel="PP (degrees)",
    ylabel="Height (km)",
    title="(B) PP vs height",
    xlim=(-180, 180),
    ylim=(60, 800),
)
ax.legend(fontsize=8, markerscale=3)
ax.grid(True, alpha=0.3)

# ── Panel C: True-height EDP ──────────────────────────────────────────────────
ax = axes[1, 0]
if edp.n_layers > 0:
    ax.plot(
        edp.plasma_freq_mhz,
        edp.true_height_km,
        "o-",
        color="steelblue",
        ms=5,
        label="fp(h) true height",
    )
    ax.plot(
        edp.plasma_freq_mhz,
        edp.virtual_height_km,
        "s--",
        color="grey",
        ms=4,
        alpha=0.5,
        label="h*(f) virtual",
    )
    if not np.isnan(edp.foF2_mhz):
        ax.axvline(
            edp.foF2_mhz,
            color="firebrick",
            lw=1,
            ls=":",
            label=f"foF2={edp.foF2_mhz:.2f} MHz",
        )
else:
    ax.text(
        0.5,
        0.5,
        "Insufficient O-mode echoes",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
ax.set(
    xlabel="Plasma frequency (MHz)",
    ylabel="Height (km)",
    title="(C) TrueHeightInversion — EDP",
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel D: Scaled parameters bar ────────────────────────────────────────────
ax = axes[1, 1]
params.plot(ax=ax)
ax.set_title("(D) IonogramScaler — foE, foF2, MUF(3000)")

# ── Panel E: EP structure function ────────────────────────────────────────────
ax = axes[2, 0]
sf = irreg.structure_function
valid = sf["D_EP_deg2"].notna() & (sf["D_EP_deg2"] > 0)
if valid.any():
    ax.loglog(
        sf.loc[valid, "delta_f_mhz"],
        sf.loc[valid, "D_EP_deg2"],
        "o",
        ms=5,
        color="steelblue",
        label="D_EP observed",
    )
    lag_fit = np.linspace(
        sf.loc[valid, "delta_f_mhz"].min(), sf.loc[valid, "delta_f_mhz"].max(), 80
    )
    ax.loglog(
        lag_fit,
        irreg.amplitude_coeff * lag_fit**irreg.spectral_index,
        "-",
        lw=2,
        color="firebrick",
        label=f"α={irreg.spectral_index:.3f}",
    )
else:
    ax.text(
        0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes
    )
ax.set(
    xlabel="Freq lag Δf (MHz)",
    ylabel="D_EP (deg²)",
    title="(E) IrregularityAnalyzer — EP structure function",
)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

# ── Panel F: NeXtYZ fp(h) with errors ─────────────────────────────────────────
ax = axes[2, 1]
if len(nextyz_res.h_true_km) > 0:
    ax.errorbar(
        nextyz_res.fp_profile_mhz,
        nextyz_res.h_true_km,
        yerr=nextyz_res.h_errors_km,
        fmt="o-",
        color="steelblue",
        ecolor="lightblue",
        capsize=3,
        label="NeXtYZ Lite",
    )
else:
    ax.text(
        0.5,
        0.5,
        "Insufficient echoes for inversion",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
ax.set(
    xlabel="Plasma frequency (MHz)",
    ylabel="True height (km)",
    title="(F) NeXtYZ Lite — 3-D WSI inversion",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

out = "docs/examples/figures/analysis_full_pipeline.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
