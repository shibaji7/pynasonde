"""HF radio absorption estimation using AbsorptionAnalyzer.

Pipeline
--------
1. Load and filter echoes from WI937_2022233235902.RIQ.
2. Classify O/X modes via PolarizationClassifier.
3. Invert the O-mode trace to an EDP via TrueHeightInversion.
4. Run all four AbsorptionAnalyzer methods:
   a. lof_absorption    — LOF-based A3 index (no calibration needed).
   b. differential_absorption — ΔL(f) = SNR_O − SNR_X per frequency bin.
   c. total_absorption  — Calibrated one-way L(f) from the radar equation.
   d. absorption_profile — κ(z) and cumulative L(z) using an empirical ν(h).
5. Print summaries and save a 4-panel figure.

Expected output
---------------
Figure saved to ``docs/examples/figures/analysis_absorption.png``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from _load_data import WI937, load_echoes

from pynasonde.vipir.analysis import (
    AbsorptionAnalyzer,
    PolarizationClassifier,
    TrueHeightInversion,
)

# ── 1. Load real data ──────────────────────────────────────────────────────────
df, station = load_echoes(WI937)
print(f"[{station}]  Filtered echoes: {len(df)}")

# ── 2. Classify O/X modes ─────────────────────────────────────────────────────
pol = PolarizationClassifier(o_mode_sign=-1)
pol_result = pol.fit(df)
ann_df = pol_result.annotated_df
o_df = ann_df[ann_df["mode"] == "O"].copy()
print(f"O-mode: {len(o_df)}  X-mode: {len(ann_df[ann_df['mode']=='X'])}")

# ── 3. True-height inversion (needed for absorption_profile) ──────────────────
inv = TrueHeightInversion(monotone_enforce=True, bin_width_mhz=5e-3)
edp = inv.fit_from_df(o_df)
print(edp.summary())

# ── 4. Absorption analysis ────────────────────────────────────────────────────
ana = AbsorptionAnalyzer(
    snr_col="snr_db",
    freq_col="frequency_khz",
    height_col="height_km",
    mode_col="mode",
    freq_bin_mhz=0.1,
    f_ref_mhz=1.0,  # quiet-day reference for LOF index
)

# 4a — LOF absorption index
lof = ana.lof_absorption(df)
print("\n" + lof.summary())

# 4b — Differential O/X absorption
diff = ana.differential_absorption(ann_df)
print(diff.summary())

# 4c — Total calibrated absorption
#   WI937 VIPIR: ~300 W transmit → EIRP ≈ 27 dBW (300 W + ~2 dBi dipole),
#   receiver gain ~0 dBi (assume isotropic), reflection coefficient –3 dB.
total = ana.total_absorption(
    ann_df,
    tx_eirp_dbw=27.0,
    rx_gain_dbi=0.0,
    reflection_coeff_db=-3.0,
)
print(total.summary())

# 4d — Height-resolved absorption profile
#   The real EDP from lamination only covers the F-region (200–350 km) where
#   the electron-neutral collision frequency is negligible → total L ≈ 0 dB.
#   To demonstrate the κ(z) calculation we also run it on a synthetic
#   D/E-region Chapman EDP (clearly labelled as a model).
#
#   Real data use-case: supply a D-region EDP from partial-reflection sounder
#   or IRI/NRLMSISE model output in place of the synthetic EDP below.


def nu_model(h_km: float) -> float:
    """Budden (1961) electron-neutral collision frequency [Hz], h in km."""
    return 1.816e11 * np.exp(-0.15 * h_km)


prof = ana.absorption_profile(edp, nu_hz=nu_model, n_interp=300, f_wave_mhz=2.0)
print(prof.summary())

# Synthetic D/E-region EDP for profile demonstration
# Chapman layer centred at h0=90 km, scale height H=8 km,
# peak plasma frequency fp0=0.5 MHz (N_max ≈ 3×10^9 m^-3, moderate E layer)
from pynasonde.vipir.analysis.inversion import EDPResult

_h_syn = np.linspace(60.0, 130.0, 140)  # 60–130 km
_xi = (_h_syn - 90.0) / 8.0  # normalised height
_fp_syn = 0.5 * np.exp(0.5 * (1.0 - _xi - np.exp(-_xi)))  # Chapman fp (MHz)
_fp_syn = np.maximum(_fp_syn, 1e-4)

# Build EDPResult directly — no inversion needed for a synthetic profile
_FP_TO_N = 1.2399e4  # N_cm3 = fp_mhz^2 * this
_peak_idx = int(np.argmax(_fp_syn))
edp_syn = EDPResult(
    true_height_km=_h_syn,
    plasma_freq_mhz=_fp_syn,
    electron_density_cm3=_fp_syn**2 * _FP_TO_N,
    virtual_height_km=_h_syn,
    frequency_mhz=_fp_syn,
    foF2_mhz=float(_fp_syn[_peak_idx]),
    hmF2_km=float(_h_syn[_peak_idx]),
    NmF2_cm3=float(_fp_syn[_peak_idx] ** 2 * _FP_TO_N),
    method="synthetic_chapman",
    n_layers=len(_h_syn),
)
prof_syn = ana.absorption_profile(edp_syn, nu_hz=nu_model, n_interp=300, f_wave_mhz=2.0)
print("Synthetic D/E profile: " + prof_syn.summary())

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"HF Absorption Analysis — {station}", fontsize=13)

# Panel (a): LOF  — show fmin on the ionogram scatter
ax = axes[0, 0]
freq_mhz = df["frequency_khz"] / 1e3
ax.scatter(freq_mhz, df["height_km"], c="steelblue", s=4, alpha=0.4, label="echoes")
if np.isfinite(lof.fmin_mhz):
    ax.axvline(
        lof.fmin_mhz,
        color="firebrick",
        lw=1.5,
        ls="--",
        label=f"fmin = {lof.fmin_mhz:.2f} MHz",
    )
ax.set(
    xlabel="Frequency (MHz)",
    ylabel="Virtual height (km)",
    title=f"(a) LOF index  A = {lof.lof_index_mhz2:.2f} MHz²",
    ylim=(60, 800),
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (b): Differential ΔL(f)
ax = axes[0, 1]
diff.plot(ax=ax)
ax.set_title(f"(b) Differential absorption  mean ΔL = {diff.mean_delta_db:.1f} dB")
ax.grid(True, alpha=0.3)

# Panel (c): Calibrated L(f)
ax = axes[1, 0]
total.plot(ax=ax)
ax.set_title("(c) Calibrated one-way absorption L(f)")
ax.grid(True, alpha=0.3)

# Panel (d): κ(z) profile
ax = axes[1, 1]
pdf = prof.profile_df
cdf = prof.cumulative_df
if not pdf.empty:
    ax.plot(
        pdf["kappa_dB_per_km"],
        pdf["height_km"],
        color="tab:blue",
        lw=1.5,
        label="κ(z) dB/km",
    )
    ax.set_xlabel("κ  (dB km⁻¹)")
    ax.set_ylabel("Height (km)")
    if not cdf.empty:
        ax2 = ax.twiny()
        ax2.plot(
            cdf["L_oneway_db"],
            cdf["height_km"],
            color="tab:orange",
            lw=1.5,
            ls="--",
            label="L(z) dB",
        )
        ax2.set_xlabel("Cumulative L  (dB, one-way)", color="tab:orange")
        ax2.tick_params(axis="x", colors="tab:orange")
    ax.set_title(f"(d) Absorption profile  total = {prof.total_absorption_db:.1f} dB")
else:
    ax.text(
        0.5,
        0.5,
        "EDP insufficient for profile",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title("(d) Absorption profile")
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = "docs/examples/figures/analysis_absorption.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
