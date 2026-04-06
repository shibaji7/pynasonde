"""High-resolution ionospheric layer imaging via Capon cross-spectrum analysis.

Implements the range-dimensional spectrum estimation of Liu et al. (2023):
the pulse-compressed gate profile of a single-frequency pulset is transformed
by the Capon estimator, improving range resolution by factor K without reducing
temporal resolution.

Pipeline
--------
1. Load echoes to find the frequency with the lowest-height returns (E/Es range
   preferred; F-layer used as fallback).  This guarantees the selected pulset
   has actual ionospheric signal in the display window.
2. Match that frequency to a pulset in the RIQ file.
3. Assemble the full complex IQ cube (pulse_count, gate_count, rx_count) —
   use ALL gates so Capon has enough frequency samples (V >> Z).
4. Image with EsCaponImager at Z = 50 and Z = 100 subbands.
5. Crop the displayed height window to where echoes are known to exist.
6. Also produce a per-pulse RTI to show temporal evolution.

Key insight: the height display window must match the frequency of the pulset.
At f = 1.6 MHz the E-layer reflects at ~122-127 km.  At f = 2 MHz the F-layer
reflects at ~256 km.  Displaying the wrong height window shows only noise.

Expected output
---------------
Figures saved to ``docs/examples/figures/analysis_es_imaging*.png``.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _load_data import WI937, load_echoes

from pynasonde.vipir.analysis import EsCaponImager
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

_C_KM_US = 299_792.458 / 1e6  # km per μs

# ── 1. Load echoes to find the best target frequency ─────────────────────────
df, station = load_echoes(WI937)
print(f"[{station}]  Filtered echoes: {len(df)}")

# Prefer E/Es range (< 200 km); fall back to F-layer.
e_df = df[df["height_km"] < 200.0]
if not e_df.empty:
    # Most-represented frequency in E-layer
    target_khz = float(e_df.groupby(e_df["frequency_khz"].round(-1)).size().idxmax())
    h_min_km = max(60.0, e_df["height_km"].min() - 20.0)
    h_max_km = min(250.0, e_df["height_km"].max() + 20.0)
    layer_label = "E/Es layer"
else:
    target_khz = float(df.groupby(df["frequency_khz"].round(-2)).size().idxmax())
    sub_f = df[
        (df["frequency_khz"] >= target_khz - 200)
        & (df["frequency_khz"] < target_khz + 200)
    ]
    h_min_km = max(60.0, sub_f["height_km"].min() - 30.0)
    h_max_km = min(800.0, sub_f["height_km"].max() + 30.0)
    layer_label = "F layer"

print(
    f"Target: {target_khz/1e3:.3f} MHz  "
    f"({layer_label}, display window {h_min_km:.0f}–{h_max_km:.0f} km)"
)

# ── 2. Load RIQ and derive gate geometry ─────────────────────────────────────
fname = "examples/data/WI937_2022233235902.RIQ"
riq = RiqDataset.create_from_file(
    fname,
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[1],
)

gate_spacing_km = riq.sct.timing.gate_step * _C_KM_US / 2
gate_start_km = riq.sct.timing.gate_start * _C_KM_US / 2
gate_count = int(riq.sct.timing.gate_count)
gate_heights = gate_start_km + np.arange(gate_count) * gate_spacing_km

print(
    f"gates={gate_count}  r₀={gate_spacing_km:.3f} km  " f"start={gate_start_km:.2f} km"
)

# ── 3. Find the pulset closest to target_khz ─────────────────────────────────
best_idx, best_diff = 0, np.inf
for idx, ps in enumerate(riq.pulsets):
    diff = abs(float(ps.pcts[0].frequency) - target_khz)
    if diff < best_diff:
        best_idx, best_diff = idx, diff

pulset = riq.pulsets[best_idx]
freq_mhz = float(pulset.pcts[0].frequency) / 1e3
n_pulse = len(pulset.pcts)
print(
    f"Selected pulset #{best_idx}  freq={freq_mhz:.3f} MHz  "
    f"Δf={best_diff:.0f} kHz  pulses={n_pulse}"
)

# ── 4. Build IQ cube (ALL gates — needed for Capon V >> Z) ───────────────────
parts = [
    pct.pulse_i.astype(np.float64) + 1j * pct.pulse_q.astype(np.float64)
    for pct in pulset.pcts
]
iq_cube = np.stack(parts, axis=0)  # (pulse_count, gate_count, rx_count)
_, n_gate, n_rx = iq_cube.shape
print(f"IQ cube: {n_pulse}×{n_gate}×{n_rx}")

# Display gate index range
lo = int(np.searchsorted(gate_heights, h_min_km))
hi = int(np.searchsorted(gate_heights, h_max_km))
hi = min(hi, n_gate - 1)
disp_heights = gate_heights[lo:hi]

# ── 5. Capon imaging on the FULL gate range ───────────────────────────────────
# K controls output grid density only (Δr = r₀/K); it has no singularity
# constraint.  Capped at 9 here so the steering matrix stays tractable for
# VIPIR's small gate count relative to WISS (V=960 vs 200).
K = min(9, n_gate // 100)
z50 = 50
z100 = 100

result_z50 = EsCaponImager(
    n_subbands=z50,
    resolution_factor=K,
    coherent_integrations=n_pulse,  # one fully integrated snapshot
    gate_start_km=gate_start_km,
    gate_spacing_km=gate_spacing_km,
).fit(iq_cube)

result_z100 = EsCaponImager(
    n_subbands=z100,
    resolution_factor=K,
    coherent_integrations=n_pulse,
    gate_start_km=gate_start_km,
    gate_spacing_km=gate_spacing_km,
).fit(iq_cube)

print(result_z50.summary())
print(result_z100.summary())

# Per-pulse RTI (n_pulse snapshots, one per raw pulse)
result_rti = EsCaponImager(
    n_subbands=z100,
    resolution_factor=K,
    coherent_integrations=1,
    gate_start_km=gate_start_km,
    gate_spacing_km=gate_spacing_km,
).fit(iq_cube)

# ── 6. Standard (non-Capon) amplitude profile for comparison ─────────────────
C_mean = iq_cube.mean(axis=0)  # (gate, rx)
amp_lin = np.mean(np.abs(C_mean), axis=-1)  # (gate,)
amp_db = 20.0 * np.log10(np.maximum(amp_lin, 1e-15))
amp_db -= amp_db.max()
amp_db_disp = amp_db[lo:hi]  # crop to display window

# ── 7. Three-panel comparison figure ─────────────────────────────────────────
VMIN = -60.0
fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
fig.suptitle(
    f"Capon range imaging — {station}  f = {freq_mhz:.3f} MHz  "
    f"K = {K}  ({layer_label})",
    fontsize=12,
)


def _crop(result, h_lo, h_hi):
    mask = (result.heights_km >= h_lo) & (result.heights_km <= h_hi)
    return result.heights_km[mask], result.pseudospectrum_db[0][mask]


# Panel (a): standard profile
ax = axes[0]
ax.plot(amp_db_disp, disp_heights, color="steelblue", lw=1.4)
ax.set_xlabel("Normalised power (dB)")
ax.set_ylabel("Virtual height (km)")
ax.set_title(f"(a) Standard  r₀ = {gate_spacing_km:.2f} km")
ax.set_xlim(VMIN, 2)
ax.set_ylim(h_min_km, h_max_km)
ax.grid(True, alpha=0.3)

# Panel (b): Capon Z=50
ax = axes[1]
h_b, p_b = _crop(result_z50, h_min_km, h_max_km)
ax.plot(p_b, h_b, color="steelblue", lw=1.4)
ax.set_xlabel("Normalised power (dB)")
ax.set_title(f"(b) Capon  Z={z50}  Δr={gate_spacing_km/K:.3f} km")
ax.set_xlim(VMIN, 2)
ax.set_ylim(h_min_km, h_max_km)
ax.grid(True, alpha=0.3)

# Panel (c): Capon Z=100
ax = axes[2]
h_c, p_c = _crop(result_z100, h_min_km, h_max_km)
ax.plot(p_c, h_c, color="steelblue", lw=1.4)
ax.set_xlabel("Normalised power (dB)")
ax.set_title(f"(c) Capon  Z={z100}  Δr={gate_spacing_km/K:.3f} km")
ax.set_xlim(VMIN, 2)
ax.set_ylim(h_min_km, h_max_km)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = "docs/examples/figures/analysis_es_imaging.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")

# ── 8. RTI figure (per-pulse temporal evolution) ─────────────────────────────
if result_rti.n_snapshots > 1:
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig2.suptitle(
        f"Pulse-by-pulse evolution — {station}  f = {freq_mhz:.3f} MHz  "
        f"({layer_label})",
        fontsize=12,
    )
    t_ax = np.arange(n_pulse)

    # Standard RTI (crop to display window)
    ax = axes2[0]
    C_pp = iq_cube[:, lo:hi, :].mean(axis=-1)  # (pulse, disp_gates)
    P_pp = 20.0 * np.log10(np.maximum(np.abs(C_pp), 1e-15))
    P_pp -= P_pp.max()
    ax.pcolormesh(
        t_ax, disp_heights, P_pp.T, cmap="jet", vmin=VMIN, vmax=0, shading="auto"
    )
    ax.set_xlabel("Pulse index")
    ax.set_ylabel("Virtual height (km)")
    ax.set_ylim(h_min_km, h_max_km)
    ax.set_title(f"(a) Standard  r₀ = {gate_spacing_km:.2f} km")

    # Capon RTI
    ax2 = axes2[1]
    h_mask = (result_rti.heights_km >= h_min_km) & (result_rti.heights_km <= h_max_km)
    ax2.pcolormesh(
        t_ax,
        result_rti.heights_km[h_mask],
        result_rti.pseudospectrum_db[:, h_mask].T,
        cmap="jet",
        vmin=VMIN,
        vmax=0,
        shading="auto",
    )
    ax2.set_xlabel("Pulse index")
    ax2.set_ylabel("")
    ax2.set_ylim(h_min_km, h_max_km)
    ax2.set_title(f"(b) Capon  Z={z100}  Δr={gate_spacing_km/K:.3f} km")

    fig2.tight_layout()
    out2 = "docs/examples/figures/analysis_es_imaging_rti.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"RTI figure saved → {out2}")
