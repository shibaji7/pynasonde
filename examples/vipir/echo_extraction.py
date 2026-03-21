"""Dynasonde-style seven-parameter echo extraction from a VIPIR RIQ file.

This example demonstrates the full echo-extraction pipeline introduced in
``pynasonde.vipir.riq.echo``.  For each frequency step the extractor derives
the Dynasonde 21 parameters:

    φ₀  gross phase          (gross_phase_deg)
    V*  phase-path velocity  (velocity_mps)
    R'  virtual height       (height_km)
    XL  eastward offset      (xl_km)
    YL  northward offset     (yl_km)
    PP  polarization angle   (polarization_deg)
    EP  LS fit residual      (residual_deg)
    A   amplitude            (amplitude_db)

Steps covered:

1. Load a ``.RIQ`` binary file with :class:`RiqDataset`.
2. Run :class:`EchoExtractor` on all pulse sets.
3. Export results as a ``pandas.DataFrame`` and ``xarray.Dataset``.
4. Plot a 2×3 diagnostic figure:

       (A) Ionogram          — amplitude vs (frequency, virtual height)
       (B) XL vs Height      — eastward offset vs virtual height
       (C) YL vs Height      — northward offset vs virtual height
       (D) Echolocation map  — XL vs YL coloured by amplitude
       (E) Doppler velocity  — V* vs virtual height
       (F) Polarization PP   — polarization angle vs virtual height

5. Save the figure to ``docs/examples/figures/echo_extraction.png``.

Update ``fname`` to point to your local copy of the file.
"""

import matplotlib.pyplot as plt
import numpy as np

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.vipir.riq.echo import EchoExtractor
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

# ---------------------------------------------------------------------------
# Configuration — update these for your dataset
# ---------------------------------------------------------------------------

fname = "examples/data/PL407_2024058061501.RIQ"
font_size = 12
setsize(font_size)

# ---------------------------------------------------------------------------
# Step 1: Load RIQ file
# ---------------------------------------------------------------------------

riq = RiqDataset.create_from_file(
    fname,
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[0],   # version 1 / data_type 2
)
print(f"Loaded RIQ : {fname}")
print(f"  Pulsets  : {len(riq.pulsets)}")
print(f"  Receivers: {riq.sct.station.rx_count}")
print(f"  Freq start: {riq.sct.frequency.base_start:.1f} kHz")
print(f"  Freq end  : {riq.sct.frequency.base_end:.1f} kHz")
print(f"  Freq steps: {riq.sct.frequency.base_steps}")
print(f"  Log step  : {riq.sct.frequency.log_step:.4f} (fraction)")
print(f"  Linear step: {riq.sct.frequency.linear_step:.2f} kHz")

# ---------------------------------------------------------------------------
# Step 2: Extract echoes
# ---------------------------------------------------------------------------

extractor = EchoExtractor(
    sct=riq.sct,
    pulsets=riq.pulsets,
    snr_threshold_db=3.0,          # minimum coherent SNR to accept an echo
    min_height_km=60.0,            # exclude direct-wave / near-field clutter
    max_height_km=1000.0,          # exclude end-of-range aliasing artefacts
    min_rx_for_direction=3,        # minimum receivers for XL/YL/EP fit
    max_echoes_per_pulset=5,       # keep the 5 strongest echoes per frequency
)
extractor.extract()

print(f"\n  Echoes extracted: {len(extractor.echoes)}")

# ---------------------------------------------------------------------------
# Step 3: Export results
# ---------------------------------------------------------------------------

df = extractor.to_dataframe()
ds = extractor.to_xarray()

n_total = len(df)
if n_total:
    print("\nDirection-parameter coverage:")
    for col in ("xl_km", "yl_km", "polarization_deg", "residual_deg"):
        n_nan = df[col].isna().sum()
        print(
            f"  {col:20s}: {n_total - n_nan}/{n_total} valid "
            f"({100 * (n_total - n_nan) / n_total:.0f}%)"
        )

print()
print(df[["frequency_khz", "height_km", "amplitude_db", "velocity_mps"]].describe())

# ---------------------------------------------------------------------------
# Step 4: Plot diagnostics — 2×3 grid
# ---------------------------------------------------------------------------
#
#  (A) Ionogram          (B) XL vs height       (C) YL vs height
#  (D) XL–YL map         (E) Doppler velocity   (F) Polarization PP
#

fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(15, 9),
    constrained_layout=True,
)
fig.suptitle(
    f"VIPIR Echo Extraction — {fname.split('/')[-1]}  (vipir_version=1)",
    fontsize=font_size + 2,
)

freq_mhz = df["frequency_khz"] / 1e3 if not df.empty else None
amp_vmin = df["amplitude_db"].quantile(0.05) if not df.empty else 0
amp_vmax = df["amplitude_db"].quantile(0.95) if not df.empty else 1

# ── (A) Ionogram: frequency vs virtual height, colour = amplitude ────────────
ax = axes[0, 0]
if not df.empty:
    sc = ax.scatter(
        freq_mhz, df["height_km"],
        c=df["amplitude_db"], cmap="plasma", s=4,
        vmin=amp_vmin, vmax=amp_vmax, rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Amplitude (dB)", fontsize=font_size)
else:
    ax.text(
        0.5, 0.5, "No echoes detected",
        ha="center", va="center", transform=ax.transAxes, fontsize=font_size - 1,
    )
ax.set_xlabel("Frequency (MHz)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title("(A) Ionogram", fontsize=font_size)
ax.set_ylim(50, 1000)
# PL407 uses a linear frequency sweep — linear x-axis matches the data spacing.

# ── (B) XL vs virtual height, colour = frequency ────────────────────────────
ax = axes[0, 1]
xl_mask = df["xl_km"].notna() if not df.empty else []
if not df.empty and xl_mask.any():
    sc = ax.scatter(
        df.loc[xl_mask, "xl_km"], df.loc[xl_mask, "height_km"],
        c=freq_mhz[xl_mask], cmap="viridis", s=4,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
else:
    ax.text(
        0.5, 0.5, "XL all NaN\n(insufficient receivers)",
        ha="center", va="center", transform=ax.transAxes, fontsize=font_size - 1,
    )
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("XL — Eastward (km)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title("(B) XL vs Height", fontsize=font_size)
ax.set_ylim(50, 1000)

# ── (C) YL vs virtual height, colour = frequency ────────────────────────────
ax = axes[0, 2]
yl_mask = df["yl_km"].notna() if not df.empty else []
if not df.empty and yl_mask.any():
    sc = ax.scatter(
        df.loc[yl_mask, "yl_km"], df.loc[yl_mask, "height_km"],
        c=freq_mhz[yl_mask], cmap="viridis", s=4,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
else:
    ax.text(
        0.5, 0.5, "YL all NaN\n(insufficient receivers)",
        ha="center", va="center", transform=ax.transAxes, fontsize=font_size - 1,
    )
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("YL — Northward (km)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title("(C) YL vs Height", fontsize=font_size)
ax.set_ylim(50, 1000)

# ── (D) XL–YL echolocation map, colour = amplitude ──────────────────────────
ax = axes[1, 0]
dir_mask = (df["xl_km"].notna() & df["yl_km"].notna()) if not df.empty else []
if not df.empty and dir_mask.any():
    sc = ax.scatter(
        df.loc[dir_mask, "xl_km"], df.loc[dir_mask, "yl_km"],
        c=df.loc[dir_mask, "amplitude_db"], cmap="plasma", s=6, alpha=0.7,
        vmin=amp_vmin, vmax=amp_vmax, rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Amplitude (dB)", fontsize=font_size)
else:
    ax.text(
        0.5, 0.5, "No direction data", ha="center", va="center",
        transform=ax.transAxes, fontsize=font_size - 1,
    )
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("XL — Eastward (km)", fontsize=font_size)
ax.set_ylabel("YL — Northward (km)", fontsize=font_size)
ax.set_title("(D) Echolocation Map (XL, YL)", fontsize=font_size)

# ── (E) Doppler velocity vs virtual height, colour = frequency ──────────────
ax = axes[1, 1]
v_mask = df["velocity_mps"].notna() if not df.empty else []
if not df.empty and v_mask.any():
    sc = ax.scatter(
        df.loc[v_mask, "velocity_mps"], df.loc[v_mask, "height_km"],
        c=freq_mhz[v_mask], cmap="coolwarm", s=4, alpha=0.6,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
else:
    ax.text(
        0.5, 0.5, "No velocity data",
        ha="center", va="center", transform=ax.transAxes, fontsize=font_size - 1,
    )
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("V* — Phase-path velocity (m/s)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title("(E) Doppler Velocity", fontsize=font_size)
ax.set_ylim(50, 1000)

# ── (F) Polarization PP vs virtual height, colour = frequency ───────────────
ax = axes[1, 2]
pp_mask = df["polarization_deg"].notna() if not df.empty else []
if not df.empty and pp_mask.any():
    sc = ax.scatter(
        df.loc[pp_mask, "polarization_deg"], df.loc[pp_mask, "height_km"],
        c=freq_mhz[pp_mask], cmap="RdBu", s=4, alpha=0.7,
        vmin=-180, vmax=180, rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
else:
    ax.text(
        0.5, 0.5, "PP all NaN\n(no orthogonal antenna pairs)",
        ha="center", va="center", transform=ax.transAxes, fontsize=font_size - 1,
    )
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("PP — Polarization (°)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title("(F) Polarization PP", fontsize=font_size)
ax.set_ylim(50, 1000)

# ---------------------------------------------------------------------------
# Step 5: Save figure
# ---------------------------------------------------------------------------

out = "docs/examples/figures/echo_extraction.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Figure saved → {out}")

# ---------------------------------------------------------------------------
# Step 6: Demonstrate xarray CF output
# ---------------------------------------------------------------------------

print("\nxarray Dataset variables:")
for var in ds.data_vars:
    attrs = ds[var].attrs
    print(
        f"  {var:25s}  units={attrs.get('units', '—'):6s}"
        f"  long_name='{attrs.get('long_name', '—')}'"
    )

print(df.head())