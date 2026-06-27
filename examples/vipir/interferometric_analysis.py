"""Interferometric echo analysis from a VIPIR RIQ file.

This example demonstrates the three interferometric extensions added to
:class:`~pynasonde.vipir.riq.echo.EchoExtractor`:

    1. **MVDR direction** — Minimum Variance Distortionless Response
       (Capon) beamformer for echo direction in the East-North plane.
       Results land in ``xl_km_mvdr`` / ``yl_km_mvdr`` / ``residual_deg_mvdr``.

    2. **3-D elevation** — Full three-component baseline LS solve for the
       vertical direction cosine, yielding echo elevation angle
       ``elevation_deg`` directly from the data (rather than inferring it
       as ``arccos(sqrt(1−l²−m²))``).

    3. **Doppler spectrum** — FFT across the pulse dimension at each gate,
       providing a full Doppler power spectrum (``doppler_spectrum``) that
       reveals multiple simultaneous velocity components and spectral width.

All three extensions are **off by default** and run *alongside* the
existing seven-parameter pipeline — no existing output fields are changed.

Steps covered
-------------
1. Load a ``.RIQ`` file.
2. Run :class:`EchoExtractor` with all three extensions enabled.
3. Compare LS vs. MVDR direction estimates (XL, YL scatter plot).
4. Plot elevation angle distribution.
5. Plot Doppler spectra for selected echoes.
6. Save a three-panel diagnostic figure.

Update ``fname`` to point to your local copy of the file.
"""

import matplotlib.pyplot as plt
import numpy as np

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.vipir.riq.echo import EchoExtractor
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

fname = "examples/data/WI937_2022233235902.RIQ"
font_size = 12
setsize(font_size)

# ---------------------------------------------------------------------------
# Step 1: Load RIQ file
# ---------------------------------------------------------------------------

riq = RiqDataset.create_from_file(
    fname,
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[1],
)
print(f"Loaded  : {fname}")
print(f"Pulsets : {len(riq.pulsets)}")
print(f"Rx count: {riq.sct.station.rx_count}")

# ---------------------------------------------------------------------------
# Step 2: Extract echoes with all interferometric extensions enabled
# ---------------------------------------------------------------------------

extractor = EchoExtractor(
    sct=riq.sct,
    pulsets=riq.pulsets,
    snr_threshold_db=3.0,
    min_height_km=60.0,
    max_height_km=1000.0,
    min_rx_for_direction=3,
    max_echoes_per_pulset=5,
    # ── interferometric extensions ──
    enable_mvdr=True,  # Capon beamformer XL/YL
    enable_elevation=True,  # 3-D baseline elevation angle
    enable_doppler_spectrum=True,  # FFT Doppler spectrum per echo
)
extractor.extract()

df = extractor.to_dataframe()
print(f"\nEchoes extracted: {len(df)}")

# Coverage report
for col in ("xl_km", "yl_km", "xl_km_mvdr", "yl_km_mvdr", "elevation_deg"):
    n_valid = df[col].notna().sum()
    pct = 100 * n_valid / max(len(df), 1)
    print(f"  {col:22s}: {n_valid}/{len(df)}  ({pct:.0f}%)")

# ---------------------------------------------------------------------------
# Step 3 – 5: Build diagnostic figure  (3 rows × 2 columns)
#
#   Row 1  LS vs MVDR direction comparison
#           (A) XL_LS vs XL_MVDR          (B) YL_LS vs YL_MVDR
#   Row 2  MVDR-only echolocation
#           (C) MVDR XL–YL map            (D) Elevation angle distribution
#   Row 3  Doppler spectra
#           (E) Spectrum of 3 example echoes (strongest at low / mid / high h')
#           (F) Spectral width vs virtual height
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(13, 15),
    constrained_layout=True,
)
fig.suptitle(
    f"Interferometric Echo Analysis — {fname.split('/')[-1]}",
    fontsize=font_size + 2,
)

freq_mhz = df["frequency_khz"] / 1e3 if not df.empty else None


# helpers
def _no_data(ax, msg="No data"):
    ax.text(
        0.5,
        0.5,
        msg,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=font_size - 1,
    )


# ── (A) XL comparison: LS vs MVDR ────────────────────────────────────────────
ax = axes[0, 0]
mask = df["xl_km"].notna() & df["xl_km_mvdr"].notna()
if mask.any():
    sc = ax.scatter(
        df.loc[mask, "xl_km"],
        df.loc[mask, "xl_km_mvdr"],
        c=df.loc[mask, "height_km"],
        cmap="viridis",
        s=6,
        alpha=0.6,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Height (km)", fontsize=font_size)
    lim = (
        max(
            np.nanmax(np.abs(df["xl_km"].dropna())),
            np.nanmax(np.abs(df["xl_km_mvdr"].dropna())),
        )
        * 1.05
    )
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="1:1")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=font_size - 1)
else:
    _no_data(ax)
ax.set_xlabel("XL  LS estimate (km)", fontsize=font_size)
ax.set_ylabel("XL  MVDR estimate (km)", fontsize=font_size)
ax.set_title("(A)  XL: LS vs MVDR", fontsize=font_size)

# ── (B) YL comparison: LS vs MVDR ────────────────────────────────────────────
ax = axes[0, 1]
mask = df["yl_km"].notna() & df["yl_km_mvdr"].notna()
if mask.any():
    sc = ax.scatter(
        df.loc[mask, "yl_km"],
        df.loc[mask, "yl_km_mvdr"],
        c=df.loc[mask, "height_km"],
        cmap="viridis",
        s=6,
        alpha=0.6,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Height (km)", fontsize=font_size)
    lim = (
        max(
            np.nanmax(np.abs(df["yl_km"].dropna())),
            np.nanmax(np.abs(df["yl_km_mvdr"].dropna())),
        )
        * 1.05
    )
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="1:1")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=font_size - 1)
else:
    _no_data(ax)
ax.set_xlabel("YL  LS estimate (km)", fontsize=font_size)
ax.set_ylabel("YL  MVDR estimate (km)", fontsize=font_size)
ax.set_title("(B)  YL: LS vs MVDR", fontsize=font_size)

# ── (C) MVDR XL–YL echolocation map ─────────────────────────────────────────
ax = axes[1, 0]
mask = df["xl_km_mvdr"].notna() & df["yl_km_mvdr"].notna()
if mask.any():
    sc = ax.scatter(
        df.loc[mask, "xl_km_mvdr"],
        df.loc[mask, "yl_km_mvdr"],
        c=df.loc[mask, "amplitude_db"],
        cmap="plasma",
        s=8,
        alpha=0.7,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Amplitude (dB)", fontsize=font_size)
else:
    _no_data(ax)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("XL — Eastward  MVDR (km)", fontsize=font_size)
ax.set_ylabel("YL — Northward  MVDR (km)", fontsize=font_size)
ax.set_title("(C)  MVDR Echolocation Map", fontsize=font_size)

# ── (D) Elevation angle distribution ─────────────────────────────────────────
ax = axes[1, 1]
el = df["elevation_deg"].dropna()
if len(el) > 0:
    ax.scatter(
        el,
        df.loc[el.index, "height_km"],
        c=freq_mhz[el.index],
        cmap="coolwarm",
        s=5,
        alpha=0.6,
        rasterized=True,
    )
    sc = ax.scatter(
        el,
        df.loc[el.index, "height_km"],
        c=freq_mhz[el.index],
        cmap="coolwarm",
        s=5,
        alpha=0.6,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
    ax.axvline(90, color="k", lw=0.5, ls="--", label="Vertical (90°)")
    ax.legend(fontsize=font_size - 1)
else:
    _no_data(ax, "elevation_deg all NaN\n(flat array — no Up baseline variation)")
ax.set_xlabel("Elevation angle (°)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title("(D)  Echo Elevation Angle", fontsize=font_size)
ax.set_ylim(50, 1000)

# ── (E) Doppler spectra for three representative echoes ──────────────────────
ax = axes[2, 0]
echoes_with_spec = [
    e
    for e in extractor.echoes
    if e.doppler_spectrum is not None and np.isfinite(e.height_km)
]
if echoes_with_spec:
    # Pick three echoes: near E-layer, mid F1, upper F2
    heights = np.array([e.height_km for e in echoes_with_spec])
    targets = [100.0, 250.0, 400.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for target, color in zip(targets, colors):
        idx = int(np.argmin(np.abs(heights - target)))
        e = echoes_with_spec[idx]
        spec = e.doppler_spectrum
        if not np.isfinite(spec.velocity_axis).any():
            continue
        ax.plot(
            spec.velocity_axis,
            spec.spectrum,
            color=color,
            lw=1.2,
            label=f"h'={e.height_km:.0f} km  f={e.frequency_khz/1e3:.1f} MHz",
        )
    ax.legend(fontsize=font_size - 2)
    ax.set_xlabel("Doppler velocity (m/s)", fontsize=font_size)
    ax.set_ylabel("Normalised power", fontsize=font_size)
else:
    _no_data(ax, "No Doppler spectra available")
ax.set_title("(E)  Doppler Spectra (selected echoes)", fontsize=font_size)

# ── (F) Doppler spectral width vs virtual height ─────────────────────────────
ax = axes[2, 1]
widths, spec_heights, spec_freqs = [], [], []
for e in echoes_with_spec:
    spec = e.doppler_spectrum
    if not np.isfinite(spec.velocity_axis).any() or len(spec.spectrum) < 3:
        continue
    peak = spec.spectrum.max()
    half = peak / 2.0
    above = spec.velocity_axis[spec.spectrum >= half]
    if len(above) >= 2:
        widths.append(float(above[-1] - above[0]))
        spec_heights.append(e.height_km)
        spec_freqs.append(e.frequency_khz / 1e3)

if widths:
    sc = ax.scatter(
        widths,
        spec_heights,
        c=spec_freqs,
        cmap="plasma",
        s=6,
        alpha=0.7,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Frequency (MHz)", fontsize=font_size)
else:
    _no_data(ax)
ax.set_xlabel("Doppler spectral width  FWHM (m/s)", fontsize=font_size)
ax.set_ylabel("Virtual Height (km)", fontsize=font_size)
ax.set_title("(F)  Spectral Width vs Height", fontsize=font_size)
ax.set_ylim(50, 1000)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out = "docs/examples/figures/interferometric_analysis.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")

# ---------------------------------------------------------------------------
# Quick numeric summary
# ---------------------------------------------------------------------------

if not df.empty:
    mask_both = df["xl_km"].notna() & df["xl_km_mvdr"].notna()
    if mask_both.any():
        diff_xl = df.loc[mask_both, "xl_km_mvdr"] - df.loc[mask_both, "xl_km"]
        diff_yl = df.loc[mask_both, "yl_km_mvdr"] - df.loc[mask_both, "yl_km"]
        print(f"\nLS vs MVDR direction difference:")
        print(f"  XL  mean={diff_xl.mean():.2f} km   std={diff_xl.std():.2f} km")
        print(f"  YL  mean={diff_yl.mean():.2f} km   std={diff_yl.std():.2f} km")

    el = df["elevation_deg"].dropna()
    if len(el):
        print(
            f"\nElevation angle:  mean={el.mean():.1f}°  "
            f"std={el.std():.1f}°  range=[{el.min():.1f}, {el.max():.1f}]°"
        )
