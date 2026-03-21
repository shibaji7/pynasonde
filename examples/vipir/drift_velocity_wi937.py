"""3-D ionospheric drift velocity estimation from a VIPIR RIQ file (WI937).

Builds on the echo-extraction pipeline (``echo_extraction_wi937.py``) and
demonstrates :meth:`~pynasonde.vipir.riq.echo.EchoExtractor.fit_drift_velocity`.

Physics background
------------------
Each ionospheric echo provides one line-of-sight (LOS) velocity measurement:

    V*_i = l_i · Vx + m_i · Vy + n_i · Vz

where the direction cosines (l, m, n) are derived from the echo's own
echolocation parameters (XL, YL, R'):

    l = XL / R'   (East),   m = YL / R'   (North),   n = sqrt(1 - l^2 - m^2)

Solving the overdetermined system via weighted least-squares over many echoes
with geometrically diverse arrival directions gives the 3-D drift vector
[Vx, Vy, Vz].  Iterative sigma-clipping removes outlier echoes whose LOS
velocity is inconsistent with the bulk fit.

Steps covered
-------------
1. Load ``WI937_2022233235902.RIQ`` and extract echoes.
2. Whole-sounding fit   — one [Vx, Vy, Vz] for the entire ionogram.
3. Height-binned fit    — separate [Vx, Vy, Vz] per 100 km height bin.
4. Plot a 2-panel figure:
       (A) Vx, Vy, Vz vs height bin (coloured lines)
       (B) RMS LOS residual and echo count vs height bin
5. Save to ``docs/examples/figures/drift_velocity_wi937.png``.

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
# Step 1: Load and extract echoes
# ---------------------------------------------------------------------------

riq = RiqDataset.create_from_file(
    fname,
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[1],
)

extractor = EchoExtractor(
    sct=riq.sct,
    pulsets=riq.pulsets,
    snr_threshold_db=3.0,
    min_height_km=60.0,    # exclude direct-wave clutter
    max_height_km=1000.0,  # exclude end-of-range artefacts
    min_rx_for_direction=3,
    max_echoes_per_pulset=5,
)
extractor.extract()
print(f"Echoes extracted: {len(extractor.echoes)}")

# ---------------------------------------------------------------------------
# Step 2: Whole-sounding fit
# ---------------------------------------------------------------------------

df_whole = extractor.fit_drift_velocity(
    height_bin_km=None,   # single fit over all echoes
    min_echoes=6,
    snr_weight=True,
    n_sigma=2.5,          # iterative sigma-clipping
    max_ep_deg=None,      # skip EP filter (WI937 8-Rx EP naturally large)
)

print("\n=== Whole-sounding drift velocity ===")
print(f"  Vx = {df_whole['vx_mps'].iloc[0]:+.1f} m/s  (East)")
print(f"  Vy = {df_whole['vy_mps'].iloc[0]:+.1f} m/s  (North)")
print(f"  Vz = {df_whole['vz_mps'].iloc[0]:+.1f} m/s  (Vertical)")
print(f"  RMS LOS residual : {df_whole['residual_mps'].iloc[0]:.1f} m/s")
print(f"  Condition number : {df_whole['condition_number'].iloc[0]:.1f}")
print(f"  Echoes used      : {df_whole['n_echoes'].iloc[0]}  "
      f"(rejected: {df_whole['n_rejected'].iloc[0]})")

# ---------------------------------------------------------------------------
# Step 3: Height-binned fit (100 km bins)
# ---------------------------------------------------------------------------

df_bins = extractor.fit_drift_velocity(
    height_bin_km=10.0,
    min_echoes=6,
    snr_weight=True,
    n_sigma=2.5,
    max_ep_deg=None,
)

print("\n=== Height-binned drift velocity ===")
print(df_bins[["height_bin_km", "vx_mps", "vy_mps", "vz_mps",
               "residual_mps", "n_echoes", "n_rejected"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Step 4: Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
fig.suptitle(
    "WI937  2022-233  23:59 UT — Ionospheric Drift Velocity  (vipir_version=0)",
    fontsize=font_size,
)

valid = df_bins.dropna(subset=["vx_mps"])
h = valid["height_bin_km"].values

# ── Panel A: velocity components vs height ────────────────────────────────
ax = axes[0]
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.plot(valid["vx_mps"], h, "o-", color="tab:blue",   label="Vx (East)",     ms=5)
ax.plot(valid["vy_mps"], h, "s-", color="tab:orange",  label="Vy (North)",    ms=5)
ax.plot(valid["vz_mps"], h, "^-", color="tab:green",   label="Vz (Vertical)", ms=5)

# Dotted verticals for whole-sounding estimates
for val, color in zip(
    [df_whole["vx_mps"].iloc[0],
     df_whole["vy_mps"].iloc[0],
     df_whole["vz_mps"].iloc[0]],
    ["tab:blue", "tab:orange", "tab:green"],
):
    if np.isfinite(val):
        ax.axvline(val, color=color, lw=1.2, ls=":", alpha=0.7)

ax.set_xlabel("Velocity (m/s)", fontsize=font_size)
ax.set_ylabel("Virtual height (km)", fontsize=font_size)
ax.set_title("Drift components", fontsize=font_size)
ax.legend(fontsize=font_size - 2)
ax.grid(True, alpha=0.3)

# ── Panel B: residual and echo count vs height ───────────────────────────
ax2 = axes[1]
ax2_r = ax2.twiny()

ax2.barh(h, valid["residual_mps"], height=60,
         color="tab:purple", alpha=0.6, label="RMS residual")
ax2_r.plot(valid["n_echoes"], h, "D--", color="tab:red", ms=5,
           label="N echoes used")

ax2.set_xlabel("RMS LOS residual (m/s)", fontsize=font_size, color="tab:purple")
ax2_r.set_xlabel("Echoes used per bin",   fontsize=font_size, color="tab:red")
ax2.set_title("Fit quality", fontsize=font_size)
ax2.grid(True, alpha=0.3)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           fontsize=font_size - 2, loc="lower right")

h_all = h if len(h) else np.array([100, 900])
axes[0].set_ylim(max(0, h_all.min() - 50), h_all.max() + 50)

fig.tight_layout()

out_path = "docs/examples/figures/drift_velocity_wi937.png"
fig.savefig(out_path, dpi=150)
print(f"\nFigure saved → {out_path}")
plt.close(fig)
