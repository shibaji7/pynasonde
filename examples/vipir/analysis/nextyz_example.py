"""3-D electron density inversion using NeXtYZInverter (Lite mode).

Pipeline
--------
1. Load and filter echoes from WI937_2022233235902.RIQ.
2. Classify O/X modes via PolarizationClassifier.
3. Run NeXtYZInverter (Lite) with station geomagnetic parameters.
4. Print the solved profile with tilt angles and error estimates.
5. Plot fp(h) profile with error bars and tilt angles.

Expected output
---------------
Figure saved to ``docs/examples/figures/analysis_nextyz.png``.

Station parameters used here
-----------------------------
    Bear Lake Observatory, Utah (41° N, 111° W) — WI937:
        dip_angle_deg  ≈ 66°
        declination_deg ≈ +11° (East)
        B_gauss         ≈ 0.55

References
----------
Zabotin, N. A., Wright, J. W., & Zhbankov, G. A. (2006).  NeXtYZ:
Three-dimensional electron density inversion for dynasonde ionograms.
Radio Science, 41, RS6S32.
"""

import os

import matplotlib.pyplot as plt
from _load_data import WI937, load_echoes

from pynasonde.vipir.analysis import NeXtYZInverter, PolarizationClassifier

# ── 1. Load real data ─────────────────────────────────────────────────────────
df, station = load_echoes(WI937)
print(f"[{station}]  Filtered echoes: {len(df)}")

# ── 2. O/X labelling ──────────────────────────────────────────────────────────
ann = PolarizationClassifier(o_mode_sign=-1).fit(df).annotated_df

# ── 3. NeXtYZ Lite inversion ──────────────────────────────────────────────────
# Bear Lake Observatory geomagnetic parameters (approximate)
inv = NeXtYZInverter(
    dip_angle_deg=66.0,
    declination_deg=11.0,
    B_gauss=0.55,
    fp_step_mhz=0.2,
    min_echoes=6,
    max_echoes=50,
    mode="Lite",
    fp_start_mhz=2.0,
    xl_col="xl_km",
    yl_col="yl_km",
    height_col="height_km",
    freq_col="frequency_khz",
    mode_col="mode",
    amp_col="amplitude_db",
)

result = inv.fit(ann)

print(result.summary())
df_out = result.to_dataframe()
if not df_out.empty:
    print("\nSolved wedge profile (first 10 rows):")
    print(
        df_out[
            [
                "fp_lo_mhz",
                "fp_hi_mhz",
                "h_upper_km",
                "h_error_km",
                "tilt_meridional_deg",
                "tilt_zonal_deg",
                "residual_km",
                "n_echoes",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

# ── 4. Plot ───────────────────────────────────────────────────────────────────
result.plot()  # built-in two-panel plot (profile + tilt)

fig = plt.gcf()
fig.suptitle(f"NeXtYZ Lite — 3-D WSI electron density inversion\n({station})")
fig.tight_layout()
out = "docs/examples/figures/analysis_nextyz.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
