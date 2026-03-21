"""Filter then full parameter analysis for a VIPIR WI937 sounding.

This example combines the echo-filtering pipeline with the complete
Dynasonde parameter suite, demonstrating the effect of coherent noise
rejection on the derived ionospheric parameters.

Pipeline
--------
1. Load ``WI937_2022233235902.RIQ``  (vipir_version=0, configs[1]).
2. Run :class:`~pynasonde.vipir.riq.echo.EchoExtractor` → raw echo cloud.
3. Apply :class:`~pynasonde.vipir.riq.parsers.filter.IonogramFilter`
   (6 stages: RFI → EP → Multi-hop → DBSCAN → RANSAC → skip temporal).
4. Compute 3-D drift velocity [Vx, Vy, Vz] from **raw** and **filtered**
   echoes via height-binned weighted least-squares.
5. Plot a 3×3 diagnostic figure:

       (A) Raw ionogram (amplitude)
       (B) Filtered ionogram (amplitude)
       (C) Amplitude vs height: raw vs filtered
       (D) EP vs height: raw vs filtered
       (E) PP (polarization) vs height: raw vs filtered
       (F) V* (LOS velocity) vs height: raw vs filtered
       (G) Vx/Vy/Vz vs height: raw fit
       (H) Vx/Vy/Vz vs height: filtered fit
       (I) RMS LOS residual & echo count: raw vs filtered fit

6. Save figure to ``docs/examples/figures/ionogram_full_analysis_wi937.png``.

Update ``fname`` to point to your local copy of the file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.vipir.riq.echo import EchoExtractor
from pynasonde.vipir.riq.parsers.filter import IonogramFilter
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

fname     = "examples/data/WI937_2022233235902.RIQ"
font_size = 12
setsize(font_size)

# ---------------------------------------------------------------------------
# Helper: fit 3-D drift velocity from a DataFrame of echoes
# ---------------------------------------------------------------------------

def _fit_group_df(grp: pd.DataFrame, min_echoes: int,
                  snr_weight: bool, n_sigma: float) -> dict | None:
    """Weighted LS fit of [Vx, Vy, Vz] from one height-bin DataFrame."""
    valid = grp.dropna(subset=["xl_km", "yl_km", "velocity_mps"])
    if len(valid) < min_echoes:
        return None

    xl = valid["xl_km"].to_numpy(float)
    yl = valid["yl_km"].to_numpy(float)
    h  = valid["height_km"].to_numpy(float)
    v  = valid["velocity_mps"].to_numpy(float)

    R  = np.sqrt(h**2 + xl**2 + yl**2)
    R  = np.where(R > 0, R, h)
    l  = xl / R
    m  = yl / R
    n  = np.sqrt(np.maximum(0.0, 1.0 - l**2 - m**2))
    A  = np.column_stack([l, m, n])

    if snr_weight and "snr_db" in valid.columns:
        w = 10.0 ** (valid["snr_db"].fillna(0.0).to_numpy(float) / 20.0)
    else:
        w = np.ones(len(valid))

    mask = np.ones(len(valid), dtype=bool)
    vel  = np.zeros(3)
    for _ in range(5):
        if mask.sum() < min_echoes:
            return None
        Aw = A[mask] * w[mask, None]
        vw = v[mask] * w[mask]
        vel, _, _, _ = np.linalg.lstsq(Aw, vw, rcond=None)
        res = np.abs(v - A @ vel)
        std = res[mask].std()
        if std == 0:
            break
        new_mask = res < n_sigma * std
        if new_mask.sum() < min_echoes:
            break
        mask = new_mask

    Aw   = A[mask] * w[mask, None]
    vw   = v[mask] * w[mask]
    vel, _, _, _ = np.linalg.lstsq(Aw, vw, rcond=None)
    res  = np.abs(v[mask] - A[mask] @ vel)
    cond = np.linalg.cond(Aw)

    return {
        "vx_mps": float(vel[0]),
        "vy_mps": float(vel[1]),
        "vz_mps": float(vel[2]),
        "residual_mps":    float(res.mean()),
        "condition_number": float(cond),
        "n_echoes":   int(mask.sum()),
        "n_rejected": int(len(valid) - mask.sum()),
    }


def fit_drift_from_df(
    df: pd.DataFrame,
    height_bin_km: float = 50.0,
    min_echoes: int = 6,
    snr_weight: bool = True,
    n_sigma: float = 2.5,
) -> pd.DataFrame:
    """Height-binned 3-D drift velocity from a filtered echo DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Echo DataFrame (must contain xl_km, yl_km, height_km, velocity_mps).
    height_bin_km : float
        Bin width in km.
    min_echoes, snr_weight, n_sigma
        Passed to the weighted LS fit.

    Returns
    -------
    pd.DataFrame
        Columns: height_bin_km, vx_mps, vy_mps, vz_mps,
                 residual_mps, condition_number, n_echoes, n_rejected.
    """
    df = df.copy()
    df["_bin"] = (
        (df["height_km"] / height_bin_km).astype(int) * height_bin_km
        + height_bin_km / 2.0
    )
    rows = []
    for b, grp in df.groupby("_bin"):
        r = _fit_group_df(grp, min_echoes, snr_weight, n_sigma)
        if r is not None:
            r["height_bin_km"] = float(b)
            rows.append(r)
    if not rows:
        return pd.DataFrame(
            columns=["height_bin_km", "vx_mps", "vy_mps", "vz_mps",
                     "residual_mps", "condition_number", "n_echoes", "n_rejected"]
        )
    return pd.DataFrame(rows).sort_values("height_bin_km").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 1: Load and extract echoes
# ---------------------------------------------------------------------------

riq = RiqDataset.create_from_file(
    fname,
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[1],   # vipir_version=0 / data_type=1
)

extractor = EchoExtractor(
    sct=riq.sct,
    pulsets=riq.pulsets,
    snr_threshold_db=3.0,
    min_height_km=60.0,
    max_height_km=1000.0,
    min_rx_for_direction=3,
    max_echoes_per_pulset=5,
)
extractor.extract()
df_raw = extractor.to_dataframe()
print(f"Raw echoes      : {len(df_raw)}")

# ---------------------------------------------------------------------------
# Step 2: Filter
# ---------------------------------------------------------------------------

filt = IonogramFilter(
    rfi_enabled=True,      rfi_height_iqr_km=300.0, rfi_min_echoes=3,
    ep_filter_enabled=True, ep_max_deg=90.0,
    multihop_enabled=True,  multihop_orders=(2, 3),
    multihop_height_tol_km=50.0, multihop_snr_margin_db=6.0,
    dbscan_enabled=True,   dbscan_eps=1.0, dbscan_min_samples=5,
    dbscan_features=(
        "frequency_khz", "height_km",
        "velocity_mps", "amplitude_db", "residual_deg",
    ),
    ransac_enabled=True,   ransac_residual_km=100.0,
    ransac_min_samples=10, ransac_n_iter=200,
    ransac_poly_degree=3,  ransac_min_inlier_fraction=0.3,
    temporal_enabled=False,
)

df_filt = filt.filter(extractor)
print(f"Filtered echoes : {len(df_filt)}")
print()
print(filt.summary())

# ---------------------------------------------------------------------------
# Step 3: Drift velocity — raw and filtered
# ---------------------------------------------------------------------------

HEIGHT_BIN = 50.0   # km

df_vel_raw  = fit_drift_from_df(df_raw,  height_bin_km=HEIGHT_BIN)
df_vel_filt = fit_drift_from_df(df_filt, height_bin_km=HEIGHT_BIN)

print("\n=== Drift velocity (raw echoes) ===")
print(df_vel_raw[["height_bin_km","vx_mps","vy_mps","vz_mps",
                   "residual_mps","n_echoes"]].to_string(index=False))

print("\n=== Drift velocity (filtered echoes) ===")
print(df_vel_filt[["height_bin_km","vx_mps","vy_mps","vz_mps",
                    "residual_mps","n_echoes"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Step 4: Plot 3×3 diagnostic figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True)
fig.suptitle(
    "WI937  2022-233  23:59 UT — Filter → Full Parameter Analysis",
    fontsize=font_size + 1,
)

freq_mhz_raw  = df_raw["frequency_khz"]  / 1e3
freq_mhz_filt = df_filt["frequency_khz"] / 1e3

amp_vmin = df_raw["amplitude_db"].quantile(0.05)
amp_vmax = df_raw["amplitude_db"].quantile(0.95)

# ── (A) Raw ionogram ─────────────────────────────────────────────────────────
ax = axes[0, 0]
sc = ax.scatter(freq_mhz_raw, df_raw["height_km"],
                c=df_raw["amplitude_db"], cmap="plasma",
                s=3, vmin=amp_vmin, vmax=amp_vmax, rasterized=True)
fig.colorbar(sc, ax=ax, pad=0.02).set_label("A (dB)", fontsize=font_size - 1)
ax.set(xlabel="Frequency (MHz)", ylabel="Height (km)",
       title=f"(A) Raw ionogram  [{len(df_raw)} echoes]", ylim=(50, 1000))
ax.grid(True, alpha=0.3)

# ── (B) Filtered ionogram ────────────────────────────────────────────────────
ax = axes[0, 1]
sc2 = ax.scatter(freq_mhz_filt, df_filt["height_km"],
                 c=df_filt["amplitude_db"], cmap="plasma",
                 s=3, vmin=amp_vmin, vmax=amp_vmax, rasterized=True)
fig.colorbar(sc2, ax=ax, pad=0.02).set_label("A (dB)", fontsize=font_size - 1)
ax.set(xlabel="Frequency (MHz)", ylabel="Height (km)",
       title=f"(B) Filtered ionogram  [{len(df_filt)} echoes]", ylim=(50, 1000))
ax.grid(True, alpha=0.3)

# ── (C) Amplitude vs height ───────────────────────────────────────────────────
ax = axes[0, 2]
ax.scatter(df_raw["amplitude_db"],  df_raw["height_km"],
           color="tab:grey", s=2, alpha=0.3, label=f"Raw ({len(df_raw)})",
           rasterized=True)
ax.scatter(df_filt["amplitude_db"], df_filt["height_km"],
           color="tab:blue", s=2, alpha=0.6, label=f"Filtered ({len(df_filt)})",
           rasterized=True)
ax.set(xlabel="Amplitude (dB)", ylabel="Height (km)",
       title="(C) Amplitude vs height", ylim=(50, 1000))
ax.legend(fontsize=font_size - 3, markerscale=3)
ax.grid(True, alpha=0.3)

# ── (D) EP (residual_deg) vs height ──────────────────────────────────────────
ax = axes[1, 0]
ep_raw  = df_raw["residual_deg"].notna()
ep_filt = df_filt["residual_deg"].notna()
ax.scatter(df_raw.loc[ep_raw,   "residual_deg"],
           df_raw.loc[ep_raw,   "height_km"],
           color="tab:grey", s=2, alpha=0.3,
           label=f"Raw ({ep_raw.sum()})", rasterized=True)
ax.scatter(df_filt.loc[ep_filt, "residual_deg"],
           df_filt.loc[ep_filt, "height_km"],
           color="tab:orange", s=2, alpha=0.7,
           label=f"Filtered ({ep_filt.sum()})", rasterized=True)
ax.axvline(filt.ep_max_deg, color="k", lw=1, ls="--",
           label=f"EP threshold ({filt.ep_max_deg}°)")
ax.set(xlabel="EP — wavefront residual (°)", ylabel="Height (km)",
       title="(D) EP vs height", ylim=(50, 1000))
ax.legend(fontsize=font_size - 3, markerscale=3)
ax.grid(True, alpha=0.3)

# ── (E) PP (polarization_deg) vs height ──────────────────────────────────────
ax = axes[1, 1]
pp_raw  = df_raw["polarization_deg"].notna()
pp_filt = df_filt["polarization_deg"].notna()
ax.scatter(df_raw.loc[pp_raw,   "polarization_deg"],
           df_raw.loc[pp_raw,   "height_km"],
           color="tab:grey", s=2, alpha=0.3,
           label=f"Raw ({pp_raw.sum()})", rasterized=True)
ax.scatter(df_filt.loc[pp_filt, "polarization_deg"],
           df_filt.loc[pp_filt, "height_km"],
           color="tab:green", s=2, alpha=0.7,
           label=f"Filtered ({pp_filt.sum()})", rasterized=True)
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set(xlabel="PP — polarization (°)", ylabel="Height (km)",
       title="(E) PP vs height", ylim=(50, 1000), xlim=(-180, 180))
ax.legend(fontsize=font_size - 3, markerscale=3)
ax.grid(True, alpha=0.3)

# ── (F) V* (LOS velocity) vs height ──────────────────────────────────────────
ax = axes[1, 2]
v_raw  = df_raw["velocity_mps"].notna()
v_filt = df_filt["velocity_mps"].notna()
ax.scatter(df_raw.loc[v_raw,   "velocity_mps"],
           df_raw.loc[v_raw,   "height_km"],
           color="tab:grey", s=2, alpha=0.3,
           label=f"Raw ({v_raw.sum()})", rasterized=True)
ax.scatter(df_filt.loc[v_filt, "velocity_mps"],
           df_filt.loc[v_filt, "height_km"],
           color="tab:red", s=2, alpha=0.7,
           label=f"Filtered ({v_filt.sum()})", rasterized=True)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set(xlabel="V* — LOS velocity (m/s)", ylabel="Height (km)",
       title="(F) V* vs height", ylim=(50, 1000))
ax.legend(fontsize=font_size - 3, markerscale=3)
ax.grid(True, alpha=0.3)

# ── (G) Vx/Vy/Vz vs height — raw echoes ─────────────────────────────────────
ax = axes[2, 0]
if not df_vel_raw.empty:
    vr = df_vel_raw.dropna(subset=["vx_mps"])
    h  = vr["height_bin_km"].values
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.plot(vr["vx_mps"], h, "o-", color="tab:blue",   ms=5, label="Vx East")
    ax.plot(vr["vy_mps"], h, "s-", color="tab:orange", ms=5, label="Vy North")
    ax.plot(vr["vz_mps"], h, "^-", color="tab:green",  ms=5, label="Vz Up")
ax.set(xlabel="Velocity (m/s)", ylabel="Height (km)",
       title="(G) 3-D drift — raw", ylim=(50, 1000))
ax.legend(fontsize=font_size - 3)
ax.grid(True, alpha=0.3)

# ── (H) Vx/Vy/Vz vs height — filtered echoes ─────────────────────────────────
ax = axes[2, 1]
if not df_vel_filt.empty:
    vf = df_vel_filt.dropna(subset=["vx_mps"])
    h  = vf["height_bin_km"].values
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.plot(vf["vx_mps"], h, "o-", color="tab:blue",   ms=5, label="Vx East")
    ax.plot(vf["vy_mps"], h, "s-", color="tab:orange", ms=5, label="Vy North")
    ax.plot(vf["vz_mps"], h, "^-", color="tab:green",  ms=5, label="Vz Up")
ax.set(xlabel="Velocity (m/s)", ylabel="Height (km)",
       title="(H) 3-D drift — filtered", ylim=(50, 1000))
ax.legend(fontsize=font_size - 3)
ax.grid(True, alpha=0.3)

# ── (I) RMS residual and echo count: raw vs filtered ─────────────────────────
ax = axes[2, 2]
ax2 = ax.twiny()

if not df_vel_raw.empty:
    vr = df_vel_raw.dropna(subset=["residual_mps"])
    ax.plot(vr["residual_mps"], vr["height_bin_km"],
            "o--", color="tab:grey",   ms=4, label="RMS raw")
    ax2.plot(vr["n_echoes"], vr["height_bin_km"],
             "D:", color="tab:grey",   ms=4, alpha=0.5, label="N raw")

if not df_vel_filt.empty:
    vf = df_vel_filt.dropna(subset=["residual_mps"])
    ax.plot(vf["residual_mps"], vf["height_bin_km"],
            "o-",  color="tab:purple", ms=4, label="RMS filtered")
    ax2.plot(vf["n_echoes"], vf["height_bin_km"],
             "D-",  color="tab:red",    ms=4, label="N filtered")

ax.set_xlabel("RMS LOS residual (m/s)", fontsize=font_size - 1, color="tab:purple")
ax2.set_xlabel("Echoes per bin",        fontsize=font_size - 1, color="tab:red")
ax.set_ylabel("Height (km)",            fontsize=font_size - 1)
ax.set_title("(I) Fit quality: raw vs filtered", fontsize=font_size)
ax.set_ylim(50, 1000)

lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=font_size - 3, loc="lower right")
ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------

out = "docs/examples/figures/ionogram_full_analysis_wi937.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved → {out}")
