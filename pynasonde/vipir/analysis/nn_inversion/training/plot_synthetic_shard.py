"""Synthetic-shard QC plot.

Loads one NetCDF shard produced by synthetic_data.py and renders a 6-panel
diagnostic figure using ngi.plotlib.Ionogram.

Panels
------
(0,0) Sampled Ne profiles — up to N_SHOW random profiles overlaid
(0,1) Corresponding h'(f) traces — ionogram-style (log10 freq axis)
(0,2) foF2 distribution histogram
(1,0) hmF2 distribution histogram
(1,1) Kp distribution (actual values from IRI date index)
(1,2) F10.7 distribution

CLI
---
    conda activate pynasonde
    python plot_synthetic_shard.py \\
        --shard /tmp/nn_polan_test_nc/shard_00359.nc \\
        --out_dir /tmp/nn_polan_figs \\
        --n_show 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pynasonde.vipir.analysis.nn_inversion.forward_model import find_foF2, ne_to_fp
from pynasonde.vipir.ngi.plotlib import Ionogram

N_SHOW_DEFAULT = 30  # profiles to overlay in panels (0,0) and (0,1)


def plot_synthetic_shard(
    shard_path: str | Path,
    out_dir: str | Path = ".",
    n_show: int = N_SHOW_DEFAULT,
    show: bool = False,
) -> Path:
    """Generate and save the shard QC figure.

    Parameters
    ----------
    shard_path : path to a shard NetCDF file
    out_dir    : output directory (created if absent)
    n_show     : number of random samples to overlay in Ne / trace panels
    show       : call plt.show() before saving

    Returns
    -------
    Path to the saved PNG.
    """
    shard_path = Path(shard_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shard ─────────────────────────────────────────────────────────────
    ds = xr.open_dataset(shard_path)

    ne_all = ds["ne_cm3"].values.astype(np.float64)  # (N, N_h)
    hv_all = ds["h_virtual"].values.astype(np.float32)  # (N, N_f)
    mask_all = ds["obs_mask"].values.astype(bool)  # (N, N_f)
    cond_all = ds["cond"].values.astype(np.float32)  # (N, 6)
    h_grid = ds["height_km"].values  # (N_h,)
    f_grid = ds["freq_mhz"].values  # (N_f,)
    cond_cols = list(ds["cond_dim"].values)

    # cond column indices
    kp_idx = cond_cols.index("kp")
    f107_idx = cond_cols.index("f107_sfu")

    n_total = len(ne_all)
    n_show = min(n_show, n_total)
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(n_total, size=n_show, replace=False)

    # Derived scalars for all profiles
    fp_all = ne_to_fp(ne_all)  # (N, N_h) MHz
    foF2_all = fp_all.max(axis=1)  # (N,) MHz
    hmF2_all = h_grid[fp_all.argmax(axis=1)]  # (N,) km
    kp_all = cond_all[:, kp_idx]
    f107_all = cond_all[:, f107_idx]

    shard_id = ds.attrs.get("shard", shard_path.stem)

    # ── Build figure ──────────────────────────────────────────────────────────
    ion = Ionogram(
        fig_title=f"NN-POLAN Synthetic Shard {shard_id}  (N={n_total})",
        nrows=2,
        ncols=3,
        font_size=9,
        figsize=(5, 3.5),
    )

    cmap_lines = plt.cm.plasma(np.linspace(0.15, 0.85, n_show))

    # ── Panel (0,0): sampled Ne profiles ──────────────────────────────────────
    ax0 = ion._add_axis(del_ticks=False)
    for k, i in enumerate(idx):
        ax0.plot(ne_all[i], h_grid, color=cmap_lines[k], lw=0.6, alpha=0.7)
    ax0.set_xlabel(r"$N_e$ [cm$^{-3}$]", fontsize=9)
    ax0.set_ylabel("Altitude [km]", fontsize=9)
    ax0.set_ylim(h_grid[0], 500)
    ax0.set_xscale("log")
    ax0.set_xlim(1e1, ne_all.max() * 2)  # clip near-zero noise floor
    ax0.set_title(f"(a) Ne profiles (n={n_show})", fontsize=9)

    # ── Panel (0,1): h'(f) traces — ionogram style ────────────────────────────
    ax1 = ion._add_axis(del_ticks=False)
    xticks = [
        t for t in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15] if f_grid[0] <= t <= f_grid[-1]
    ]
    for k, i in enumerate(idx):
        hv = hv_all[i]
        msk = mask_all[i]
        if msk.any():
            ax1.plot(
                np.log10(f_grid[msk]), hv[msk], color=cmap_lines[k], lw=0.7, alpha=0.7
            )
    ax1.set_xlabel("Frequency [MHz]", fontsize=9)
    ax1.set_ylabel("Virtual Height h' [km]", fontsize=9)
    ax1.set_ylim(h_grid[0], 500)
    ax1.set_xlim(np.log10(f_grid[0]), np.log10(f_grid[-1]))
    ax1.set_xticks(np.log10(xticks))
    ax1.set_xticklabels(xticks)
    ax1.set_title(f"(b) h'(f) traces (n={n_show})", fontsize=9)

    # ── Panel (0,2): foF2 histogram ───────────────────────────────────────────
    ax2 = ion._add_axis(del_ticks=False)
    ax2.hist(foF2_all, bins=30, color="steelblue", edgecolor="white", lw=0.4)
    ax2.set_xlabel(r"$foF_2$ [MHz]", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.axvline(
        np.median(foF2_all),
        color="tomato",
        lw=1.2,
        ls="--",
        label=f"Median {np.median(foF2_all):.1f} MHz",
    )
    ax2.legend(fontsize=7)
    ax2.set_title(r"(c) $foF_2$ distribution", fontsize=9)

    # ── Panel (1,0): hmF2 histogram ───────────────────────────────────────────
    ax3 = ion._add_axis(del_ticks=False)
    ax3.hist(hmF2_all, bins=30, color="darkorange", edgecolor="white", lw=0.4)
    ax3.set_xlabel(r"$hmF_2$ [km]", fontsize=9)
    ax3.set_ylabel("Count", fontsize=9)
    ax3.axvline(
        np.median(hmF2_all),
        color="navy",
        lw=1.2,
        ls="--",
        label=f"Median {np.median(hmF2_all):.0f} km",
    )
    ax3.legend(fontsize=7)
    ax3.set_title(r"(d) $hmF_2$ distribution", fontsize=9)

    # ── Panel (1,1): Kp distribution ──────────────────────────────────────────
    ax4 = ion._add_axis(del_ticks=False)
    ax4.hist(kp_all, bins=20, color="purple", edgecolor="white", lw=0.4)
    ax4.set_xlabel("Kp index", fontsize=9)
    ax4.set_ylabel("Count", fontsize=9)
    ax4.axvline(
        np.median(kp_all),
        color="gold",
        lw=1.2,
        ls="--",
        label=f"Median {np.median(kp_all):.1f}",
    )
    ax4.legend(fontsize=7)
    ax4.set_title("(e) Kp distribution", fontsize=9)

    # ── Panel (1,2): F10.7 distribution ──────────────────────────────────────
    ax5 = ion._add_axis(del_ticks=False)
    ax5.hist(
        f107_all,
        bins=len(np.unique(f107_all)),
        color="forestgreen",
        edgecolor="white",
        lw=0.4,
    )
    ax5.set_xlabel("F10.7 [SFU]", fontsize=9)
    ax5.set_ylabel("Count", fontsize=9)
    ax5.set_title("(f) F10.7 distribution", fontsize=9)

    ion.fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()

    out_path = out_dir / f"synthetic_shard_{shard_id}.png"
    ion.save(str(out_path))
    ion.close()
    ds.close()
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="QC plot for a synthetic NN-POLAN shard (.nc)."
    )
    p.add_argument(
        "--shard", required=True, help="Path to a shard NetCDF file (shard_XXXXX.nc)."
    )
    p.add_argument(
        "--out_dir",
        default=".",
        help="Output directory for PNG (default: current dir).",
    )
    p.add_argument(
        "--n_show",
        type=int,
        default=N_SHOW_DEFAULT,
        help=f"Profiles to overlay in Ne/trace panels (default {N_SHOW_DEFAULT}).",
    )
    p.add_argument("--show", action="store_true", help="Call plt.show() before saving.")
    args = p.parse_args()
    plot_synthetic_shard(
        shard_path=args.shard,
        out_dir=args.out_dir,
        n_show=args.n_show,
        show=args.show,
    )


if __name__ == "__main__":
    _cli()
