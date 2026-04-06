"""Forward-model validation plots.

Produces a 6-panel diagnostic figure that validates the Abel integral
implementation in forward_model.py using synthetic Chapman Ne profiles.

Panels
------
(0,0) Ne(h) profiles — three Chapman test cases overlaid
(0,1) h'(f) traces (forward_scalar) for each profile — ionogram-style
(0,2) Scalar vs batch comparison for one profile
(1,0) |scalar − batch| residual vs frequency
(1,1) Forward traces for all test profiles overlaid (batch result)
(1,2) foF2 recovery: plasma-frequency profile fₚ(h) vs altitude

CLI
---
    conda activate pynasonde
    python plot_forward_model.py --out_dir /tmp/nn_polan_figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pynasonde.vipir.analysis.nn_inversion.forward_model import (
    F_GRID_MHZ,
    H_GRID_KM,
    find_foF2,
    forward_batch,
    forward_scalar,
    fp_to_ne,
    ne_to_fp,
)
from pynasonde.vipir.ngi.plotlib import Ionogram

# ---------------------------------------------------------------------------
# Test profiles — synthetic Chapman layers
# ---------------------------------------------------------------------------


def _chapman(h: np.ndarray, NmF2: float, hmF2: float, H: float) -> np.ndarray:
    """Single Chapman α-layer.  N in cm⁻³, h/hmF2/H in km."""
    z = (h - hmF2) / H
    return NmF2 * np.exp(0.5 * (1.0 - z - np.exp(-z)))


def _build_test_profiles() -> tuple[np.ndarray, list[str]]:
    """Return (3, N_h) test Ne array and labels."""
    h = H_GRID_KM
    # Quiet, moderate, active
    profiles = np.stack(
        [
            _chapman(h, NmF2=3e5, hmF2=270.0, H=50.0),  # quiet,    foF2 ≈ 4.9 MHz
            _chapman(h, NmF2=6e5, hmF2=300.0, H=60.0),  # moderate, foF2 ≈ 6.9 MHz
            _chapman(h, NmF2=1e6, hmF2=330.0, H=70.0),  # active,   foF2 ≈ 8.9 MHz
        ],
        axis=0,
    ).astype(np.float64)
    labels = ["Quiet (~4.9 MHz)", "Moderate (~6.9 MHz)", "Active (~8.9 MHz)"]
    return profiles, labels


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

COLORS = ["steelblue", "darkorange", "forestgreen"]


def plot_forward_model_validation(
    out_dir: str | Path = ".",
    show: bool = False,
) -> Path:
    """Generate and save the forward-model validation figure.

    Parameters
    ----------
    out_dir : output directory (created if absent)
    show    : call plt.show() before saving

    Returns
    -------
    Path to the saved PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles, labels = _build_test_profiles()

    # Scalar h'(f) for each profile
    scalar_traces = np.array([forward_scalar(p) for p in profiles])  # (3, N_f)

    # Batch h'(f)
    batch_traces = forward_batch(profiles)  # (3, N_f)

    # Residual for profile 0
    residual = np.abs(scalar_traces[0] - batch_traces[0])  # (N_f,)

    # Plasma-frequency profiles
    fp_profiles = ne_to_fp(profiles)  # (3, N_h) MHz

    # ── Build figure ──────────────────────────────────────────────────────────
    ion = Ionogram(
        fig_title="NN-POLAN: Forward Model Validation",
        nrows=2,
        ncols=3,
        font_size=9,
        figsize=(5, 3.5),
    )

    # ── Panel (0,0): Ne profiles ───────────────────────────────────────────────
    ax0 = ion._add_axis(del_ticks=False)
    for i, (ne, lbl) in enumerate(zip(profiles, labels)):
        ax0.plot(ne, H_GRID_KM, color=COLORS[i], lw=1.2, label=lbl)
    ax0.set_xlabel(r"$N_e$ [cm$^{-3}$]", fontsize=9)
    ax0.set_ylabel("Altitude [km]", fontsize=9)
    ax0.set_ylim(60, 500)
    ax0.set_xscale("log")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.set_title("(a) Input Ne profiles", fontsize=9)

    # ── Panel (0,1): h'(f) scalar traces — ionogram style ────────────────────
    ax1 = ion._add_axis(del_ticks=False)
    xticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    for i, (hv, lbl) in enumerate(zip(scalar_traces, labels)):
        valid = np.isfinite(hv)
        ax1.plot(
            np.log10(F_GRID_MHZ[valid]), hv[valid], color=COLORS[i], lw=1.2, label=lbl
        )
    ax1.set_xlabel("Frequency [MHz]", fontsize=9)
    ax1.set_ylabel("Virtual Height [km]", fontsize=9)
    ax1.set_ylim(60, 500)
    ax1.set_xlim(np.log10(F_GRID_MHZ[0]), np.log10(F_GRID_MHZ[-1]))
    _ticks = [t for t in xticks if F_GRID_MHZ[0] <= t <= F_GRID_MHZ[-1]]
    ax1.set_xticks(np.log10(_ticks))
    ax1.set_xticklabels(_ticks)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title("(b) h'(f) scalar (reference)", fontsize=9)

    # ── Panel (0,2): Scalar vs batch overlay for profile 1 ───────────────────
    ax2 = ion._add_axis(del_ticks=False)
    i = 1  # moderate profile
    v_sc = np.isfinite(scalar_traces[i])
    v_bt = np.isfinite(batch_traces[i])
    ax2.plot(
        np.log10(F_GRID_MHZ[v_sc]),
        scalar_traces[i][v_sc],
        color="steelblue",
        lw=1.5,
        label="Scalar",
    )
    ax2.plot(
        np.log10(F_GRID_MHZ[v_bt]),
        batch_traces[i][v_bt],
        color="tomato",
        lw=1.0,
        ls="--",
        label="Batch",
    )
    ax2.set_xlabel("Frequency [MHz]", fontsize=9)
    ax2.set_ylabel("Virtual Height [km]", fontsize=9)
    ax2.set_ylim(60, 500)
    ax2.set_xlim(np.log10(F_GRID_MHZ[0]), np.log10(F_GRID_MHZ[-1]))
    ax2.set_xticks(np.log10(_ticks))
    ax2.set_xticklabels(_ticks)
    ax2.legend(fontsize=7)
    ax2.set_title("(c) Scalar vs Batch (moderate)", fontsize=9)

    # ── Panel (1,0): Residual |scalar − batch| ────────────────────────────────
    ax3 = ion._add_axis(del_ticks=False)
    valid_both = np.isfinite(scalar_traces[i]) & np.isfinite(batch_traces[i])
    ax3.semilogy(F_GRID_MHZ[valid_both], residual[valid_both], color="purple", lw=1.2)
    ax3.axhline(0.5, color="gray", lw=0.8, ls="--", label="0.5 km")
    ax3.set_xlabel("Frequency [MHz]", fontsize=9)
    ax3.set_ylabel("|Scalar − Batch| [km]", fontsize=9)
    ax3.legend(fontsize=7)
    ax3.set_title("(d) Residual (profile 1)", fontsize=9)

    # ── Panel (1,1): Batch traces, all profiles ───────────────────────────────
    ax4 = ion._add_axis(del_ticks=False)
    for i_p, (hv, lbl) in enumerate(zip(batch_traces, labels)):
        valid = np.isfinite(hv)
        ax4.plot(
            np.log10(F_GRID_MHZ[valid]), hv[valid], color=COLORS[i_p], lw=1.2, label=lbl
        )
    ax4.set_xlabel("Frequency [MHz]", fontsize=9)
    ax4.set_ylabel("Virtual Height [km]", fontsize=9)
    ax4.set_ylim(60, 500)
    ax4.set_xlim(np.log10(F_GRID_MHZ[0]), np.log10(F_GRID_MHZ[-1]))
    ax4.set_xticks(np.log10(_ticks))
    ax4.set_xticklabels(_ticks)
    ax4.legend(fontsize=7, loc="upper left")
    ax4.set_title("(e) h'(f) batch (all profiles)", fontsize=9)

    # ── Panel (1,2): fₚ(h) profiles ──────────────────────────────────────────
    ax5 = ion._add_axis(del_ticks=False)
    for i_p, (fp, lbl) in enumerate(zip(fp_profiles, labels)):
        foF2 = float(fp.max())
        ax5.plot(fp, H_GRID_KM, color=COLORS[i_p], lw=1.2, label=f"{lbl}")
        ax5.axvline(foF2, color=COLORS[i_p], lw=0.7, ls=":", alpha=0.8)
    ax5.set_xlabel(r"Plasma Freq $f_p$ [MHz]", fontsize=9)
    ax5.set_ylabel("Altitude [km]", fontsize=9)
    ax5.set_ylim(60, 500)
    ax5.set_xlim(0, F_GRID_MHZ[-1] + 1)
    ax5.set_title(r"(f) $f_p(h)$ profiles", fontsize=9)

    ion.fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()

    out_path = out_dir / "forward_model_validation.png"
    ion.save(str(out_path))
    ion.close()
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Plot forward-model validation for NN-POLAN."
    )
    p.add_argument("--out_dir", default=".", help="Output directory for PNG")
    p.add_argument("--show", action="store_true", help="Call plt.show()")
    args = p.parse_args()
    plot_forward_model_validation(out_dir=args.out_dir, show=args.show)


if __name__ == "__main__":
    _cli()
