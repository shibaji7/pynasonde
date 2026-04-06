"""es_imaging_sanity_check.py — Reproduce Liu et al. (2023) Figure 1 simulation.

This script validates EsCaponImager against the published simulation results.

Setup (Table 1 + Section 3 of the paper):
- r₀ = 3.84 km  (WISS bit duration 25.6 μs → c·t_p/2)
- V  = 200 range bins  (1st–200th bins, L = 768 km)
- N  = 256 duplicate soundings (pulses)
- Two reflectors at D₁ = 110 km and D₂ = 112 km
  (Q₁ = D₁/r₀ ≈ 28.65,  Q₂ ≈ 29.17 — both within the same 3.84-km gate bin)
- K  = 10  → effective resolution 384 m (Δr = r₀/K)
- Z  = 50, 100, 150 (panels b, c, d)
- SNR = 20 dB for the main panels; −5 … 20 dB for panel (f)

Singularity constraint (from paper Eq. 11):
  R_f = G·G^H/(V-Z+1) is Z×Z with rank ≤ (V-Z+1).
  Non-singular requires  Z ≤ (V+1)/2 = 100 for V=200.
  Z=150 → rank-deficient → imaging deteriorates (Fig 1d).
  K has NO singularity constraint — it only sets the output grid spacing.

The synthetic range profile is constructed as:
    G_ss[m] = A₁·exp(j·2π·Q₁·m/V) + A₂·exp(j·2π·Q₂·m/V)   m = 0…V-1
    R_ss    = IFFT{G_ss}   ← what is fed to EsCaponImager.fit()

Output: docs/examples/figures/es_imaging_sanity_check.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── path ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from pynasonde.vipir.analysis import EsCaponImager

# ── simulation parameters (WISS / Liu et al. 2023) ────────────────────────────
R0_KM = 3.84  # intrinsic gate spacing (km)
V = 200  # number of range bins
N_PULSES = 100  # duplicate soundings
K = 10  # resolution factor → Δr = R0_KM / K = 384 m
D1_KM = 110.0  # reflector 1 range (km)
D2_KM = 112.0  # reflector 2 range — 2 km separation, within one bin
Q1 = D1_KM / R0_KM  # ≈ 28.65
Q2 = D2_KM / R0_KM  # ≈ 29.17


# ── synthetic IQ cube construction ────────────────────────────────────────────


def _make_cube(
    n_pulses: int, v: int, q1: float, q2: float, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Return complex IQ cube (n_pulses, v) for two reflectors.

    G_ss[m] = exp(j·2π·q1·m/v) + exp(j·2π·q2·m/v)  (equal amplitude)
    R_ss    = IFFT{G_ss}   — the pulse-compressed range profile fed to fit().
    Noise is added in the R_ss domain at the specified SNR.
    """
    m = np.arange(v, dtype=float)
    G_clean = np.exp(1j * 2 * np.pi * q1 * m / v) + np.exp(1j * 2 * np.pi * q2 * m / v)
    R_clean = np.fft.ifft(G_clean)  # shape (v,)

    sig_pwr = np.mean(np.abs(R_clean) ** 2)
    noise_std = np.sqrt(sig_pwr / 10 ** (snr_db / 10) / 2)

    cube = np.stack(
        [
            R_clean + noise_std * (rng.standard_normal(v) + 1j * rng.standard_normal(v))
            for _ in range(n_pulses)
        ],
        axis=0,
    )
    return cube  # (n_pulses, v)


# ── helper: run Capon for a given Z ───────────────────────────────────────────


def _run_capon(cube: np.ndarray, z: int) -> "EsImagingResult":
    imager = EsCaponImager(
        n_subbands=z,
        resolution_factor=K,
        coherent_integrations=1,  # per-pulse (all 256 snapshots)
        gate_start_km=0.0,
        gate_spacing_km=R0_KM,
    )
    return imager.fit(cube)


# ── helpers: conventional amplitude image ────────────────────────────────────


def _raw_amp_db(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_pulses, V) amplitude image normalised to 0 dB and gate heights."""
    amp = 20 * np.log10(np.abs(cube) + 1e-15)
    amp -= amp.max()
    gates = np.arange(V) * R0_KM  # km
    return amp, gates


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "docs", "examples", "figures"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── generate synthetic data ──────────────────────────────────────────
    cube_20dB = _make_cube(N_PULSES, V, Q1, Q2, snr_db=20.0, rng=rng)

    # ── run imagers ──────────────────────────────────────────────────────
    print("Running Capon Z=50  (K=10) …")
    res_z50 = _run_capon(cube_20dB, z=50)
    print("Running Capon Z=100 (K=10) …")
    res_z100 = _run_capon(cube_20dB, z=100)
    print("Running Capon Z=150 (K=10) …")
    res_z150 = _run_capon(cube_20dB, z=150)

    print(
        f"  Z=50  → K={res_z50.resolution_factor}  "
        f"Δr={res_z50.effective_resolution_km*1e3:.0f} m  "
        f"n_hr={res_z50.pseudospectrum_db.shape[1]}"
    )
    print(
        f"  Z=100 → K={res_z100.resolution_factor}  "
        f"Δr={res_z100.effective_resolution_km*1e3:.0f} m  "
        f"n_hr={res_z100.pseudospectrum_db.shape[1]}"
    )
    print(
        f"  Z=150 → K={res_z150.resolution_factor}  "
        f"Δr={res_z150.effective_resolution_km*1e3:.0f} m  "
        f"n_hr={res_z150.pseudospectrum_db.shape[1]}"
    )

    amp_raw, gate_h = _raw_amp_db(cube_20dB)
    t_axis = np.arange(N_PULSES)

    # ── figure ───────────────────────────────────────────────────────────
    vmin_rti = -30  # conventional image colour range
    vmin_cap = -150  # Capon image colour range (matches paper)
    cmap = "jet"
    ylim = (100, 150)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    def _rtipanel(ax, t, h, spec_T, vmin, title):
        """pcolormesh RTI panel with correct dimensions."""
        ax.pcolormesh(t, h, spec_T, cmap=cmap, vmin=vmin, vmax=0, shading="nearest")
        for hh in [D1_KM, D2_KM]:
            ax.axhline(hh, color="white", lw=0.8, ls="--", alpha=0.6)
        ax.set(
            xlabel="Pulse index", ylabel="Virtual height (km)", title=title, ylim=ylim
        )

    # (a) conventional
    _rtipanel(
        axes[0, 0],
        t_axis,
        gate_h,
        amp_raw.T,
        vmin_rti,
        f"(a) Conventional  r₀={R0_KM} km",
    )

    # (b) Z=50
    _rtipanel(
        axes[0, 1],
        t_axis,
        res_z50.heights_km,
        res_z50.pseudospectrum_db.T,
        vmin_cap,
        f"(b) Capon Z=50  Δr={res_z50.effective_resolution_km*1e3:.0f} m",
    )
    print(f"  Z=50: effective shape {res_z50.pseudospectrum_db.shape}")

    # (c) Z=100
    _rtipanel(
        axes[1, 0],
        t_axis,
        res_z100.heights_km,
        res_z100.pseudospectrum_db.T,
        vmin_cap,
        f"(c) Capon Z=100  Δr={res_z100.effective_resolution_km*1e3:.0f} m",
    )

    # (d) Z=150
    _rtipanel(
        axes[1, 1],
        t_axis,
        res_z150.heights_km,
        res_z150.pseudospectrum_db.T,
        vmin_cap,
        f"(d) Capon Z=150  Δr={res_z150.effective_resolution_km*1e3:.0f} m  "
        f"[rank-deficient — expected degradation]",
    )

    # (e) first-pulse slice: original vs Z=100
    ax = axes[2, 0]
    single_raw = 20 * np.log10(np.abs(cube_20dB[0]) + 1e-15)
    single_raw -= single_raw.max()
    ax.plot(
        gate_h,
        single_raw,
        "o-",
        ms=4,
        color="tab:orange",
        label="Original (gate)",
        zorder=3,
    )
    # overlay all three Capon results for the first pulse
    for res, lbl, col in [
        (res_z50, "Capon Z=50", "tab:green"),
        (res_z100, "Capon Z=100", "tab:blue"),
        (res_z150, "Capon Z=150", "tab:red"),
    ]:
        ax.plot(res.heights_km, res.pseudospectrum_db[0], lw=1.2, color=col, label=lbl)
    for hh in [D1_KM, D2_KM]:
        ax.axvline(hh, color="gray", lw=0.8, ls="--", alpha=0.8)
    ax.set(
        xlabel="Virtual height (km)",
        ylabel="Normalised intensity (dB)",
        title="(e) First-pulse slice  [original vs Capon]",
        xlim=(100, 150),
        ylim=(-100, 5),
    )
    ax.legend(fontsize=8)

    # (f) SNR scan — Z=100, first-pulse slice, 105–115 km
    ax = axes[2, 1]
    snr_vals = [-5, 0, 5, 10, 15, 20]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snr_vals)))
    for snr, col in zip(snr_vals, colors):
        cube_snr = _make_cube(N_PULSES, V, Q1, Q2, snr_db=snr, rng=rng)
        res_snr = _run_capon(cube_snr, z=100)
        mask = (res_snr.heights_km >= 105) & (res_snr.heights_km <= 115)
        ax.plot(
            res_snr.heights_km[mask],
            res_snr.pseudospectrum_db[0][mask],
            color=col,
            lw=1.2,
            label=f"SNR={snr:+d} dB",
        )
        print(f"  SNR={snr:+3d} dB  done")
    for hh in [D1_KM, D2_KM]:
        ax.axvline(hh, color="gray", lw=0.8, ls="--", alpha=0.8)
    ax.set(
        xlabel="Virtual height (km)",
        ylabel="Normalised intensity (dB)",
        title=f"(f) SNR scan  Z=100  K=10  (first pulse, 105–115 km)",
        xlim=(105, 115),
        ylim=(-60, 5),
    )
    ax.legend(fontsize=7, ncol=2)

    # ── colorbars ─────────────────────────────────────────────────────────
    for ax_obj, vmin in [
        (axes[0, 0], vmin_rti),
        (axes[0, 1], vmin_cap),
        (axes[1, 0], vmin_cap),
        (axes[1, 1], vmin_cap),
    ]:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=0))
        sm.set_array([])
        fig.colorbar(
            sm, ax=ax_obj, label="Normalised spectrum (dB)", fraction=0.046, pad=0.04
        )

    fig.suptitle(
        "Liu et al. (2023) Fig. 1 reproduction — EsCaponImager sanity check\n"
        f"Reflectors at {D1_KM:.0f} km & {D2_KM:.0f} km  "
        f"(separation {D2_KM-D1_KM:.0f} km < r₀={R0_KM} km),  "
        f"V={V},  N={N_PULSES},  K={K},  SNR=20 dB",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    out_path = os.path.join(out_dir, "es_imaging_sanity_check.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")

    _quantitative_checks(res_z50, res_z100, res_z150)


def _quantitative_checks(res_z50, res_z100, res_z150) -> None:
    """Print assertions matching the paper's qualitative claims."""
    print("\n── Quantitative checks ──────────────────────────────────────────")

    for res, label in [(res_z50, "Z=50 "), (res_z100, "Z=100"), (res_z150, "Z=150")]:
        mask = (res.heights_km >= 105) & (res.heights_km <= 120)
        h = res.heights_km[mask]
        s = res.pseudospectrum_db[0][mask]

        # simple local-maximum finder
        pk_idx = [
            i
            for i in range(1, len(s) - 1)
            if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > -50
        ]
        pk_h = h[pk_idx] if pk_idx else np.array([])

        print(
            f"\n  {label}: Δr={res.effective_resolution_km*1e3:.0f} m  "
            f"max_in_roi={s.max():.1f} dB"
        )
        if len(pk_h) >= 2:
            top2 = np.sort(pk_h[np.argsort(s[pk_idx])[-2:]])
            sep = top2[-1] - top2[0]
            err1 = abs(top2[0] - D1_KM)
            err2 = abs(top2[1] - D2_KM)
            print(
                f"    Two peaks: {top2[0]:.2f} km, {top2[1]:.2f} km  "
                f"(sep={sep:.2f} km, true={D2_KM-D1_KM:.1f} km)"
            )
            print(f"    Peak errors vs true: {err1:.2f} km, {err2:.2f} km")
            resolved = err1 < R0_KM and err2 < R0_KM
            print(f"    Resolved within r₀: {'✓ YES' if resolved else '✗ NO'}")
        elif len(pk_h) == 1:
            print(f"    Single peak at {pk_h[0]:.2f} km — cannot resolve two layers")
        else:
            print(f"    No peaks above −50 dB — imaging degraded (expected Z=150)")

    print("\n── Expected (from Liu et al. 2023 Fig. 1) ──────────────────────")
    print("  Z=50:  partial improvement; may not fully separate 110/112 km")
    print("  Z=100: two peaks clearly resolved ~110 and ~112 km (Fig 1c, 1e)")
    print("  Z=150: covariance rank-deficient → degraded output (Fig 1d)")
    print()


if __name__ == "__main__":
    main()
