"""es_aggregator_example.py — Single-file sanity check + 20-file time-series RTI.

Figure 1 — sanity check (first file only, 4 panels):
  (a) Conventional RTI       pulse-by-pulse amplitude (Rx-averaged), 80–120 km
  (b) Capon Z=50  RTI        high-resolution RTI
  (c) Capon Z=100 RTI        higher-Z RTI
  (d) Mean height profile    conventional vs Capon Z=50 vs Capon Z=100 (1-D)

Figure 2 — time-series RTI (up to MAX_FILES files, 3 panels):
  (a) Conventional           mean amplitude per file
  (b) Capon Z=50             mean Capon spectrum per file
  (c) Capon Z=100            mean Capon spectrum per file

Output
------
docs/examples/figures/es_aggregator_example.png
docs/examples/figures/es_aggregator_timeseries.png
"""

import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pynasonde.vipir.analysis import EsCaponImager
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

# ── configuration ──────────────────────────────────────────────────────────────
RIQ_GLOB = "tmp/pynasonde_riq_test*/*.RIQ"  # pre-decompressed files
FREQ_TARGET_KHZ = 3000.0  # 3 MHz — safely below summer foEs at Wallops
VIPIR_VERSION_IDX = 1
BLANK_MIN_KM = 60.0  # zero gates below this height (direct-wave blanking)
MAX_FILES = 20  # maximum files for the time-series plot

# Capon parameters
Z50 = 50  # lower Z — partial improvement
Z100 = 100  # Z ≈ V/2 — optimal non-singular (V=960 → (V+1)//2=480)
K = 10  # resolution factor: Δr = r₀/K

# Display
YLIM_RTI = (80.0, 120.0)
YLIM_PROF = (80.0, 120.0)
VMIN_CONV = -40.0
VMIN_CAP = -30.0
CMAP = "jet"

_C_KM_US = 299_792.458 / 1e6  # km per μs


# ── helpers ────────────────────────────────────────────────────────────────────


def _load_cube(riq_path, freq_khz, version_idx):
    """Load Rx-averaged cube (n_pulse, n_gate) for the pulset closest to freq_khz.

    Per Liu et al. (2023), all snapshots fed to the Capon covariance must be at
    the same carrier frequency — the ionospheric response is frequency-dependent
    and mixing frequencies would contaminate the covariance matrix.
    """
    riq = RiqDataset.create_from_file(
        riq_path,
        unicode="latin-1",
        vipir_config=VIPIR_VERSION_MAP.configs[version_idx],
    )
    gs = riq.sct.timing.gate_step * _C_KM_US / 2
    g0 = riq.sct.timing.gate_start * _C_KM_US / 2

    freqs = np.array([float(ps.pcts[0].frequency) for ps in riq.pulsets])
    idx = int(np.argmin(np.abs(freqs - freq_khz)))
    freq_actual = freqs[idx]

    profiles = []
    for pct in riq.pulsets[idx].pcts:
        iq = pct.pulse_i.astype(np.float64) + 1j * pct.pulse_q.astype(np.float64)
        profiles.append(iq.mean(axis=-1) if iq.ndim == 2 else iq)
    cube = np.stack(profiles, axis=0)  # (n_pulse, n_gate)

    print(
        f"Loaded pulset #{idx}  f={freq_actual/1e3:.3f} MHz  "
        f"shape={cube.shape}  r₀={gs:.3f} km  start={g0:.2f} km"
    )
    return cube, gs, g0, freq_actual


def _blank(cube_2d, gate_blank):
    """Zero first gate_blank gates in a 2-D (n_pulse, n_gate) array."""
    if gate_blank <= 0:
        return cube_2d
    out = cube_2d.copy()
    out[:, :gate_blank] = 0.0
    return out


def _run_capon(cube_2d, z, k, gate_start_km, gate_spacing_km):
    """Run EsCaponImager on a (n_pulse, n_gate) Rx-averaged cube."""
    imager = EsCaponImager(
        n_subbands=z,
        resolution_factor=k,
        coherent_integrations=1,
        gate_start_km=gate_start_km,
        gate_spacing_km=gate_spacing_km,
    )
    return imager.fit(cube_2d)


def _renorm_window(spec_db, heights_km, ylim):
    """Re-normalise a (n_snap, n_hr) spectrum to its max within ylim."""
    mask = (heights_km >= ylim[0]) & (heights_km <= ylim[1])
    win = spec_db[:, mask]
    offset = win.max()
    return win - offset, heights_km[mask], offset


def _process_file(src):
    """Load one RIQ file and return per-pulse columns.  Called in parallel."""
    cube_i, r0_i, gs_i, _ = _load_cube(src, FREQ_TARGET_KHZ, VIPIR_VERSION_IDX)
    gb_i = max(0, int((BLANK_MIN_KM - gs_i) / r0_i))
    blk_i = _blank(cube_i, gb_i)

    conv_cols = 20 * np.log10(np.abs(blk_i) + 1e-15)  # (n_pulse, n_gate)
    r50_i = _run_capon(blk_i, Z50, K, gs_i, r0_i)
    r100_i = _run_capon(blk_i, Z100, K, gs_i, r0_i)

    return dict(
        label=os.path.basename(src)[:15],
        conv=conv_cols,  # (n_pulse, n_gate)
        cap50=r50_i.pseudospectrum_db,  # (n_pulse, n_hr)
        cap100=r100_i.pseudospectrum_db,  # (n_pulse, n_hr)
        h_native=gs_i + np.arange(cube_i.shape[1]) * r0_i,
        h_cap50=r50_i.heights_km,
        h_cap100=r100_i.heights_km,
        r0=r0_i,
        dr50=r50_i.effective_resolution_km,
        dr100=r100_i.effective_resolution_km,
    )


# ── load first file ────────────────────────────────────────────────────────────
all_paths = sorted(glob.glob(RIQ_GLOB))
if not all_paths:
    raise FileNotFoundError(f"No RIQ files matched: {RIQ_GLOB!r}")

cube, r0, gate_start, freq_hz = _load_cube(
    all_paths[0], FREQ_TARGET_KHZ, VIPIR_VERSION_IDX
)

n_pulse, n_gate = cube.shape
gate_blank = max(0, int((BLANK_MIN_KM - gate_start) / r0))
print(f"Gate blanking: first {gate_blank} gates zeroed (below {BLANK_MIN_KM:.0f} km)")

# ── conventional amplitude ─────────────────────────────────────────────────────
blanked = _blank(cube, gate_blank)
heights_native = gate_start + np.arange(n_gate) * r0

mean_amp = np.abs(blanked).mean(axis=0)
amp_db = 20 * np.log10(mean_amp + 1e-15)
amp_db_norm = amp_db - amp_db.max()

rti_conv = 20 * np.log10(np.abs(blanked) + 1e-15)
rti_conv -= rti_conv.max()

# ── Capon imaging ──────────────────────────────────────────────────────────────
print(f"\nRunning Capon Z={Z50}  K={K} …")
res50 = _run_capon(blanked, Z50, K, gate_start, r0)

print(f"Running Capon Z={Z100}  K={K} …")
res100 = _run_capon(blanked, Z100, K, gate_start, r0)

print(f"  Z={Z50}:  {res50.summary()}")
print(f"  Z={Z100}: {res100.summary()}")

# ── Figure 1: sanity check ─────────────────────────────────────────────────────
t_axis = np.arange(n_pulse)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
h_mask = (heights_native >= YLIM_RTI[0]) & (heights_native <= YLIM_RTI[1])
rti_win = rti_conv[:, h_mask]
rti_win -= rti_win.max()
ax.pcolormesh(
    t_axis,
    heights_native[h_mask],
    rti_win.T,
    cmap=CMAP,
    vmin=VMIN_CONV,
    vmax=0,
    shading="nearest",
)
ax.set_xlabel("Pulse index")
ax.set_ylabel("Virtual height (km)")
ax.set_ylim(*YLIM_RTI)
ax.set_title(f"(a) Conventional  r₀={r0:.2f} km")
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN_CONV, 0))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Normalised power (dB)", fraction=0.046, pad=0.04)

ax = axes[0, 1]
spec50, h50, _ = _renorm_window(res50.pseudospectrum_db, res50.heights_km, YLIM_RTI)
ax.pcolormesh(
    t_axis, h50, spec50.T, cmap=CMAP, vmin=VMIN_CAP, vmax=0, shading="nearest"
)
ax.set_xlabel("Pulse index")
ax.set_ylabel("Virtual height (km)")
ax.set_ylim(*YLIM_RTI)
ax.set_title(f"(b) Capon Z={Z50}  Δr={res50.effective_resolution_km*1e3:.0f} m")
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN_CAP, 0))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Normalised power (dB)", fraction=0.046, pad=0.04)

ax = axes[1, 0]
spec100, h100, _ = _renorm_window(res100.pseudospectrum_db, res100.heights_km, YLIM_RTI)
ax.pcolormesh(
    t_axis, h100, spec100.T, cmap=CMAP, vmin=VMIN_CAP, vmax=0, shading="nearest"
)
ax.set_xlabel("Pulse index")
ax.set_ylabel("Virtual height (km)")
ax.set_ylim(*YLIM_RTI)
ax.set_title(f"(c) Capon Z={Z100}  Δr={res100.effective_resolution_km*1e3:.0f} m")
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN_CAP, 0))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Normalised power (dB)", fraction=0.046, pad=0.04)

ax = axes[1, 1]
h_mask_prof = (heights_native >= YLIM_PROF[0]) & (heights_native <= YLIM_PROF[1])
ax.plot(
    amp_db_norm[h_mask_prof],
    heights_native[h_mask_prof],
    color="tab:orange",
    lw=1.4,
    label=f"Conventional  r₀={r0:.2f} km",
    zorder=3,
)
mean50 = res50.pseudospectrum_db.mean(axis=0)
mean50 -= mean50.max()
mask50 = (res50.heights_km >= YLIM_PROF[0]) & (res50.heights_km <= YLIM_PROF[1])
ax.plot(
    mean50[mask50],
    res50.heights_km[mask50],
    color="tab:green",
    lw=1.2,
    label=f"Capon Z={Z50}  Δr={res50.effective_resolution_km*1e3:.0f} m",
)
mean100 = res100.pseudospectrum_db.mean(axis=0)
mean100 -= mean100.max()
mask100 = (res100.heights_km >= YLIM_PROF[0]) & (res100.heights_km <= YLIM_PROF[1])
ax.plot(
    mean100[mask100],
    res100.heights_km[mask100],
    color="tab:blue",
    lw=1.2,
    label=f"Capon Z={Z100}  Δr={res100.effective_resolution_km*1e3:.0f} m",
)
ax.set_xlabel("Normalised power (dB)")
ax.set_ylabel("Virtual height (km)")
ax.set_xlim(-60, 2)
ax.set_ylim(*YLIM_PROF)
ax.set_title("(d) Mean pulse profile — conventional vs Capon")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"EsCaponImager — WI937  f={freq_hz/1e3:.3f} MHz  "
    f"K={K}  blanking<{BLANK_MIN_KM:.0f} km\n"
    f"V={n_gate}  n_pulse={n_pulse} (Rx-avg)  "
    f"r₀={r0:.3f} km → Δr={r0/K:.3f} km",
    fontsize=11,
)
fig.tight_layout()

out_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "docs", "examples", "figures"
)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "es_aggregator_example.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure 1 saved → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — time-series RTI over up to MAX_FILES files
# ══════════════════════════════════════════════════════════════════════════════
ts_paths = all_paths[:MAX_FILES]
print(f"\nTime-series: processing {len(ts_paths)} files in parallel …")

# Submit all files concurrently; preserve original file order in results.
# ThreadPoolExecutor is used so numpy (which releases the GIL) can run
# multiple Capon inversions simultaneously without pickling overhead.
ordered = {}  # index → result dict
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(_process_file, src): i for i, src in enumerate(ts_paths)}
    for future in as_completed(futures):
        i = futures[future]
        try:
            ordered[i] = future.result()
            print(f"  [{i+1:2d}/{len(ts_paths)}] done  {ordered[i]['label']}")
        except Exception as exc:
            print(f"  [{i+1:2d}/{len(ts_paths)}] skipped: {exc}")

# Re-sort by file index and unpack
results = [ordered[i] for i in sorted(ordered)]
ts_conv = [r["conv"] for r in results]
ts_cap50 = [r["cap50"] for r in results]
ts_cap100 = [r["cap100"] for r in results]
ts_labels = [r["label"] for r in results]
if results:
    h_native_ts = results[-1]["h_native"]
    h_cap50_ts = results[-1]["h_cap50"]
    h_cap100_ts = results[-1]["h_cap100"]
    r0_ts = results[-1]["r0"]
    dr50 = results[-1]["dr50"]
    dr100 = results[-1]["dr100"]

if ts_conv:
    n_files = len(ts_conv)
    n_pulse_per_file = ts_conv[0].shape[0]
    n_t = n_files * n_pulse_per_file  # e.g. 20 × 4 = 80
    t_idx = np.arange(n_t)

    # each element is (n_pulse, n_gate/n_hr) → concatenate → (n_files*n_pulse, ...)
    conv_mat = np.concatenate(ts_conv, axis=0)
    cap50_mat = np.concatenate(ts_cap50, axis=0)
    cap100_mat = np.concatenate(ts_cap100, axis=0)

    def _ts_norm(mat, heights):
        mask = (heights >= YLIM_RTI[0]) & (heights <= YLIM_RTI[1])
        win = mat[:, mask]
        return win - win.max(), heights[mask]

    conv_win, hc = _ts_norm(conv_mat, h_native_ts)
    cap50_win, hc5 = _ts_norm(cap50_mat, h_cap50_ts)
    cap100_win, hc1 = _ts_norm(cap100_mat, h_cap100_ts)

    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax = axes2[0]
    ax.pcolormesh(
        t_idx, hc, conv_win.T, cmap=CMAP, vmin=VMIN_CONV, vmax=0, shading="nearest"
    )
    ax.set_ylabel("Virtual height (km)")
    ax.set_ylim(*YLIM_RTI)
    ax.set_title(f"(a) Conventional  r₀={r0_ts:.2f} km")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN_CONV, 0))
    sm.set_array([])
    fig2.colorbar(sm, ax=ax, label="Normalised power (dB)", fraction=0.03, pad=0.02)

    ax = axes2[1]
    ax.pcolormesh(
        t_idx, hc5, cap50_win.T, cmap=CMAP, vmin=VMIN_CAP, vmax=0, shading="nearest"
    )
    ax.set_ylabel("Virtual height (km)")
    ax.set_ylim(*YLIM_RTI)
    ax.set_title(f"(b) Capon Z={Z50}  Δr={dr50*1e3:.0f} m")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN_CAP, 0))
    sm.set_array([])
    fig2.colorbar(sm, ax=ax, label="Normalised power (dB)", fraction=0.03, pad=0.02)

    ax = axes2[2]
    ax.pcolormesh(
        t_idx, hc1, cap100_win.T, cmap=CMAP, vmin=VMIN_CAP, vmax=0, shading="nearest"
    )
    ax.set_ylabel("Virtual height (km)")
    ax.set_ylim(*YLIM_RTI)
    ax.set_title(f"(c) Capon Z={Z100}  Δr={dr100*1e3:.0f} m")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN_CAP, 0))
    sm.set_array([])
    fig2.colorbar(sm, ax=ax, label="Normalised power (dB)", fraction=0.03, pad=0.02)

    # one tick per file, placed at the first pulse of each file
    file_ticks = np.arange(n_files) * n_pulse_per_file
    axes2[-1].set_xlabel("Pulse index  (ticks = file boundaries)")
    axes2[-1].set_xticks(file_ticks)
    axes2[-1].set_xticklabels(ts_labels, rotation=45, ha="right", fontsize=7)

    fig2.suptitle(
        f"EsCaponImager time-series — WI937  f={FREQ_TARGET_KHZ/1e3:.3f} MHz  "
        f"K={K}  blanking<{BLANK_MIN_KM:.0f} km\n"
        f"{n_files} files × {n_pulse_per_file} pulses = {n_t} time columns",
        fontsize=11,
    )
    fig2.tight_layout()

    out_path2 = os.path.join(out_dir, "es_aggregator_timeseries.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Figure 2 saved → {out_path2}")
else:
    print("No files processed — time-series figure skipped.")
