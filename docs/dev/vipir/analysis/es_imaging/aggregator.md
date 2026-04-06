# `aggregator.py` — RiqAggregator

<div class="hero">
  <h3>Multi-File Es Layer Imager</h3>
  <p>
    Loads multiple VIPIR RIQ files at a target frequency and produces a
    high-resolution Capon pseudospectrum RTI using two output modes:
    per-file (one column per file) and moving-average (sliding window of
    N consecutive files averaged together).
  </p>
</div>

## Source

`pynasonde/vipir/analysis/es_imaging/aggregator.py`

## API reference

::: pynasonde.vipir.analysis.es_imaging.aggregator
    options:
      members:
        - RiqAggregator
      show_root_heading: true
      show_source: true

---

## `RiqAggregator`

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_subbands` | `int` | `100` | Capon Z — passed to internal `EsCaponImager` |
| `resolution_factor` | `int` | `10` | Capon K — output grid = K·V high-res bins |
| `rx_weights` | `ndarray \| None` | `None` | Reserved for future use.  All Rx channels are currently stacked as independent snapshots for the multi-snapshot covariance estimator |
| `gate_start_km` | `float` | `0.0` | Height of first gate (km).  Overridden from RIQ header inside `load()` |
| `gate_spacing_km` | `float` | `1.499` | Native gate spacing r₀ (km).  Overridden from RIQ header inside `load()` |
| `diagonal_loading` | `float` | `1e-3` | Capon ε regularisation |
| `output_mode` | `str` | `"per_file"` | `"per_file"` → one column per file; `"moving_avg"` → sliding-window average |
| `window` | `int` | `8` | Number of files per averaged column (`moving_avg` only) |
| `step` | `int` | `1` | Sliding step in files (`moving_avg` only) |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `load(file_list, freq_target_khz, freq_tol_khz=50.0, vipir_version_idx=1) → list[ndarray]` | Load IQ cubes from RIQ files; update gate geometry from first file's header |
| `combine` | `combine(cubes) → EsImagingResult` | Produce an imaging result from pre-loaded cubes |
| `fit` | `fit(file_list, freq_target_khz, freq_tol_khz=50.0, vipir_version_idx=1) → EsImagingResult` | Convenience: `load()` + `combine()` in one call |

---

## Multi-snapshot Capon covariance

Instead of beamforming Rx channels and running Capon per pulse, the aggregator
treats every **(pulse, Rx) pair** as an independent range-profile snapshot and
**averages all L covariance matrices before inverting once**:

```
R_f = (1 / L·cols) Σ_{l=1}^{L} G_l · G_l^H
```

where `G_l` is the Hankel subband matrix (shape `Z × cols`, `cols = V − Z + 1`)
of profile `l`.  This is the true multi-snapshot Capon estimator.

| Mode | L (snapshots per output column) |
|------|---------------------------------|
| `"per_file"` | `n_pulse × n_rx` (e.g. 4 × 8 = **32**) |
| `"moving_avg"` | `window × n_pulse × n_rx` (e.g. 8 × 4 × 8 = **256**) |

A better-conditioned R_f produces a cleaner Capon pseudospectrum, making weak
Es echoes (typically 40–60 dB below the direct-wave clutter) visible in the
normalised output.

---

## Output modes

| `output_mode` | `pseudospectrum_db` shape | `n_snapshots` | Use case |
|---------------|--------------------------|---------------|----------|
| `"per_file"` | `(n_files, K·V)` | `n_files` | One column per file, ~1 min cadence RTI |
| `"moving_avg"` | `((N-W)//S+1, K·V)` | `(N-W)//S+1` | Smoothed RTI; W files averaged per column |

Where N = number of files, W = `window`, S = `step`.

---

## Usage

### Per-file RTI

```python
from pynasonde.vipir.analysis import RiqAggregator
import glob

file_list = sorted(glob.glob("data/20230601_01??.RIQ"))

agg = RiqAggregator(
    n_subbands=100,
    resolution_factor=10,
    output_mode="per_file",
)
result = agg.fit(file_list, freq_target_khz=5000.0, vipir_version_idx=1)

print(result.summary())
# EsImagingResult: snapshots=60  Z=100  K=10  r₀=1.499 km → Δr=0.150 km
result.plot()   # 60-column RTI
```

### Moving-average RTI

```python
agg = RiqAggregator(
    n_subbands=100,
    resolution_factor=10,
    output_mode="moving_avg",
    window=8,   # average 8 consecutive files per column
    step=1,     # slide by 1 file → maximum temporal overlap
)
result = agg.fit(file_list, freq_target_khz=5000.0, vipir_version_idx=1)

# For 60 files, window=8, step=1 → 53 columns
print(result.n_snapshots)   # 53
result.plot()
```

### Separate load + combine

```python
# Load cubes (gate geometry auto-read from first file header)
cubes = agg.load(file_list, freq_target_khz=5000.0)
print(f"r₀ = {agg.gate_spacing_km:.3f} km  start = {agg.gate_start_km:.2f} km")

# Combine with chosen mode
result = agg.combine(cubes)
```

### Choosing the target frequency for Es

Es echoes appear **below foEs** (the Es critical frequency).  For summer daytime
Es at Wallops Island (foEs typically 2–8 MHz), use 3000–4000 kHz:

```python
# 3500 kHz is safely below typical summer foEs → Es is reflected, not transparent
agg = RiqAggregator(n_subbands=100, resolution_factor=10,
                    output_mode="moving_avg", window=8, step=1)
result = agg.fit(file_list, freq_target_khz=3500.0)
```

---

## `load()` details

`load()` iterates over `file_list`, opens each RIQ file via `RiqDataset`, finds the
pulset whose frequency is closest to `freq_target_khz`, assembles the full
`(n_pulse, n_gate, n_rx)` complex IQ cube, and returns the list.

- Gate geometry (`gate_spacing_km`, `gate_start_km`) is updated from the **first**
  successfully loaded file's SCT header (`timing.gate_step`, `timing.gate_start`).
- Files whose closest pulset differs by more than `freq_tol_khz` are skipped with a
  warning.
- Files that cannot be read (IO error, wrong format) are skipped with a warning.
- Raises `RuntimeError` if no valid cubes are loaded from any file.

---

## See Also

- [Package Overview](index.md)
- [EsCaponImager — capon.py](capon.md)
- [EsImagingResult](capon.md#esimagingresult)
- [Es Imaging Example](../../../../examples/vipir/es_imaging.md)

## References

Liu, T., Yang, G., & Jiang, C. (2023). *Space Weather*, 21, e2022SW003195.
[https://doi.org/10.1029/2022SW003195](https://doi.org/10.1029/2022SW003195)
