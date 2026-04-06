# `capon.py` — EsCaponImager & EsImagingResult

<div class="hero">
  <h3>Single-Cube Capon Imager</h3>
  <p>
    Operates on one <code>(pulse_count, gate_count [, rx_count])</code> IQ cube —
    a single RIQ file or a pre-built NumPy array.  Best when many pulses per
    frequency are available (e.g. WISS: 256 pulses, or coherently-integrated VIPIR).
  </p>
</div>

## Source

`pynasonde/vipir/analysis/es_imaging/capon.py`

## API reference

::: pynasonde.vipir.analysis.es_imaging.capon
    options:
      members:
        - EsCaponImager
        - EsImagingResult
      show_root_heading: true
      show_source: true

---

## `EsCaponImager`

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_subbands` | `int` | `100` | Z — Capon subbands.  Recommended: Z ≤ (V+1)/2 |
| `resolution_factor` | `int` | `10` | K — output grid multiplier (K·V high-res bins).  No singularity constraint |
| `coherent_integrations` | `int` | `1` | Pulses averaged before imaging (1 = per-pulse; N = N-pulse coherent stack) |
| `rx_index` | `int` | `0` | Rx channel to extract when input is 3-D `(pulses, gates, rx)` |
| `diagonal_loading` | `float` | `1e-3` | ε — R_f regularisation: R_f ← R_f + ε·tr(R_f)/Z·I |
| `gate_start_km` | `float` | `90.0` | Height of first range gate (km) |
| `gate_spacing_km` | `float` | `3.84` | Native gate spacing r₀ (km).  WISS = 3.84 km, VIPIR ≈ 1.499 km |

### `fit(iq_cube)` → `EsImagingResult`

```python
from pynasonde.vipir.analysis import EsCaponImager

imager = EsCaponImager(
    n_subbands=100,           # Z
    resolution_factor=10,     # K
    coherent_integrations=1,  # per-pulse imaging
    rx_index=0,               # single Rx channel from 3-D cube
    diagonal_loading=1e-3,
    gate_start_km=90.0,
    gate_spacing_km=3.84,     # WISS r₀
)

# 2-D input: (pulse_count, gate_count)
result = imager.fit(iq_cube_2d)

# 3-D input: (pulse_count, gate_count, rx_count) — rx_index=0 extracted
result = imager.fit(iq_cube_3d)

print(result.summary())
# EsImagingResult: snapshots=256  Z=100  K=10  r₀=3.84 km → Δr=0.384 km
#   height=90.0–857.7 km
```

### Internal methods (documented for custom pipelines)

| Method | Signature | Description |
|--------|-----------|-------------|
| `_validate` | `_validate(V)` | Check Z < V; warn when Z > (V+1)/2 |
| `_covariance` | `_covariance(G_ss) → R_f_inv` | Build Hankel G, compute R_f, apply diagonal loading, invert |
| `_steering_matrix` | `_steering_matrix(V) → A` | Construct A (K·V, Z) with ω_l = 2π·l/(K·V) |
| `_capon` | `_capon(R_f_inv, A) → P` | P[l] = 1/(a^H R_f⁻¹ a), shape (K·V,) |
| `_process_pulse` | `_process_pulse(range_profile, A) → P` | FFT → covariance → Capon for one range profile |

---

## `EsImagingResult`

Shared result dataclass returned by both `EsCaponImager.fit()` and `RiqAggregator.combine()`.

### Fields

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `pseudospectrum_db` | `ndarray` | `(n_snapshots, K·V)` | Normalised Capon pseudospectrum (dB, max = 0 dB) |
| `heights_km` | `ndarray` | `(K·V,)` | High-resolution height axis (km), spacing = r₀/K |
| `gate_heights_km` | `ndarray` | `(V,)` | Original gate heights (km), spacing = r₀ |
| `n_subbands` | `int` | — | Z used |
| `resolution_factor` | `int` | — | K used |
| `coherent_integrations` | `int` | — | Pulses combined per snapshot |
| `gate_spacing_km` | `float` | — | Native gate spacing r₀ (km) |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `effective_resolution_km` | `float` | r₀ / K |
| `n_snapshots` | `int` | Number of rows in `pseudospectrum_db` |

### Methods

```python
result.summary()
# "EsImagingResult: snapshots=1  Z=100  K=10  r₀=1.499 km → Δr=0.150 km
#   height=90.0–1537.9 km"

result.to_dataframe(snapshot=0)
# DataFrame with columns: height_km, power_db

result.plot()             # RTI intensity map when n_snapshots > 1
result.plot(snapshot=0)   # single 1-D profile
result.plot(vmin=-60)     # custom dB floor  (default −60 dB)
result.plot(cmap="plasma") # custom colormap  (default "jet")
```

### Plot behaviour

| `n_snapshots` | `snapshot` arg | Output |
|---------------|---------------|--------|
| 1 | any / None | 1-D profile (power dB vs height km) |
| > 1 | None | 2-D RTI (pulse index × height, pcolormesh) |
| > 1 | int | 1-D profile for that snapshot only |

---

## WISS vs VIPIR quick reference

| Setting | WISS (Liu et al.) | VIPIR WI937 |
|---------|-------------------|-------------|
| `gate_spacing_km` | 3.84 | 1.499 |
| `gate_start_km` | ~0 | ~90 |
| Typical V | 200 | 960 |
| Recommended Z | 100 | 480 |
| K | 10 | 10 |
| Δr | 384 m | 150 m |

---

## See Also

- [Package Overview](index.md)
- [RiqAggregator — aggregator.py](aggregator.md)
- [Es Imaging Example](../../../../examples/vipir/es_imaging.md)

## References

Liu, T., Yang, G., & Jiang, C. (2023). *Space Weather*, 21, e2022SW003195.
[https://doi.org/10.1029/2022SW003195](https://doi.org/10.1029/2022SW003195)
