# Es Imaging Package (`pynasonde.vipir.analysis.es_imaging`)

<div class="hero">
  <h3>High-Resolution Sporadic-E Layer Range Imaging</h3>
  <p>
    Capon minimum-variance cross-spectrum analysis (Liu et al. 2023) applied to
    VIPIR pulse-compressed RIQ gate data.  Achieves up to 10× finer range
    resolution — e.g. 150 m from a 1.499 km VIPIR native gate — without
    sacrificing temporal resolution.
  </p>
</div>

## Package contents

| Module | Classes | Description |
|--------|---------|-------------|
| [capon.py](capon.md) | `EsCaponImager`, `EsImagingResult` | Single-cube Capon imager — one RIQ file or pre-built IQ array |
| [aggregator.py](aggregator.md) | `RiqAggregator` | Multi-file A+B+C imager — loads RIQ files, coherent Rx beamform + incoherent averaging |

## Public imports

All three classes are re-exported from the package root and from
`pynasonde.vipir.analysis`:

```python
# Any of these work:
from pynasonde.vipir.analysis.es_imaging import EsCaponImager, RiqAggregator, EsImagingResult
from pynasonde.vipir.analysis import EsCaponImager, RiqAggregator, EsImagingResult
```

## Algorithm overview

### Capon cross-spectrum range imaging (Liu et al. 2023)

```
IQ cube (pulses × gates [× rx])
        │
        ▼
FFT along gate axis → cross-power spectrum G_ss (V points)
        │
        ▼
Hankel subband matrix G  shape (Z, V−Z+1)
        │
        ▼
Covariance  R_f = G · G^H / (V−Z+1)  +  ε·tr(R_f)/Z · I
        │
        ▼
Steering matrix A  shape (K·V, Z),  ω_l = 2π·l / (K·V)
        │
        ▼
Capon pseudospectrum  P[l] = 1 / (a^H · R_f⁻¹ · a)   shape (K·V,)
```

Effective range resolution = r₀ / K.

### Multi-file A+B+C combining (RiqAggregator)

```
n_files × n_pulse × n_rx IQ cubes
        │
        ├─ Option A: coherent Rx beamform within each pulse
        │       R_beam[p] = cube[p,:,:] @ w*    (+9 dB SNR for 8 Rx)
        │
        ├─ Option B: Capon per pulse → incoherent mean per file
        │       P_file = mean_p{ P_capon(R_beam[p,:]) }   (÷ n_pulse variance)
        │
        └─ Option C: incoherent mean/stack across files
                single:   P = mean_f{ P_file }            (÷ n_files variance)
                slow_rti: P = stack_f{ P_file }            (n_files time columns)
```

## Singularity constraint

| Parameter | Constraint | Effect |
|-----------|-----------|--------|
| `Z` (subbands) | Hard: `Z < V`; recommended: `Z ≤ (V+1)/2` | Above `(V+1)/2`, R_f is rank-deficient; warning issued |
| `K` (resolution factor) | **None** — K is a free parameter | Only sets output grid density; does **not** enter R_f |

## Instrument parameters

| Instrument | `gate_spacing_km` (r₀) | Typical V | Recommended Z | K | Δr |
|------------|------------------------|-----------|---------------|---|-----|
| WISS (Liu et al. 2023) | 3.84 km | 200 | 100 | 10 | 384 m |
| VIPIR WI937 | 1.499 km | 960 | 480 | 10 | 150 m |

## See Also

- [EsCaponImager — capon.py](capon.md)
- [RiqAggregator — aggregator.py](aggregator.md)
- [Analysis Sub-package Overview](../index.md)
- [Es Imaging Example](../../../../examples/vipir/es_imaging.md)

## References

Liu, T., Yang, G., & Jiang, C. (2023). High-resolution sporadic E layer observation
based on ionosonde using a cross-spectrum analysis imaging technique. *Space Weather*,
21, e2022SW003195. [https://doi.org/10.1029/2022SW003195](https://doi.org/10.1029/2022SW003195)
