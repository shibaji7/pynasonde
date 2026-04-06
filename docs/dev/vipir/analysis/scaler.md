# Ionogram Scaler (`pynasonde.vipir.analysis.scaler`)

<div class="hero">
  <h3>Automatic Ionogram Parameter Scaling</h3>
  <p>
    Derives standard URSI/CCIR ionospheric parameters from a filtered,
    O-mode-labelled echo DataFrame with bootstrap uncertainty estimates.
  </p>
</div>

## Scaled parameters

| Parameter | Description |
|-----------|-------------|
| `foE_mhz` | E-layer critical frequency (MHz) |
| `h_prime_E_km` | E-layer minimum virtual height (km) |
| `foF1_mhz` | F1-layer critical frequency (MHz); NaN when absent |
| `h_prime_F1_km` | F1-layer minimum virtual height (km); NaN when absent |
| `foF2_mhz` | F2-layer critical frequency (MHz) |
| `h_prime_F2_km` | F2-layer minimum virtual height (km) |
| `MUF3000_mhz` | Maximum usable frequency for a 3 000 km path (MHz) |
| `M3000F2` | Transmission factor MUF(3000)/foF2 |
| `foF2_sigma_mhz` | Bootstrap σ(foF2) (MHz) |
| `h_prime_F2_sigma_km` | Bootstrap σ(h′F2) (km) |

Layer height windows used internally:

| Layer | Height range (km) | Frequency range (MHz) |
|-------|------------------|-----------------------|
| E | 90 – 160 | 1.0 – 4.5 |
| F1 | 160 – 250 | — |
| F2 | 160 – 800 | ≥ 2.0 |

## Classes

::: pynasonde.vipir.analysis.scaler
    options:
      members:
        - IonogramScaler
        - ScaledParameters
      show_root_heading: true
      show_source: true

---

## `IonogramScaler`

### Quick start

```python
from pynasonde.vipir.analysis import IonogramScaler, PolarizationClassifier

clf = PolarizationClassifier(o_mode_sign=-1)
pol = clf.fit(echo_df)

scaler = IonogramScaler()
params = scaler.fit(pol.annotated_df)

print(params.summary())
# ScaledParameters: foF2=8.20 MHz  h'F2=245 km  foE=3.10 MHz  MUF(3000)=14.5 MHz

params.plot()
```

---

## `ScaledParameters` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `foE_mhz` | `float` | E-layer critical frequency |
| `h_prime_E_km` | `float` | E-layer min virtual height |
| `foF1_mhz` | `float` | F1 critical frequency (NaN if absent) |
| `h_prime_F1_km` | `float` | F1 min virtual height (NaN if absent) |
| `foF2_mhz` | `float` | F2 critical frequency |
| `h_prime_F2_km` | `float` | F2 min virtual height |
| `MUF3000_mhz` | `float` | MUF for 3 000 km path |
| `M3000F2` | `float` | MUF(3000)/foF2 |
| `foF2_sigma_mhz` | `float` | Bootstrap σ(foF2) |
| `h_prime_F2_sigma_km` | `float` | Bootstrap σ(h′F2) |
| `quality_flags` | `dict` | `"E_detected"`, `"F1_detected"`, `"F2_detected"`, `"foF2_reliable"` |

### Methods

```python
params.summary()     # one-line summary string
params.plot()        # ionogram with layer annotations and scaled values
```

---

## References

- Reinisch, B. W., & Huang, X. (1983). *Radio Science*, 18(3), 477–492.
- Piggott, W. R., & Rawer, K. (1972). *URSI Handbook of Ionogram Interpretation
  and Reduction* (2nd ed.). World Data Center A for Solar-Terrestrial Physics.

## See Also

- [Analysis Overview](index.md)
- [Polarization Classifier](polarization.md) — provides mode-labelled input
- [True Height Inversion](inversion.md) — converts foF2 trace to N(h)
