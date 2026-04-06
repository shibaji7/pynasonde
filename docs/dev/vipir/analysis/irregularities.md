# Irregularity Analyzer (`pynasonde.vipir.analysis.irregularities`)

<div class="hero">
  <h3>Small-Scale Ionospheric Irregularity Analysis via EP Structure Function</h3>
  <p>
    Computes the power-law spectral index of sub-wavelength irregularities from
    the EP (residual phase) parameter, binned by virtual height.
  </p>
</div>

## Theory

The EP parameter carries information about sub-wavelength irregularities.  The
second-order **structure function** of EP vs frequency lag Δf is:

```
D_EP(Δf) = ⟨ [EP(f + Δf) − EP(f)]² ⟩
```

For a power-law irregularity spectrum with spectral index α:

```
D_EP(Δf) ∝ Δf^α
```

A log–log fit yields α (spectral index) and A₀ (amplitude coefficient).
The outer scale L_outer is estimated as the lag where D_EP saturates (≥ 85% of max).

**Anisotropy proxy**: σ_EP(O-mode) / σ_EP(X-mode) — ratio close to unity → isotropic;
deviations → field-aligned anisotropy.

## Classes

::: pynasonde.vipir.analysis.irregularities
    options:
      members:
        - IrregularityAnalyzer
        - IrregularityProfile
      show_root_heading: true
      show_source: true

---

## `IrregularityAnalyzer`

### Quick start

```python
from pynasonde.vipir.analysis import IrregularityAnalyzer, PolarizationClassifier

clf = PolarizationClassifier(o_mode_sign=-1)
pol = clf.fit(echo_df)

ia = IrregularityAnalyzer()
profile = ia.fit(pol.annotated_df)

print(profile.summary())
# IrregularityProfile: n_echoes=1842  spectral_index=1.23
#   outer_scale=0.18 MHz  anisotropy=1.04
profile.plot()
```

---

## `IrregularityProfile` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `structure_function` | `DataFrame` | Columns: `delta_f_mhz`, `D_EP_deg2`, `n_pairs` |
| `spectral_index` | `float` | α from log-log fit (NaN if fit failed) |
| `amplitude_coeff` | `float` | A₀ in deg² (NaN if fit failed) |
| `outer_scale_mhz` | `float` | L_outer (MHz) where D_EP saturates (NaN if not observed) |
| `anisotropy_ratio` | `float` | σ_EP(O) / σ_EP(X); NaN when X-mode EP unavailable |
| `height_profile` | `DataFrame` | Columns: `height_bin_km`, `spectral_index`, `amplitude_coeff`, `outer_scale_mhz`, `n_echoes` |
| `n_echoes_total` | `int` | Total echoes used |

### Methods

```python
profile.summary()      # one-line summary string
profile.to_dataframe() # returns height_profile DataFrame
profile.plot()         # structure function + height-resolved spectral index
```

---

## References

- Hysell, D. L., & Burcham, J. D. (1998). JULIA radar studies of equatorial spread F.
  *J. Geophys. Res.*, 103(A12), 29155–29167.
- Kintner, P. M., & Seyler, C. E. (1985). High-latitude ionospheric turbulence.
  *Space Science Reviews*, 41, 91–129.

## See Also

- [Analysis Overview](index.md)
- [Polarization Classifier](polarization.md) — provides mode-labelled input with EP
- [Spread-F Analyzer](spread_f.md) — related large-scale irregularity analysis
