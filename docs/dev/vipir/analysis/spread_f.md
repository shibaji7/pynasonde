# Spread-F Analyzer (`pynasonde.vipir.analysis.spread_f`)

<div class="hero">
  <h3>Spread-F Detection and Characterisation</h3>
  <p>
    Classifies ionograms as range spread-F, frequency spread-F, mixed, or none,
    based on height IQR per frequency step and echo persistence beyond foF2.
  </p>
</div>

## Theory

Spread-F appears as diffuse scattering from F-layer irregularities:

- **Range spread-F** — echoes at a given frequency spread over a wide height range
  (height IQR > threshold).  Caused by large-scale bottomside irregularities.
- **Frequency spread-F** — echoes persist above the vertical-incidence foF2
  (`fsF2 − foF2 > 0`).  Caused by field-aligned irregularities that scatter
  signals obliquely.
- **Mixed spread-F** — both criteria are simultaneously met.

## Classes

::: pynasonde.vipir.analysis.spread_f
    options:
      members:
        - SpreadFAnalyzer
        - SpreadFResult
      show_root_heading: true
      show_source: true

---

## `SpreadFAnalyzer`

### Quick start

```python
from pynasonde.vipir.analysis import SpreadFAnalyzer, PolarizationClassifier

clf = PolarizationClassifier(o_mode_sign=-1)
pol = clf.fit(echo_df)

sf = SpreadFAnalyzer()
result = sf.fit(pol.annotated_df)

print(result.summary())
# SpreadFResult: classification='range'  foF2=8.20 MHz
#   freq_spread=0.00 MHz  height_IQR=145.3 km
result.plot()
```

---

## `SpreadFResult` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `classification` | `str` | `"none"`, `"range"`, `"frequency"`, or `"mixed"` |
| `freq_spread_mhz` | `float` | fsF2 − foF2 (MHz); NaN when foF2 unavailable |
| `height_iqr_km` | `float` | Median height IQR across F-layer frequency steps (km) |
| `spread_onset_freq_mhz` | `float` | Frequency where height spread first exceeds threshold; NaN when absent |
| `fo_f2_mhz` | `float` | Estimated foF2 (MHz); NaN when insufficient O-mode echoes |
| `ep_by_height` | `DataFrame` | Columns: `height_bin_km`, `ep_mean_deg`, `ep_std_deg`, `n_echoes` |
| `range_spread_flags` | `DataFrame` | Columns: `frequency_mhz`, `height_iqr_km`, `is_spread` |

### Methods

```python
result.summary()        # one-line summary string
result.to_dataframe()   # returns ep_by_height DataFrame
result.plot()           # height IQR vs frequency + EP mean vs height bin
```

---

## References

- Aarons, J. (1993). Longitudinal morphology of equatorial F-layer irregularities.
  *Space Science Reviews*, 63, 209–243.
- Hysell, D. L. (2000). Overview of plasma irregularities in equatorial spread F.
  *J. Atmos. Solar-Terr. Phys.*, 62, 1037–1056.

## See Also

- [Analysis Overview](index.md)
- [Polarization Classifier](polarization.md) — provides mode-labelled input
- [Irregularities](irregularities.md) — EP spectral index analysis
