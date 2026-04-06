# HF Absorption (`pynasonde.vipir.analysis.absorption`)

<div class="hero">
  <h3>HF Radio Absorption Estimation from VIPIR Echo DataFrames</h3>
  <p>
    Four independent estimators — from the calibration-free LOF index through
    full height-resolved Appleton-Hartree absorption profiles.
  </p>
</div>

## Theory

An HF radio wave is attenuated whenever free electrons collide with neutral molecules
while oscillating.  Under the **no-field, weak-collision** Appleton-Hartree limit
the local absorption rate is:

```
κ(z) = 4.343 · ν(z) · fp²(z)/f² / (c · √(1 − fp²(z)/f²))   [dB/km]
```

where `fp(z)` is the plasma frequency, `f` is the wave frequency, and `ν(z)` is
the electron-neutral collision frequency.  The total one-way absorption is:

```
L(f) = ∫₀^{h_r(f)} κ(z) dz   [dB]
```

## Classes

::: pynasonde.vipir.analysis.absorption
    options:
      members:
        - AbsorptionAnalyzer
        - LOFResult
        - DifferentialResult
        - TotalAbsorptionResult
        - AbsorptionProfileResult
      show_root_heading: true
      show_source: true

---

## `AbsorptionAnalyzer`

### Constructor

```python
AbsorptionAnalyzer(
    freq_col: str = "frequency_khz",   # auto-detects kHz vs MHz from median
    height_col: str = "height_km",
    snr_col: str = "snr_db",
    mode_col: str = "mode",            # "O" / "X" labels from PolarizationClassifier
    freq_bin_mhz: float = 0.1,         # grouping bin for Methods 2 & 3
    f_ref_mhz: float = 1.0,            # quiet-day reference for LOF index
)
```

### Methods

| Method | Input | Output | Calibration needed |
|--------|-------|--------|--------------------|
| `lof_absorption(df)` | Any echo DataFrame | `LOFResult` | None |
| `differential_absorption(df)` | Mode-labelled DataFrame | `DifferentialResult` | None |
| `total_absorption(df, tx_eirp_dbw, …)` | Echo DataFrame + hardware params | `TotalAbsorptionResult` | EIRP (dBW) + Rx gain (dBi) |
| `absorption_profile(edp, nu_hz, …)` | `EDPResult` + collision profile | `AbsorptionProfileResult` | N(h) + ν(z) model |

### Quick start

```python
from pynasonde.vipir.analysis import AbsorptionAnalyzer, TrueHeightInversion, PolarizationClassifier

az = AbsorptionAnalyzer(freq_bin_mhz=0.1, f_ref_mhz=1.0)

# Method 1 — LOF index (no calibration)
lof = az.lof_absorption(echo_df)
print(lof.summary())
# LOFResult: fmin=2.450 MHz  A=5.008 MHz²  (f_ref=1.00 MHz)

# Method 2 — differential O/X absorption
clf = PolarizationClassifier(o_mode_sign=-1)
pol = clf.fit(echo_df)
diff = az.differential_absorption(pol.annotated_df)
print(diff.summary())
diff.plot()

# Method 3 — calibrated total absorption
total = az.total_absorption(echo_df, tx_eirp_dbw=30.0, rx_gain_dbi=2.0)
total.plot()

# Method 4 — height-resolved profile (needs EDP + collision model)
inv = TrueHeightInversion()
edp = inv.fit_from_df(pol.o_mode_df())
nu_func = lambda h_km: 3e7 * np.exp(-(h_km - 80) / 8)  # simple exponential
profile = az.absorption_profile(edp, nu_hz=nu_func, f_wave_mhz=3.0)
print(profile.summary())
profile.plot()
```

---

## Result dataclasses

### `LOFResult`

| Field | Type | Description |
|-------|------|-------------|
| `fmin_mhz` | `float` | Lowest observed frequency (MHz) |
| `lof_index_mhz2` | `float` | A = fmin² − f_ref² (MHz²) |
| `f_ref_mhz` | `float` | Reference frequency used |
| `n_echoes` | `int` | Total echoes in input DataFrame |

### `DifferentialResult`

| Field | Type | Description |
|-------|------|-------------|
| `profile_df` | `DataFrame` | Columns: `frequency_mhz`, `snr_o_db`, `snr_x_db`, `delta_snr_db`, `n_o`, `n_x` |
| `mean_delta_db` | `float` | Mean ΔL = SNR_O − SNR_X (dB) across all bins |
| `n_echoes_o` | `int` | O-mode echo count |
| `n_echoes_x` | `int` | X-mode echo count |

### `TotalAbsorptionResult`

| Field | Type | Description |
|-------|------|-------------|
| `profile_df` | `DataFrame` | Columns: `frequency_mhz`, `virtual_height_km`, `fsl_db`, `absorption_db` |
| `tx_eirp_dbw` | `float` | EIRP used |
| `rx_gain_dbi` | `float` | Rx gain used |
| `reflection_coeff_db` | `float` | Reflection coefficient used |

### `AbsorptionProfileResult`

| Field | Type | Description |
|-------|------|-------------|
| `profile_df` | `DataFrame` | Columns: `height_km`, `nu_hz`, `fp_mhz`, `X`, `kappa_dB_per_km` |
| `cumulative_df` | `DataFrame` | Columns: `height_km`, `L_oneway_db` |
| `total_absorption_db` | `float` | Total one-way absorption to top of EDP (dB) |

---

## References

- Davies, K. (1990). *Ionospheric Radio*. Peter Peregrinus, London.
- McNamara, L. F. (1991). *The Ionosphere*. Krieger, Florida.
- Budden, K. G. (1985). *The Propagation of Radio Waves*. Cambridge University Press.

## See Also

- [Analysis Overview](index.md)
- [Polarization Classifier](polarization.md) — provides mode labels for Method 2
- [True Height Inversion](inversion.md) — provides EDPResult for Method 4
