# True Height Inversion (`pynasonde.vipir.analysis.inversion`)

<div class="hero">
  <h3>Virtual Height → True Height Abel / Lamination Inversion</h3>
  <p>
    Inverts an O-mode ionogram trace h′(f) to the true electron density profile
    N(h) using the Titheridge (1967) lamination method.
  </p>
</div>

## Theory

An ionogram records the **virtual height** h′(f) — the height a pulse would reach
at the speed of light.  Because the O-mode group refractive index μ′ > 1, the true
reflection height is always lower.  The Abel integral:

```
h′(f) = ∫₀^{h_r(f)} μ′(f, z) dz,   μ′(f, fp) = 1 / √(1 − fp²/f²)
```

is discretised by the **lamination method** into N horizontal layers:

```
r_n = h′(f_n) − Σ_{j=1}^{n-1} (μ′(f_n, f_j) − 1) × Δh_j
```

Electron density follows from N = fp² × 1.2399×10⁴ cm⁻³.

## Classes

::: pynasonde.vipir.analysis.inversion
    options:
      members:
        - TrueHeightInversion
        - EDPResult
      show_root_heading: true
      show_source: true

---

## `TrueHeightInversion`

### Constructor

```python
TrueHeightInversion(
    method: str = "lamination",         # only option currently
    min_freq_mhz: float = 1.0,          # low-frequency cutoff
    max_freq_mhz: float | None = None,  # optional upper cutoff
    monotone_enforce: bool = True,       # remove non-monotone h_true points
    freq_col: str = "frequency_mhz",    # column name for fit_from_df
    height_col: str = "height_km",
    mode_col: str = "mode",             # O-mode filter for fit_from_df
    bin_width_mhz: float = 0.05,        # binning step for fit_from_df stability
)
```

### Methods

| Method | Input | Output |
|--------|-------|--------|
| `fit(freq_mhz, h_virtual_km)` | Arrays of matching shape | `EDPResult` |
| `fit_from_df(df)` | Echo DataFrame (O-mode filtered inside) | `EDPResult` |

!!! note "fit_from_df expects a virtual height trace"
    `fit_from_df` extracts median virtual height per frequency bin — it requires
    echoes that form a valid O-mode trace h′(f) (monotone increasing in height).
    Do **not** pass a raw N(h) profile here; construct `EDPResult` directly instead.

### Quick start

```python
from pynasonde.vipir.analysis import TrueHeightInversion, PolarizationClassifier

inv = TrueHeightInversion(min_freq_mhz=1.5, bin_width_mhz=0.05)

# From arrays
edp = inv.fit(freq_mhz=trace_f, h_virtual_km=trace_h)
print(edp.summary())
# EDPResult (lamination): n_layers=24  foF2=8.20 MHz  hmF2=280.5 km  NmF2=8.31e+05 cm⁻³

# From an echo DataFrame (O-mode filter applied automatically)
clf = PolarizationClassifier(o_mode_sign=-1)
pol = clf.fit(echo_df)
edp = inv.fit_from_df(pol.annotated_df)   # O-mode column used for filtering
edp.plot()
```

---

## `EDPResult` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `true_height_km` | `ndarray` | True reflection heights (km) |
| `plasma_freq_mhz` | `ndarray` | Plasma frequency at each layer (MHz) = sounding frequency |
| `electron_density_cm3` | `ndarray` | Electron density at each layer (cm⁻³) |
| `virtual_height_km` | `ndarray` | Input virtual heights (km) |
| `frequency_mhz` | `ndarray` | Input sounding frequencies (MHz) |
| `foF2_mhz` | `float` | F2 critical frequency (MHz) |
| `hmF2_km` | `float` | F2 peak true height (km) |
| `NmF2_cm3` | `float` | Peak electron density (cm⁻³) |
| `method` | `str` | `"lamination"` |
| `n_layers` | `int` | Number of layers after monotone filter |

### Methods

```python
edp.summary()          # "EDPResult (lamination): n_layers=24  foF2=8.20 MHz …"
edp.to_dataframe()     # DataFrame: frequency_mhz, virtual_height_km, true_height_km, …
edp.to_csv("edp.csv")  # write to file
edp.plot()             # two-panel: N(h) left, virtual vs true height right
```

### Build EDPResult directly (for synthetic / model profiles)

When you have a known N(h) profile (e.g. Chapman layer, model output) rather than an
ionogram trace, bypass the inversion and construct EDPResult directly:

```python
import numpy as np
from pynasonde.vipir.analysis.inversion import EDPResult

_FP_TO_N = 1.2399e4          # N_cm3 = fp_mhz² × this
h = np.linspace(60.0, 130.0, 140)
xi = (h - 90.0) / 8.0
fp = np.maximum(0.5 * np.exp(0.5 * (1.0 - xi - np.exp(-xi))), 1e-4)
peak = int(np.argmax(fp))

edp = EDPResult(
    true_height_km=h, plasma_freq_mhz=fp,
    electron_density_cm3=fp**2 * _FP_TO_N,
    virtual_height_km=h, frequency_mhz=fp,
    foF2_mhz=float(fp[peak]), hmF2_km=float(h[peak]),
    NmF2_cm3=float(fp[peak]**2 * _FP_TO_N),
    method="synthetic_chapman", n_layers=len(h),
)
```

---

## References

- Titheridge, J. E. (1967). A new method for the analysis of ionospheric h'(f) records.
  *J. Atmos. Terr. Phys.*, 29, 763–778.
- Paul, A. K. (1975). POLAN — A program for true-height analysis of ionograms.
  *NOAA Technical Report ERL 324-SEL 31*.

## See Also

- [Analysis Overview](index.md)
- [Absorption Profile](absorption.md) — uses `EDPResult` for Method 4
- [NeXtYZ Inversion](nextyz.md) — full 3-D inversion alternative
