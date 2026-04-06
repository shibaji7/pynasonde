# Polarization Classifier (`pynasonde.vipir.analysis.polarization`)

<div class="hero">
  <h3>O/X Wave-Mode Separation via PP Polarization Parameter</h3>
  <p>
    Labels each echo "O", "X", "ambiguous", or "unknown" by thresholding the
    PP chirality parameter from the Dynasonde seven-parameter signal model.
  </p>
</div>

## Theory

The PP parameter (`polarization_deg`) measures the chirality of the reflected
wavefront from differential phase between quasi-orthogonal antenna pairs.  For
vertically incident HF signals the O and X modes have opposite chirality, mapping
to opposite signs of PP.

The sign convention is station-specific:

- **Northern hemisphere**: negative PP → O-mode (`o_mode_sign=-1`, default)
- **Southern hemisphere**: positive PP → O-mode (`o_mode_sign=+1`)

## Classes

::: pynasonde.vipir.analysis.polarization
    options:
      members:
        - PolarizationClassifier
        - PolarizationResult
      show_root_heading: true
      show_source: true

---

## `PolarizationClassifier`

### Constructor

```python
PolarizationClassifier(
    o_mode_sign: int = -1,                     # -1 (NH default) or +1 (SH)
    pp_ambiguous_threshold_deg: float = 20.0,  # |PP| below this → "ambiguous"
    pp_col: str = "polarization_deg",
)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `fit(df) → PolarizationResult` | Classify all echoes in df |
| `infer_o_mode_sign` | `infer_o_mode_sign(station_lat) → int` | Static helper: −1 for lat≥0, +1 for lat<0 |

### Labels assigned

| Label | Condition |
|-------|-----------|
| `"O"` | `|PP| ≥ threshold` and PP has the O-mode sign |
| `"X"` | `|PP| ≥ threshold` and PP has the X-mode sign |
| `"ambiguous"` | `|PP| < threshold` (near-linear polarization) |
| `"unknown"` | PP is NaN (fewer than `min_rx_for_direction` antennas used) |

### Quick start

```python
from pynasonde.vipir.analysis import PolarizationClassifier

# Infer sign from station latitude
sign = PolarizationClassifier.infer_o_mode_sign(station_lat=37.9)  # → -1 (NH)

clf = PolarizationClassifier(o_mode_sign=sign, pp_ambiguous_threshold_deg=20.0)
result = clf.fit(echo_df)

print(result.summary())
# PolarizationResult: total=1842  O=743  X=698  ambiguous=201  unknown=200
#   o_mode_sign=negative PP

# Extract mode subsets
o_df  = result.o_mode_df()   # O-mode echoes only
x_df  = result.x_mode_df()   # X-mode echoes only
all_df = result.to_dataframe()  # full DataFrame with "mode" column added

# Plot PP histogram with O/X regions shaded
result.plot()
```

---

## `PolarizationResult` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `annotated_df` | `DataFrame` | Input df + new `"mode"` column (`"O"` / `"X"` / `"ambiguous"` / `"unknown"`) |
| `o_mode_count` | `int` | O-mode echo count |
| `x_mode_count` | `int` | X-mode echo count |
| `ambiguous_count` | `int` | Ambiguous echo count |
| `unknown_count` | `int` | NaN-PP echo count |
| `o_mode_sign` | `int` | Sign convention used (`±1`) |
| `pp_ambiguous_threshold_deg` | `float` | Threshold applied |

### Methods

```python
result.summary()        # one-line summary string
result.to_dataframe()   # annotated_df copy
result.o_mode_df()      # O-mode subset
result.x_mode_df()      # X-mode subset
result.plot()           # PP histogram with coloured O/X/ambiguous bands
```

---

## References

- Zabotin, N. A., Wright, J. W., & Zhbankov, G. A. (2006). NeXtYZ. *Radio Science*, 41, RS6S32.
- Wright, J. W. & Pitteway, M. L. V. (1994). *J. Atmos. Terr. Phys.*, 56, 577–585.

## See Also

- [Analysis Overview](index.md)
- [Absorption](absorption.md) — uses mode labels for differential O/X absorption
- [True Height Inversion](inversion.md) — uses O-mode trace from `o_mode_df()`
- [Ionogram Scaler](scaler.md) — uses mode-labelled echoes for foF2 scaling
