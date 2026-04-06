# NeXtYZ Inversion (`pynasonde.vipir.analysis.nextyz`)

<div class="hero">
  <h3>NeXtYZ 3-D Electron Density Inversion for Dynasonde Ionograms</h3>
  <p>
    Physics-based Wedge-Stratified Ionosphere (WSI) model with Hamiltonian
    ray tracing.  Recovers 3-D tilted N(h) profiles from direction-finding
    echo DataFrames.  Implements Zabotin et al. (2006).
  </p>
</div>

## Theory

The **Wedge-Stratified Ionosphere (WSI)** model represents the local electron
density as a stack of plasma-frequency wedges.  Each wedge boundary is a
**frame plane** (h, nₓ, nᵧ) encoding ionospheric tilt.

A **Hamiltonian ray-tracer** (eikonal ODE with full Appleton-Lassen refractive
index) propagates sounding signals through the model.  Wedge parameters are
solved bottom-up in a least-squares loop minimising:

1. Group-range residual `ΔR′ᵢ₊₁`
2. Ground-return distance of the mean-direction ray (tilt constraint)

### Two variants

| Variant | Solved per wedge | Notes |
|---------|-----------------|-------|
| **NeXtYZ Lite** (default) | h only; tilts from mean angles of arrival | ~6× faster |
| **NeXtYZ Full** | h, nₓ, nᵧ (alternating optimisation) | Full 3-D tilt |

### Coordinate system

```
x = geographic East  (km)
y = geographic North (km)
z = vertical Up      (km)
```

ODE independent variable τ (km): `dr/dτ = group-slowness direction`.

### Required DataFrame columns

| Column | Type | Description |
|--------|------|-------------|
| `xl_km` | float | Dynasonde echolocation East coordinate (km) |
| `yl_km` | float | Dynasonde echolocation North coordinate (km) |
| `height_km` | float | Observed group range R′ (km) |
| `frequency_khz` | float | Sounding frequency (kHz) |
| `mode` | str | `"O"` or `"X"` (optional, O-mode preferred) |
| `amplitude_db` | float | Echo amplitude (optional, for weighting) |

## Classes

::: pynasonde.vipir.analysis.nextyz
    options:
      members:
        - NeXtYZInverter
        - NeXtYZResult
        - WedgePlane
      show_root_heading: true
      show_source: true

---

## `NeXtYZInverter`

### Quick start

```python
from pynasonde.vipir.analysis import NeXtYZInverter, PolarizationClassifier

# 1. Label modes
clf = PolarizationClassifier(o_mode_sign=-1)
pol = clf.fit(echo_df)

# 2. Run NeXtYZ Lite inversion
inv = NeXtYZInverter(
    mode="lite",         # "lite" (fast) or "full" (3-D tilts)
    fp_step_mhz=0.05,   # plasma frequency step between wedge boundaries
    B_gauss=0.5,        # geomagnetic field strength (Gauss) for Appleton-Lassen
    dip_deg=60.0,       # magnetic dip angle (degrees)
)
result = inv.fit(pol.annotated_df)

print(result.summary())
result.plot()
```

---

## `WedgePlane` dataclass

Parameters of one WSI wedge boundary after inversion.

| Field | Type | Description |
|-------|------|-------------|
| `height_km` | `float` | Frame plane height (km) |
| `fp_mhz` | `float` | Plasma frequency at this boundary (MHz) |
| `nx` | `float` | East tilt component of frame normal |
| `ny` | `float` | North tilt component of frame normal |
| `nz` | `float` | Vertical component (derived: √(1−nₓ²−nᵧ²)) |
| `group_range_residual` | `float` | R′ residual after optimisation (km) |

---

## `NeXtYZResult` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `wedge_planes` | `list[WedgePlane]` | Solved WSI wedge boundaries |
| `profile_df` | `DataFrame` | Columns: `height_km`, `fp_mhz`, `electron_density_cm3`, `tilt_east_deg`, `tilt_north_deg` |
| `foF2_mhz` | `float` | F2 critical frequency (MHz) |
| `hmF2_km` | `float` | F2 peak height (km) |
| `n_wedges` | `int` | Number of WSI wedge boundaries solved |
| `mean_residual_km` | `float` | Mean group-range residual across all wedges (km) |

### Methods

```python
result.summary()      # one-line summary string
result.to_dataframe() # returns profile_df
result.plot()         # N(h) profile + tilt angle vs height
```

---

## Physical assumptions

!!! note "Collisions neglected"
    Collisions are neglected (valid for E and F regions above ~90 km, per
    Zabotin et al. 2006 §4).

!!! note "First-wedge underlying ionisation"
    The underlying ionisation below the lowest echo is approximated by a
    linear ramp from 0 to fp_start.  Full Titheridge underlying-ionisation
    integration is future work.

---

## References

Zabotin, N. A., Wright, J. W., & Zhbankov, G. A. (2006). NeXtYZ: Three-dimensional
electron density inversion for Dynasonde and ARTIST ionosondes. *Radio Science*, 41,
RS6S32. [https://doi.org/10.1029/2005RS003352](https://doi.org/10.1029/2005RS003352)

## See Also

- [Analysis Overview](index.md)
- [True Height Inversion](inversion.md) — 1-D lamination alternative (faster, no tilt)
- [Polarization Classifier](polarization.md) — provides mode-labelled input
