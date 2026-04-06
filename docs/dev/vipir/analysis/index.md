# Analysis Sub-package (`pynasonde.vipir.analysis`)

<div class="hero">
  <h3>Physics Analysis Layer for Ionosonde Data</h3>
  <p>
    Instrument-agnostic algorithms that operate on filtered echo DataFrames,
    scaled O-mode traces, or raw IQ cubes — from HF absorption to 3-D electron
    density inversion and high-resolution sporadic-E layer imaging.
  </p>
</div>

## Module overview

| Module | Processor → Result | Summary |
|--------|--------------------|---------|
| [es_imaging/](es_imaging/index.md) | `EsCaponImager` → `EsImagingResult`<br>`RiqAggregator` → `EsImagingResult` | Capon cross-spectrum high-resolution Es imaging (Liu et al. 2023); multi-file A+B+C combining for VIPIR's 4-pulse-per-file constraint |
| [absorption](absorption.md) | `AbsorptionAnalyzer` → `LOFResult`, `DifferentialResult`, `TotalAbsorptionResult`, `AbsorptionProfileResult` | LOF index, differential O/X SNR, calibrated absorption L(f), height-resolved κ(z) from Appleton-Hartree |
| [inversion](inversion.md) | `TrueHeightInversion` → `EDPResult` | Titheridge (1967) lamination method; virtual → true height Abel inversion; outputs fp(h), N(h) |
| [polarization](polarization.md) | `PolarizationClassifier` → `PolarizationResult` | O/X mode separation via PP chirality; configurable sign convention per hemisphere |
| [spread_f](spread_f.md) | `SpreadFAnalyzer` → `SpreadFResult` | Range/frequency/mixed spread-F detection; height IQR and fsF2−foF2 metrics |
| [scaler](scaler.md) | `IonogramScaler` → `ScaledParameters` | URSI/CCIR parameter scaling: foE, foF1, foF2, h′F, MUF(3000), bootstrap σ |
| [irregularities](irregularities.md) | `IrregularityAnalyzer` → `IrregularityProfile` | EP structure function; power-law spectral index α; height-resolved profile; anisotropy proxy |
| [nextyz](nextyz.md) | `NeXtYZInverter` → `NeXtYZResult` | NeXtYZ 3-D WSI model + Hamiltonian ray tracing (Zabotin et al. 2006); full and lite variants |

## Typical analysis pipeline

```
EchoExtractor.to_dataframe()         (raw echoes)
        │
        ▼
PolarizationClassifier.fit()         → mode labels ("O" / "X" / …)
        │
        ├──► IonogramScaler.fit()         → foF2, h′F, MUF
        │
        ├──► TrueHeightInversion.fit()    → N(h) electron density profile
        │           │
        │           └──► AbsorptionAnalyzer.absorption_profile()  → κ(z), L(z)
        │
        ├──► AbsorptionAnalyzer.lof_absorption()          → LOF index (no calib)
        ├──► AbsorptionAnalyzer.differential_absorption()  → ΔL(f)
        ├──► AbsorptionAnalyzer.total_absorption()         → L(f) calibrated
        │
        ├──► SpreadFAnalyzer.fit()        → range/freq spread-F classification
        ├──► IrregularityAnalyzer.fit()   → EP spectral index profile
        └──► NeXtYZInverter.fit()         → 3-D N(h) with tilts
```

## All public imports

```python
from pynasonde.vipir.analysis import (
    # Es imaging
    EsCaponImager, RiqAggregator, EsImagingResult,
    # Absorption
    AbsorptionAnalyzer, LOFResult, DifferentialResult,
    TotalAbsorptionResult, AbsorptionProfileResult,
    # Inversion
    TrueHeightInversion, EDPResult,
    # Polarization
    PolarizationClassifier, PolarizationResult,
    # Spread-F
    SpreadFAnalyzer, SpreadFResult,
    # Scaler
    IonogramScaler, ScaledParameters,
    # Irregularities
    IrregularityAnalyzer, IrregularityProfile,
    # NeXtYZ
    NeXtYZInverter, NeXtYZResult, WedgePlane,
)
```

## Es imaging quick start

```python
from pynasonde.vipir.analysis import EsCaponImager, RiqAggregator

# Single RIQ file (many pulses available)
imager = EsCaponImager(n_subbands=100, resolution_factor=10, gate_spacing_km=3.84)
result = imager.fit(iq_cube)   # (pulses, gates[, rx])
print(result.summary())        # Z=100  K=10  Δr=0.384 km

# Multi-file A+B+C combining (VIPIR: 4 pulses × 8 Rx × 8 files)
agg = RiqAggregator(
    n_subbands=100, resolution_factor=10,
    gate_spacing_km=1.499,     # overridden from RIQ header in load()
    output_mode="single",      # "single" or "slow_rti"
)
result = agg.fit(file_list, freq_target_khz=5000.0, freq_tol_khz=50.0)
result.plot()
```

## Singularity constraint (Es imaging)

The Capon covariance matrix `R_f = G·G^H/(V−Z+1)` has shape `(Z, Z)` and rank at
most `V−Z+1`.  For full rank: **Z ≤ (V+1)/2**.

| Parameter | Constraint | Notes |
|-----------|-----------|-------|
| `Z` (subbands) | Hard: `Z < V`; Recommended: `Z ≤ (V+1)/2` | Warning issued above (V+1)/2; diagonal loading partially compensates |
| `K` (resolution factor) | **None** — K is a free parameter | Sets output grid `ω_l = 2π·l/(K·V)`; does NOT enter `R_f` |

## See Also

- [Es Imaging Package](es_imaging/index.md)
- [EsCaponImager (capon.py)](es_imaging/capon.md)
- [RiqAggregator (aggregator.py)](es_imaging/aggregator.md)
- [Es Imaging Example](../../../examples/vipir/es_imaging.md)
- [VIPIR Overview](../index.md)
