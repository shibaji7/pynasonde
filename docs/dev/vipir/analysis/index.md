# Analysis Sub-package (`pynasonde.vipir.analysis`)

<div class="hero">
  <h3>Physics Analysis Layer for Ionosonde Data</h3>
  <p>
    Instrument-agnostic algorithms that operate on filtered echo DataFrames,
    scaled O-mode traces, or raw IQ cubes — from HF absorption to 3-D electron
    density inversion and high-resolution sporadic-E layer imaging.
  </p>
</div>

## pynasonde 2.0 scope

Two modules are part of the stable 2.0 analysis API:

| Module | Processor → Result | Summary |
|--------|--------------------|---------|
| [polarization](polarization.md) | `PolarizationClassifier` → `PolarizationResult` | O/X mode separation via PP chirality; configurable sign convention per hemisphere |
| [scaler](scaler.md) | `IonogramScaler` → `ScaledParameters` | URSI/CCIR parameter scaling: foE, foF1, foF2, h′F, MUF(3000), bootstrap σ |

Both need only the core install (`pip install pynasonde`).

!!! note "Other analysis modules"
    `absorption`, `inversion`, `spread_f`, `irregularities`, `nextyz` and
    `es_imaging` are unchanged and their 1.x imports still work, but they are
    not part of 2.0 yet and are not covered by 2.0 docs. See the
    [1.3 documentation](https://pynasonde.readthedocs.io/en/v1.3.0/) for them.

## Typical pipeline

```
EchoExtractor.to_dataframe()         (raw echoes)
        │
        ▼
PolarizationClassifier.fit()         → mode labels ("O" / "X" / …)
        │
        └──► IonogramScaler.fit()         → foF2, h′F, MUF
```

## Public imports

```python
from pynasonde.vipir.analysis import (
    PolarizationClassifier, PolarizationResult,
    IonogramScaler, ScaledParameters,
)
```

## See Also

- [Polarization example](../../../examples/vipir/polarization.md)
- [Scaler example](../../../examples/vipir/scaler.md)
- [VIPIR Overview](../index.md)
