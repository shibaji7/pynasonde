<!--
Author(s): Shibaji Chakraborty

Disclaimer:

-->
# HF Ray Tracing

## Why HF traces matter

High-frequency (HF) radio waves launched into the ionosphere do not travel in
straight lines. The electron-density gradients of the D, E, and F layers
continuously refract the wave path, eventually bending it back toward Earth — a
process called **ionospheric reflection**. The virtual height recorded on an
ionogram is the apparent (straight-line) travel time converted to distance; the
*true* reflection height, the actual ray path, and the group range all differ
because the wave slows and curves as it propagates.

Understanding where a ray actually travels is critical for several reasons:

* **Ionogram inversion**: Converting a virtual-height trace to a true electron
  density profile (the "true-height" problem) requires knowing the full ray
  geometry, not just the echo delay.
* **HF communication path planning**: Predicting skip distance, maximum usable
  frequency (MUF), and multi-hop propagation demands accurate ray trajectories
  through a realistic ionospheric model.
* **Space weather impact assessment**: Sudden ionospheric disturbances (SIDs),
  travelling ionospheric disturbances (TIDs), and eclipse-driven depletions all
  modify ray paths in ways that cannot be seen from the raw trace alone.
* **Validation of autoscalers**: Comparing synthetic ionograms (forward-modelled
  from a density profile via ray tracing) against measured traces provides an
  objective quality metric for tools like
  [AutoScale-ISCA](isca.md).

## From ionogram to ray path

Pynasonde extracts ionospheric parameters — critical frequencies, virtual
heights, electron-density profiles — from raw instrument files. Once those
parameters are in hand, the natural next step is to **propagate rays** through
the reconstructed ionosphere to:

1. Verify that the inverted profile reproduces the observed trace.
2. Estimate ground-range footprints for oblique-incidence paths.
3. Explore sensitivity of ray paths to uncertainty in the scaled parameters.

This forward-modelling step goes beyond what Pynasonde does internally. For that
we recommend **PyTrace**.

---

## PyTrace — HF ray-tracing companion

!!! info "External tool"
    PyTrace is developed and maintained independently of Pynasonde.
    Full documentation, installation instructions, and examples are at:

    **[https://pytrace.readthedocs.io/en/latest/](https://pytrace.readthedocs.io/en/latest/)**

PyTrace is a Python library for numerical HF ray tracing through analytic and
empirical ionospheric models. It solves the Haselgrove ray-tracing equations in
3-D, supports both ordinary and extraordinary propagation modes, and can ingest
electron-density profiles derived from sources such as Pynasonde's SAO/NGI
extractors.

### Typical workflow linking Pynasonde → PyTrace

```python
# 1. Extract an electron-density profile with Pynasonde
from pynasonde.digisonde.parsers.sao import SaoExtractor

extractor = SaoExtractor("path/to/file.SAO")
extractor.extract()
hp = extractor.get_height_profile()          # DataFrame: th, ed, pf, …

# 2. Hand the profile to PyTrace (see PyTrace docs for exact API)
#    https://pytrace.readthedocs.io/en/latest/
import pytrace  # installed separately: pip install pytrace

model = pytrace.ProfileModel.from_dataframe(
    hp, height_col="th", density_col="ed"
)
rays = pytrace.trace(
    model,
    frequency_mhz=5.0,
    elevation_deg=75.0,
    azimuth_deg=0.0,
)
rays.plot()
```

### When to reach for PyTrace

| Task | Tool |
|------|------|
| Parse `.SAO`, `.RSF`, `.RIQ`, `.NGI` files | **Pynasonde** |
| Extract critical frequencies, virtual heights, EDP | **Pynasonde** |
| Forward ray-trace through a density profile | **[PyTrace](https://pytrace.readthedocs.io/en/latest/)** |
| Compute MUF, skip distance, multi-hop paths | **[PyTrace](https://pytrace.readthedocs.io/en/latest/)** |
| Validate scaled profiles against measured traces | both, in combination |

---

For questions about PyTrace itself — installation, ray-tracing equations,
supported ionospheric models — please refer to the
[PyTrace documentation](https://pytrace.readthedocs.io/en/latest/) and its
upstream issue tracker.
