# RIQ Echo Extractor (pynasonde.vipir.riq.echo)

<span class="api-badge api-package">P</span>
`pynasonde.vipir.riq.echo` — Dynasonde-style seven-parameter echo extraction for VIPIR RIQ data.

## Overview

This module implements the echo characterisation algorithm described in
Zabotin et al. (2005) *"Dynasonde 21"* for VIPIR RIQ raw IQ data.  For
each sounding frequency it fits seven physical parameters to the complex
I+Q samples stored across the receiver array:

| Symbol | Parameter | Description |
|--------|-----------|-------------|
| φ₀ | `gross_phase_deg` | Gross (mean) phase of the echo signal |
| V* | `velocity_mps` | Doppler / phase-path velocity |
| R′ | `height_km` | Group range (virtual height) |
| XL | `xl_km` | Eastward echolocation displacement |
| YL | `yl_km` | Northward echolocation displacement |
| PP | `polarization_deg` | Polarization chirality angle |
| EP | `residual_deg` | Least-squares fit residual |
| A  | `amplitude_db` | Echo amplitude |

## Classes

::: pynasonde.vipir.riq.echo.Echo
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: pynasonde.vipir.riq.echo.EchoExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - __init__
            - extract
            - to_dataframe
            - to_xarray
            - echoes

## Quick start

```python
from pynasonde.vipir.riq import RiqDataset, EchoExtractor

dataset = RiqDataset.create_from_file("WI937_2024058061501.RIQ")

ext = EchoExtractor(
    sct=dataset.sct,
    pulsets=dataset.pulsets,
    snr_threshold_db=3.0,
    min_rx_for_direction=3,
    max_echoes_per_pulset=5,
).extract()

# Pandas DataFrame — one row per echo
df = ext.to_dataframe()

# CF-convention xarray Dataset
ds = ext.to_xarray()
```

## Algorithm notes

1. **I/Q cube** — for each `Pulset` (fixed frequency), raw I and Q
   samples are assembled into a `(pulse, gate, rx)` complex array.
2. **SNR gate selection** — gates whose mean amplitude exceeds
   `snr_threshold_db` above the noise floor are retained.
3. **Gross phase (φ₀)** — `angle(mean(I + jQ))` averaged across all
   pulses and receivers.
4. **Doppler / V\*** — phase is unwrapped along the pulse-time axis;
   a linear regression yields the Doppler frequency; `V* = f_d·c / (2·f₀)`.
5. **Direction (XL, YL)** — inter-antenna phase differences are fit to a
   planar-wavefront model via least squares; the recovered direction
   cosines are projected to km offsets using the virtual height.
6. **Polarization (PP)** — quasi-orthogonal antenna pairs are identified
   from `rx_direction` dot products; the circular-polarization ratio is
   converted to an angle.
7. **Residual (EP)** — the RMS residual of the planar-wavefront LS fit.

## References

Zabotin, N. A., Wright, J. W., & Zhbankov, G. A. (2006).
NeXtYZ: Three-dimensional electron density inversion for
dynasonde ionograms. *Radio Science*, 41(6).
<https://doi.org/10.1029/2005RS003352>
