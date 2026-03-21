# Echo Extraction — Dynasonde-Style Seven-Parameter Analysis

<div class="hero">
  <h3>Dynasonde 21 Echo Parameters from VIPIR RIQ Data</h3>
  <p>
    Load a VIPIR <code>.RIQ</code> binary file, run the seven-parameter echo
    extractor, and produce a 2×3 grid of diagnostic plots — ionogram,
    XL/YL echolocation, Doppler velocity, and polarization — in a single
    pipeline.
  </p>
</div>

This page explains `examples/vipir/echo_extraction.py`.

**Physics background.**  The echo extraction follows the Dynasonde 21
framework (Zabotin et al., 2005).  For every frequency step each
range gate that exceeds the SNR threshold is characterised by eight numbers:

| Symbol | Field | Description |
|--------|-------|-------------|
| φ₀ | `gross_phase_deg` | Mean complex phase of the pulse set (°) |
| V* | `velocity_mps` | Doppler / phase-path velocity (m s⁻¹) |
| R′ | `height_km` | Virtual height / group range (km) |
| XL | `xl_km` | Eastward echolocation (km) |
| YL | `yl_km` | Northward echolocation (km) |
| PP | `polarization_deg` | Chirality / polarization rotation (°) |
| EP | `residual_deg` | Planar-wavefront LS residual (°) |
| A  | `amplitude_db` | Peak echo amplitude (dB) |

---

## Call Flow

```
RiqDataset.create_from_file()
    └─ sct  (SctType — sounder geometry, timing, frequency schedule)
    └─ pulsets  (list[Pulset] — one per frequency step)
           │
           ▼
EchoExtractor(sct, pulsets, snr_threshold_db=3.0)
    .extract()
           │
    ┌──────┴───────┐
    ▼              ▼
.to_dataframe()  .to_xarray()
```

---

## Step-by-Step

### 1 — Load the RIQ file

```python
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

riq = RiqDataset.create_from_file(
    "WI937_2022233235902.RIQ",
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[0],
)
```

The returned object carries:

* `riq.sct` — a fully populated `SctType` with receiver positions, timing, and frequency schedule.
* `riq.pulsets` — a list of `Pulset` objects; one per transmitted frequency step.

### 2 — Run EchoExtractor

```python
from pynasonde.vipir.riq.echo import EchoExtractor

extractor = EchoExtractor(
    sct=riq.sct,
    pulsets=riq.pulsets,
    snr_threshold_db=3.0,       # dB above noise floor required to accept a gate
    min_rx_for_direction=3,     # minimum receivers for the XL/YL/EP LS fit
    max_echoes_per_pulset=5,    # keep only the 5 strongest echoes per frequency
)
extractor.extract()
```

The `.extract()` call:

1. Assembles the complex I/Q cube `C[pulse, gate, rx]` for each pulset.
2. Computes the coherent mean phasor `C_mean[gate, rx]` over the pulse axis.
3. Estimates the noise floor (median amplitude) and selects high-SNR gates.
4. At each qualifying gate derives φ₀, V*, XL, YL, PP, EP, and A.

### 3 — Export results

```python
df = extractor.to_dataframe()   # pandas — one row per echo
ds = extractor.to_xarray()      # xarray — CF-convention Dataset
```

DataFrame column reference:

```
frequency_khz    height_km     amplitude_db   gross_phase_deg
doppler_hz       velocity_mps  xl_km          yl_km
polarization_deg residual_deg  snr_db         gate_index
pulse_ut         rx_count
```

### 4 — Diagnostic plots

The example script produces a **2 × 3** figure grid:

| Panel | Contents | X-axis | Y-axis | Colour |
|-------|----------|--------|--------|--------|
| **(A)** Ionogram | All echoes | Frequency (MHz) | Virtual height (km) | Amplitude (dB) |
| **(B)** XL vs Height | Eastward echoes | XL (km) | Virtual height (km) | Frequency (MHz) |
| **(C)** YL vs Height | Northward echoes | YL (km) | Virtual height (km) | Frequency (MHz) |
| **(D)** Echolocation map | XL vs YL | XL (km) | YL (km) | Amplitude (dB) |
| **(E)** Doppler velocity | V* profile | V* (m s⁻¹) | Virtual height (km) | Frequency (MHz) |
| **(F)** Polarization | PP profile | PP (°) | Virtual height (km) | Frequency (MHz) |

Panels B, C, D, F require at least `min_rx_for_direction` receivers.
When direction data are unavailable (all NaN), each panel displays a
descriptive message rather than an empty axes.

A NaN-fraction diagnostic is also printed to stdout for each direction
parameter so you can immediately tell whether the direction fit ran:

```
  xl_km               : 312/480 valid (65%)
  yl_km               : 312/480 valid (65%)
  polarization_deg    : 0/480 valid (0%)
  residual_deg        : 0/480 valid (0%)
```

```python
fig.savefig("docs/examples/figures/echo_extraction.png", dpi=150)
```

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `snr_threshold_db` | 3.0 | Lower → more (noisier) echoes; raise to 6–10 for clean ionograms |
| `min_rx_for_direction` | 3 | Set to 0 to always attempt the XL/YL fit regardless of receiver count |
| `max_echoes_per_pulset` | 5 | `None` keeps all echoes above threshold |

---

## xarray CF Output

```python
ds = extractor.to_xarray()
# ds.data_vars:
#   frequency_khz    units=kHz   long_name='Sounding frequency'
#   height_km        units=km    long_name='Virtual height R-prime'
#   velocity_mps     units=m/s   long_name='Phase-path velocity V-star'
#   xl_km            units=km    long_name='Eastward echolocation XL'
#   yl_km            units=km    long_name='Northward echolocation YL'
#   ...
```

The dataset can be saved to NetCDF with `ds.to_netcdf("echoes.nc")`.

---

## Algorithm Notes

### Gross phase (φ₀)

$$\phi_0 = \arg\!\left(\frac{1}{N_p N_r}\sum_{p,r} (I_{p,r} + j Q_{p,r})\right)$$

where $N_p$ is the pulse count and $N_r$ the receiver count.

### Doppler / phase-path velocity (V\*)

The coherent phase is unwrapped along the pulse-time axis and fitted by
linear regression to yield the phase rate $\dot\phi$:

$$f_d = \frac{\dot\phi}{2\pi}, \qquad V^* = \frac{f_d \cdot c}{2 f_0}$$

### Echolocation (XL, YL)

Inter-antenna phase differences are fitted to a planar-wavefront model:

$$\Delta\phi_{mn} = \frac{2\pi}{\lambda}\bigl[(x_m-x_n)\,l + (y_m-y_n)\,m\bigr]$$

The East/North direction cosines $(l, m)$ are obtained by least squares;
echolocations follow from $X_L = R'\,l$, $Y_L = R'\,m$.

---

## References

Zabotin, N. A., Wright, J. W., Bullett, T. W., & Zabotina, L. Ye. (2005).
Dynasonde 21 principles of data processing, transmission, storage and web
service. *Proc. Ionospheric Effects Symposium 2005*, p. 7B3-1.

Wright, J. W. & Pitteway, M. L. V. (1999). A new data acquisition concept
for digital ionosondes: Phase-based echo recognition and real-time parameter
estimation. *Radio Science*, 34, 871–882.
