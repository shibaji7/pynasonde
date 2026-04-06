<!--
Author(s): Shibaji Chakraborty

Disclaimer:

-->

# VIPIR: _Vertical Incidence Pulsed Ionospheric Radar_

<div class="hero">
  <h2>RIQ File Format, SCT/PCT Structures, and Ionogram Analysis</h2>
  <p>
    VIPIR is a pulsed HF radar system for vertical-incidence ionospheric sounding.
    Pynasonde reads its binary RIQ files and exposes the Sounding Control Table,
    Pulse Configuration Table, and raw IQ samples as Python data structures.
  </p>
</div>

VIPIR was developed by Scion Associates under a SBIR grant from AFRL.  The first
installation was at NASA Wallops Island Flight Facility in 2008; 15 instruments have
since been deployed worldwide.  Version 2 (2015) introduced improved dynamic range
and multi-channel digital down-conversion.  Pynasonde supports both generations.

Reference: Grubb et al. (2011), *Radio Science*.

## RIQ File Structure

VIPIR output is stored in **Raw In-phase and Quadrature (RIQ)** files.  Each file
consists of a fixed header block followed by one record per transmitted pulse:

```
┌─────────────────────────────┐
│ Sounding Control Table    │  (SCT) — instrument configuration
│ (variable size)           │
├─────────────────────────────┤
│ Pulse Configuration Table │  (PCT) — per-pulse metadata + IQ data
│ repeated N_pulses times   │
└─────────────────────────────┘
```

### Python usage

```python
from pynasonde.vipir.riq.parsers.read_riq import RiqReader

reader = RiqReader("station_20230101.riq")
reader.read()
sct = reader.sct      # SoundingControlTable dataclass
pcts = reader.pcts    # list of PulseConfigTable records
```

---

## Sounding Control Table (SCT)

The SCT describes the complete instrument configuration for a sounding.
Pynasonde maps it to a Python dataclass (format version 1.20).

!!! note "Format compatibility"
    `C` uses null-filled strings; `FORTRAN` uses space-filled strings.
    Both are supported.  64-bit C code must use `packed` struct alignment.

### Top-level SCT fields

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `magic` | `int32` | 4 | `0x51495200` (`\0RIQ`) — byte-order check |
| `sounding_table_size` | `int32` | 4 | Bytes in SCT structure |
| `pulse_table_size` | `int32` | 4 | Bytes in PCT structure |
| `raw_data_size` | `int32` | 4 | Bytes in raw data block (one PRI) |
| `struct_version` | `float64` | 8 | Format version (currently 1.20) |
| `start_year` | `int32` | 4 | Ionogram start year |
| `start_daynumber` | `int32` | 4 | Day of year |
| `start_month` | `int32` | 4 | Month |
| `start_day` | `int32` | 4 | Day of month |
| `start_hour` | `int32` | 4 | Hour (UT) |
| `start_minute` | `int32` | 4 | Minute |
| `start_second` | `int32` | 4 | Second |
| `start_epoch` | `int32` | 4 | UNIX epoch of measurement start |
| `readme` | `str` | 16 | Operator comment |
| `decimation_method` | `int32` | 4 | 0 = raw (no decimation) |
| `decimation_threshold` | `float64` | 8 | Threshold for decimation method |
| `station` | `StationType` | variable | Station geometry substructure |
| `timing` | `TimingType` | variable | Radar timing substructure |
| `frequency` | `FrequencyType` | variable | Frequency sweep substructure |
| `receiver` | `ReceiverType` | variable | DDC receiver settings |
| `exciter` | `ExciterType` | variable | Transmitter settings |
| `monitor` | `MonitorType` | variable | Built-in test values |

### Station substructure (`StationType`)

| Field | Type | Description |
|-------|------|-------------|
| `rx_name` | `str` | Receive station name |
| `rx_latitude` | `float64` | Rx latitude (deg N) |
| `rx_longitude` | `float64` | Rx longitude (deg E) |
| `rx_altitude` | `float64` | Rx altitude (m MSL) |
| `rx_count` | `int32` | Number of receive antennas |
| `rx_position` | `[float64]` | (East, North, Up) position of each Rx (m) |
| `rx_direction` | `[float64]` | (East, North, Up) direction of each Rx |
| `tx_latitude` | `float64` | Tx latitude (deg N) |
| `tx_longitude` | `float64` | Tx longitude (deg E) |
| `drive_band_bounds` | `[float64]` | Drive band start/stop (kHz) |
| `ref_type` | `str` | Reference oscillator type |
| `clock_type` | `str` | UT timing source |

### Timing substructure (`TimingType`)

| Field | Type | Description |
|-------|------|-------------|
| `pri` | `float64` | Pulse Repetition Interval (µs) |
| `pri_count` | `int32` | Number of PRIs per ionogram |
| `gate_count` | `int32` | Number of range gates |
| `gate_start` / `gate_end` | `float64` | Range gate placement (µs) |
| `gate_step` | `float64` | Range gate delta (µs) |
| `data_baud_count` | `int32` | Baud count in transmitted pulse |
| `data_baud` | `[float64]` | Waveform baud pattern (up to 1024) |

### Frequency substructure (`FrequencyType`)

| Field | Type | Description |
|-------|------|-------------|
| `base_start` / `base_end` | `float64` | Frequency sweep range |
| `base_steps` | `int32` | Number of sweep frequencies |
| `tune_type` | `int32` | 1=log, 2=linear, 3=table, 4=log+shuffle |
| `base_table` | `[float64]` | Up to 8192 nominal frequencies |
| `linear_step` | `float64` | Linear step (kHz) |
| `log_step` | `float64` | Log step (percent) |

---

## Pulse Configuration Table (PCT)

One PCT record exists per transmitted pulse.  It captures pulse metadata
and the raw IQ samples for all range gates and receiver channels.

### PCT fields

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `int32` | Pulse sequence number |
| `pri_ut` | `float64` | UT of this pulse |
| `frequency` | `float64` | Sounding frequency (kHz) |
| `pa_forward_power` | `float64` | Amplifier forward power |
| `pa_reflected_power` | `float64` | Reflected power |
| `pa_vswr` | `float64` | Voltage Standing Wave Ratio |
| `proc_noise_level` | `float64` | Estimated noise floor for this PRI |
| `pulse_i` | `2D array` | In-phase samples (gates × channels) |
| `pulse_q` | `2D array` | Quadrature samples (gates × channels) |

`pulse_i` and `pulse_q` have shape `(gate_count, rx_count)`.  Combine them
as `signal = pulse_i + 1j * pulse_q` to get the complex baseband voltage.

### Reconstruct ionogram power

```python
import numpy as np

signal = np.array(pct.pulse_i) + 1j * np.array(pct.pulse_q)
power_db = 20 * np.log10(np.abs(signal) + 1e-12)   # shape: (gates, channels)
```

---

## Ionogram Analysis Workflow

```
RiqReader.read()
    │
    ├─ sct  →  station geometry, timing, frequency sweep
    │
    └─ pcts →  per-pulse IQ  →  coherent integration
                                    │
                                    ├─ ionogram power(f, h')
                                    ├─ Doppler analysis
                                    └─ Ne(h) inversion
```

### Example: load and inspect

```python
from pynasonde.vipir.riq.parsers.read_riq import RiqReader

reader = RiqReader("station_20230101.riq")
reader.read()

print(f"Station: {reader.sct.station.rx_name}")
print(f"Frequencies: {reader.sct.frequency.base_steps}")
print(f"Gate count:  {reader.sct.timing.gate_count}")
print(f"Pulses read: {len(reader.pcts)}")
```

---

## Ionogram Analysis

The `pynasonde.vipir.analysis` sub-package provides physics algorithms that
operate on filtered echo DataFrames or raw IQ cubes:

| Class | Purpose |
|-------|---------|
| `EsCaponImager` | High-resolution Es layer imaging (single RIQ file) |
| `RiqAggregator` | Multi-file Es imager: per-file RTI or moving-average RTI |
| `AbsorptionAnalyzer` | LOF index, differential O/X SNR, absorption profile |
| `TrueHeightInversion` | Virtual height → true height; outputs N(h) |
| `PolarizationClassifier` | O/X mode separation via PP chirality |
| `SpreadFAnalyzer` | Spread-F detection (range/frequency/mixed) |
| `IonogramScaler` | Automatic foF2, foE, h′F, MUF scaling |

### High-resolution Es layer imaging

```python
from pynasonde.vipir.analysis import EsCaponImager

imager = EsCaponImager(
    n_subbands=100,         # Z — Capon subbands
    resolution_factor=10,   # K — 10× finer range grid
    gate_spacing_km=1.499,  # VIPIR native gate width r₀
    gate_start_km=90.0,
)
result = imager.fit(iq_cube)   # iq_cube: (pulses, gates[, rx])
print(result.summary())
# EsImagingResult: Z=100  K=10  r₀=1.499 km → Δr=0.150 km
result.plot()
```

Use `RiqAggregator` to combine multiple files into a high-SNR RTI.  Every
**(pulse, Rx) pair** is stacked as an independent snapshot, so the Capon
covariance is averaged over L profiles before a single matrix inversion —
dramatically improving sensitivity to weak Es echoes.

```python
from pynasonde.vipir.analysis import RiqAggregator

# Per-file RTI — L = n_pulse × n_rx = 32 snapshots per column
agg = RiqAggregator(n_subbands=100, resolution_factor=10,
                    output_mode="per_file")
result = agg.fit(file_list, freq_target_khz=3500.0, vipir_version_idx=1)
result.plot()

# Moving-average RTI — L = window × n_pulse × n_rx = 256 snapshots per column
agg = RiqAggregator(n_subbands=100, resolution_factor=10,
                    output_mode="moving_avg", window=8, step=1)
result = agg.fit(file_list, freq_target_khz=3500.0, vipir_version_idx=1)
result.plot()
```

See the [Es Imaging Example](../examples/vipir/es_imaging.md) and
[Analysis API](../dev/vipir/analysis/index.md) for full details.

---

## See Also

- [Read / Plot RIQ Example](../examples/vipir/proc_riq.md)
- [Es Imaging Example](../examples/vipir/es_imaging.md)
- [Analysis API Reference](../dev/vipir/analysis/index.md)
- [VIPIR RIQ API Reference](../dev/vipir/riq/parsers/read_riq.md)
