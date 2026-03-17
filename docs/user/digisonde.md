<!--
Author(s): Shibaji Chakraborty

Disclaimer:

-->

# DIGISONDE: _Digital Ionospheric Goniometric IonoSONDE_

<div class="hero">
  <h2>DPS4D Data Formats, Parsers, and Plotters</h2>
  <p>
    The Digisonde DPS4D is a digital ionospheric radar that captures amplitude,
    phase, direction of arrival, virtual height, Doppler frequency, and polarization
    of HF echoes.  Pynasonde parses every DPS4D output format into tidy
    pandas DataFrames and provides publication-ready summary plotters.
  </p>
</div>

!!! note "Data Access and Station Networks"
    Real-time and archived Digisonde data are publicly available through two
    community portals:

    - **GIRO (Global Ionosphere Radio Observatory)** — [giro.uml.edu](https://giro.uml.edu/)
      The primary international network of Digisonde stations providing
      ionospheric data products (SAO, RSF, DVL, DFT) in near real-time.
    - **UMass Lowell CEDAR Madrigal (ULCAR)** — [ulcar.uml.edu](https://ulcar.uml.edu/)
      Hosts SAO format specifications, SAOXML 5 schema, DTD files, and
      historical Digisonde archives.  Home of the ARTIST autoscaling software.

The Digisonde stands for **Digital Ionospheric Goniometric IonoSONDE**.  Rooted
in ionosonde technology pioneered by Sir Edward Appleton in the 1920s,
the DPS4D uses a cross-shaped four-antenna receive array and six interferometric
baselines to resolve echo arrival directions to 60° sectors.

**Reference**: Reinisch, B. W., et al. (2009). *Advances in ionospheric monitoring with
new Digisonde technology*, in *New Trends in Space Science*, Adv. Space Res.

## Ionospheric Echo Parameters

| Parameter | Description |
|-----------|-------------|
| Amplitude (A) | Received signal strength |
| Phase (φ) | Complex phase at each antenna |
| Direction of arrival | Zenith angle + azimuth from interferometry |
| Virtual height (h′) | Two-way propagation delay × c/2 |
| Doppler frequency (f_D) | Radial velocity of reflecting layer |
| Polarization | Ordinary (O) and extraordinary (X) magnetoionic modes |

From these raw echoes the DPS4D computes:

- **Electron Density Profiles** — near real-time Ne(h) with error bars via ARTIST autoscaler.
- **Classical Ionospheric Parameters** — foF2, foF1, foE, foEs, MUF(3000)F2, hmF2, hmF1, hmE, IRI B0/B1.
- **Vertical TEC** — integrated electron content from scaled Ne profile.
- **Plasma Drift Velocities** — E- and F-region vector drift from Doppler analysis (DVL files).
- **Skymaps** — ionospheric reflection-point distributions for vertical and oblique soundings (SKY files).

## Output File Formats

| Extension | Format | Content |
|-----------|--------|---------|
| `.SAO` | Structured ASCII / SAOXML 5 | Scaled ionosonde parameters and electron-density profiles |
| `.SKY` | Structured ASCII | Skymap echo-trace data after interferometric inversion |
| `.DVL` | Structured ASCII | Drift velocity estimates (Vx, Vy, Vz, Az) |
| `.DFT` | Structured Binary | Full Doppler Fourier spectra (amplitude + phase per height gate) |
| `.RSF` | Structured Binary (`data_format=4`) | Raw Sounding File — full amplitude/phase/direction per frequency |
| `.SBF` | Structured Binary (`data_format=5`) | Single Byte Format — compact ionogram without directional data |

---

## Standard Archiving Output (SAO / SAOXML 5)

SAO files store scaled ionosonde parameters for each sounding.  The legacy
fixed-width ASCII format coexists with the modern **SAOXML 5** variant which
uses XML encoding for unified exchange between data producers and users.

!!! note "Specifications"
    - [SAO-4 Format](https://ulcar.uml.edu/~iag/SAO-4.htm) — legacy text format reference at ULCAR.
    - [SAOXML 5 Specification](https://ulcar.uml.edu/SAOXML/SAO.XML%205.0%20specification%20v1.0.pdf) — XML schema and field definitions.
    - Data available at [GIRO](https://giro.uml.edu/) for 100+ stations worldwide.

### SAORecord — top-level XML fields

| Field | Type | Description |
|-------|------|-------------|
| `FormatVersion` | `str` | SAO format version (default `"5.0"`) |
| `StartTimeUTC` | `str` | Sounding start time in UTC |
| `URSICode` | `str` | URSI station code (e.g. `"KR835"`) |
| `StationName` | `str` | Station name (e.g. `"KIRTLAND"`) |
| `GeoLatitude` | `str` | Geographic latitude of the station |
| `GeoLongitude` | `str` | Geographic longitude of the station |
| `Source` | `str` | Data source descriptor (default `"Ionosonde"`) |
| `SourceType` | `str` | Instrument type identifier |
| `ScalerType` | `str` | Autoscaler identifier (e.g. `"ARTIST"`) |
| `SystemInfo` | `SystemInfo` | UML station ID and IUWDS code sub-object |
| `CharacteristicList` | `CharacteristicList` | URSI, Modeled, and Custom parameter entries |
| `TraceList` | `TraceList` | Ionogram trace elements (frequency, range, amplitude) |
| `ProfileList` | `ProfileList` | Electron density profile data |

### URSI Characteristics (CharacteristicList.URSI)

Each `URSI` entry holds one ionospheric parameter from the standard URSI set
(foF2, foF1, foE, foEs, hmF2, MUF3000, B0, B1, etc.):

| Field | Type | Description |
|-------|------|-------------|
| `ID` | any | URSI parameter identifier |
| `Val` | `float` | Parameter value (coerced to float) |
| `Name` | `str` | Human-readable parameter name |
| `Units` | `str` | Physical units (MHz, km, etc.) |
| `QL` | `str` | Quality level code |
| `DL` | `str` | Detection level code |
| `SigFig` | `str` | Significant figures |
| `UpperBound` / `LowerBound` | `str` | Confidence bound metadata |
| `Flag` | `str` | Optional quality or status flag |

### Trace (TraceList.Trace)

Ionogram traces store the raw amplitude-vs-frequency-vs-height echo data:

| Field | Type | Description |
|-------|------|-------------|
| `Type` | `str` | Trace type (default `"standard"`) |
| `Layer` | `str` | Layer identifier (E, F1, F2, Es, …) |
| `Polarization` | `str` | O or X mode |
| `FrequencyList` | `List[float]` | Frequency axis values (MHz) |
| `RangeList` | `List[float]` | Virtual height axis values (km) |
| `TraceValueList` | `List[TraceValueList]` | Named value arrays (amplitude, phase, etc.) |

### Profile (ProfileList.Profile)

| Field | Type | Description |
|-------|------|-------------|
| `Algorithm` | `str` | Profile inversion algorithm (e.g. `"NHPC"`) |
| `AlgorithmVersion` | `str` | Version string |
| `Type` | `str` | Profile type (default `"vertical"`) |
| `Tabulated.AltitudeList` | `List[float]` | Altitude axis (km) |
| `Tabulated.ProfileValueList` | `List[ProfileValueList]` | Named parameter arrays (Ne, fp, …) |

### Python usage

```python
from pynasonde.digisonde.parsers.sao import SaoExtractor

# Load all SAO files in a directory with 8 parallel workers
df = SaoExtractor.load_SAO_files(
    folders=["path/to/SAO/"],
    ext="KR835_*.SAO",
    func_name="height_profile",   # or "scaled"
    n_procs=8,
)
print(df[["datetime", "th", "pf", "ed"]].head())
```

### DataFrame columns (`func_name="height_profile"`)

| Column | Description |
|--------|-------------|
| `datetime` | UTC sounding time |
| `th` | Virtual height of the echo (km) |
| `pf` | Plasma frequency at that height (MHz) |
| `ed` | Electron density (cm⁻³) |

### Isodensity contour plot

Mimics the [Digisonde-Isodensity.gif](https://digisonde.com/images/Digisonde-Isodensity.gif) visualization —
time on x-axis, virtual height on y-axis, colored by plasma frequency, contour lines at each MHz level.

```python
import pandas as pd
from pynasonde.digisonde.digi_plots import SaoSummaryPlots

df["datetime"] = pd.to_datetime(df["datetime"])
p = SaoSummaryPlots(figsize=(10, 4), font_size=10)
p.add_isodensity_contours(
    df, xparam="datetime", yparam="th", zparam="pf",
    ylim=[50, 500], fbins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    text="KR835  2023-10-14",
)
p.save("sao_isodensity.png")
p.close()
```

---

## Skymap Data File (SKY)

Each SKY file holds one complete sounding with echo traces after interferometric
inversion, giving reflection-point locations in sky coordinates.

| Field | Type | Description |
|-------|------|-------------|
| `zenith_angle` | `float` | Zenith angle of echo arrival (deg) |
| `sampl_freq` | `float` | Sampling frequency (Hz) |
| `group_range` | `float` | Group path / virtual range (km) |
| `gain_ampl` | `float` | Receiver gain amplitude |
| `height_spctrum_ampl` | `array[float]` | Amplitude spectrum along height bins |
| `max_height_spctrum_ampl` | `float` | Maximum amplitude in height spectrum |
| `n_sources` | `int` | Number of detected echo sources |
| `polarization` | `str` | Echo polarization (O or X) |
| `x_coord` / `y_coord` | `float` | Skymap antenna-array projection coordinates |
| `spect_amp` | `float` | Spectral line amplitude |
| `spect_dop` | `int` | Raw Doppler index |
| `spect_dop_freq` | `float` | Doppler shift frequency (Hz) |
| `rms_error` | `float` | RMS error of the direction estimate |
| `datetime` | `datetime` | UTC sounding time |
| `local_datetime` | `datetime` | Station local time |

---

## Differential Velocity (DVL)

DVL files contain per-sounding vector plasma drift estimates derived from
Doppler analysis of echoes across multiple sky directions.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Always `"DVL"` |
| `version` | `str` | Data-processing version |
| `station_id` | `int32` | Numeric station ID |
| `ursi_tag` | `str` | URSI station code |
| `lat` / `lon` | `float64` | Station latitude / longitude (deg) |
| `date` | `dt.date` | UTC date of record |
| `doy` | `int32` | Day of year |
| `time` | `dt.time` | UTC time of record |
| `Vx` | `float64` | Velocity along magnetic north (m/s) |
| `Vx_err` | `float64` | Error in Vx (m/s) |
| `Vy` | `float64` | Velocity along magnetic east (m/s) |
| `Vy_err` | `float64` | Error in Vy (m/s) |
| `Az` | `float64` | Azimuth of drift, CW from magnetic north (deg) |
| `Az_err` | `float64` | Error in Az |
| `Vh` | `float64` | Velocity along geographic height (m/s) |
| `Vh_err` | `float64` | Error in Vh |
| `Vz` | `float64` | Vertical velocity (m/s) |
| `Vz_err` | `float64` | Error in Vz |
| `Cord` | `str` | Coordinate system: `COM` (Compass), `GEO` (Geographic), `CGm` (Corrected Geomagnetic) |
| `Hb` / `Ht` | `float64` | Bottom / top virtual height of measurement (km) |
| `Fl` / `Fu` | `float64` | Lower / upper sounding frequency (MHz) |

---

## DFT — Doppler Fourier Spectra

The `.DFT` file stores full Fourier spectra of Doppler-shifted signals per height
gate, enabling detailed analysis of plasma drift velocity distributions.

!!! note "Manual Reference"
    See [Digisonde-4D Technical Manual](https://digisonde.com/pdf/Digisonde4DManual_LDI-web.pdf)
    §5.166 for the complete format specification.

### File layout

| Entity | Size | Description |
|--------|------|-------------|
| Block | 4 096 bytes | Fundamental storage unit (fixed) |
| Sub-cases per block | 16 | One per height gate |
| Amplitude set | 128 bytes | 8-bit amplitude values per sub-case |
| Phase set | 128 bytes | 8-bit phase values per sub-case |

Header information is embedded in the **least-significant bits (LSBs) of amplitude bytes**
across all 16 sub-cases of a block.  The parser reconstructs a 256-bit header
bitstring and decodes all fields.

### DftHeader fields

| Field | Type | Description |
|-------|------|-------------|
| `record_type` | `hex` | Numeric record type identifier (nibble 0–3) |
| `year` | `int` | Year of measurement (BCD, two nibbles) |
| `doy` | `int` | Day of year (three nibbles, ×100 + ×10 + ×1) |
| `hour` | `int` | Hour (two nibbles) |
| `minute` | `int` | Minute (two nibbles) |
| `second` | `int` | Second (two nibbles) |
| `schdule` | `int` | Schedule code (nibble) |
| `program` | `int` | Program code (nibble) |
| `drift_data_flag` | `hex` | Drift data presence flag (byte) |
| `journal` | `hex` | Journal flags: bit0=new gain, bit1=new height, bit2=new freq, bit3=new case |
| `first_height_sampling_winodw` | `int` | First height sampling window index |
| `height_resolution` | `int` | Height bin resolution (×5 km per step) |
| `number_of_heights` | `int` | Number of height bins in the block |
| `start_frequency` | `int` | Start frequency (raw: ×1e5 + ×1e4 + … + ×1 → Hz) |
| `disk_io` | `hex` | Disk I/O flag |
| `freq_search_enabled` | `bool` | Frequency search enabled flag |
| `fine_frequency_step` | `int` | Fine frequency step (Hz) |
| `number_small_steps_scan_abs` | `int` | Absolute count of small frequency steps |
| `number_small_steps_scan` | `int` | Signed count of small steps (−16 … +15) |
| `start_frequency_case` | `int` | Case-specific start frequency index |
| `coarse_frequency_step` | `int` | Coarse frequency step |
| `end_frequency` | `int` | End frequency of scan |
| `bottom_height` | `int` | Bottom height index (×height_resolution × 5 km + 80 km) |
| `top_height` | `int` | Top height index |
| `stn_id` | `int` | Station ID (three BCD nibbles) |
| `phase_code` | `int` | Phase modulation code: 1=complementary, 2=short, 3=75% duty, 4=100% duty, +8=no switch |
| `multi_antenna_sequence` | `int` | Multi-antenna sequencing flag |
| `cit_length` | `int` | Coherent Integration Time length (ms) |
| `num_doppler_lines` | `int` | Number of Doppler spectral lines |
| `pulse_repeat_rate` | `int` | Pulse repetition rate (pps) |
| `waveform_type` | `hex` | Waveform type identifier |
| `delay` | `int` | Inter-pulse or processing delay |
| `frequency_search_offset` | `int` | Frequency search offset |
| `auto_gain` | `int` | AGC setting |
| `heights_to_output` | `int` | Number of heights written to the file |
| `num_of_polarizations` | `int` | Number of polarization channels |
| `start_gain` | `int` | Receiver start gain (dB) |

### SubCaseHeader fields

One per frequency/height sub-case embedded after the block header:

| Field | Type | Description |
|-------|------|-------------|
| `frequency` | `int` | Sub-case frequency (×1e4 + … + ×1 → 10 Hz units) |
| `height_mpa` | `int` | Height at most-probable amplitude (×1e3 + … + ×1 raw) |
| `height_bin` | `int` | Height bin index |
| `agc_offset` | `int` | AGC offset for this sub-case |
| `polarization` | `int` | Polarization identifier |

### DopplerSpectra fields

| Field | Type | Description |
|-------|------|-------------|
| `amplitude` | `np.array` | 128-element amplitude array (×3/8 → dB-like units) |
| `phase` | `np.array` | 128-element raw phase array (0–255) |

### DataFrame columns from `DftExtractor.to_pandas()`

| Column | Description |
|--------|-------------|
| `block_idx` | Block index (0-based) |
| `subcase_idx` | Height sub-case within block (0–15) |
| `height_km` | Estimated virtual height: 80 + subcase × height_resolution × 5 (km) |
| `doppler_bin` | Signed Doppler bin (−64 … +63, centred at bin 64) |
| `amplitude` | Signal amplitude (dB-like) |
| `phase` | Raw phase byte (0–255) |
| `frequency_hz` | Start frequency from block header (Hz) |
| `date` | Measurement timestamp |

### Python usage

```python
from pynasonde.digisonde.parsers.dft import DftExtractor

dft = DftExtractor("KR835_2023287000915.DFT", extract_time_from_name=True, extract_stn_from_name=True)
dft.extract()
df = dft.to_pandas()
```

### Doppler waterfall and spectra

```python
from pynasonde.digisonde.digi_plots import SkySummaryPlots

# Waterfall: Doppler bin × height, amplitude color
sk = SkySummaryPlots(figsize=(7, 5), font_size=10, subplot_kw={})
sk.plot_doppler_waterfall(df, cmap="inferno", text="KR835  2023-10-14 00:09 UT")
sk.save("dft_waterfall.png")
sk.close()

# Spectra: amplitude vs Doppler bin, one line per height
sk2 = SkySummaryPlots(figsize=(7, 4), font_size=10, subplot_kw={})
sk2.plot_doppler_spectra(df, n_heights=8, cmap="viridis", text="KR835  2023-10-14 00:09 UT")
sk2.save("dft_spectra.png")
sk2.close()
```

!!! note "subplot_kw override"
    `SkySummaryPlots` defaults to a polar projection (for skymap use).
    Pass `subplot_kw={}` when creating waterfall or spectra plots.

---

## RSF — Raw Sounding File

The RSF ionogram file (`data_format=4`) is the richest DPS4D output, storing
full echo data per frequency group including amplitude, phase, Doppler index,
and interferometrically-resolved arrival direction.

!!! note "Manual Reference"
    See [Digisonde-4D Manual](https://digisonde.com/pdf/Digisonde4DManual_LDI-web.pdf)
    Tables 5C-37 (PREFACE), 5C-38 (RSF Header), 5C-39 (Frequency Group layout).

### File layout

| Entity | Size | Description |
|--------|------|-------------|
| PREFACE | variable | Operator-selectable parameters (Table 5C-37, §4) |
| Block | 4 096 bytes | RSF Header + Frequency Groups |
| RSF Header | variable | Block-level metadata (Table 5C-38) |
| Frequency Group | variable | PRELUDE (6 bytes) + height profile |
| PRELUDE | 6 bytes | Per-group preamble (Table 5C-39) |

**Polarization rule**: PREFACE character #29 (`A`) < 8 → both O- and X-mode
groups stored; `A` ≥ 8 → O-mode only.

### RsfHeader fields

| Field | Type | Raw units | After `__post_init__` |
|-------|------|-----------|----------------------|
| `record_type` | `int` | raw | — |
| `header_length` | `int` | bytes | — |
| `version_maker` | `int` | raw | — |
| `year` / `month` / `dom` | `int` | calendar | — |
| `doy` | `int` | day of year | — |
| `hour` / `minute` / `second` | `int` | time | — |
| `stn_code_rx` / `stn_code_tx` | `str` | 000–999 | — |
| `schedule` | `int` | 1–6 | — |
| `program` | `int` | 1–7 (A–G) | — |
| `start_frequency` | `float` | ×100 Hz | → Hz (×1e2) |
| `coarse_frequency_step` | `float` | kHz | → Hz (×1e3) |
| `stop_frequency` | `float` | ×100 Hz | → Hz (×1e2) |
| `fine_frequency_step` | `float` | kHz | → Hz (×1e3) |
| `num_small_steps_in_scan` | `int` | −15 … 15; negative=no multiplex | — |
| `phase_code` | `int` | 1=complementary, 2=short, 3=75% duty, 4=100% duty, +8=no switch | — |
| `option_code` | `int` | 0=sum, 1–4=individual ant., 7=ant. scan, +8=O-only | — |
| `number_of_samples` | `int` | encoded (power of 2, 3–7) | — |
| `pulse_repetition_rate` | `int` | pps: 50, 100, 200; nibble4: 0=active, 1=silent | — |
| `range_start` | `int` | km (0–9999) | — |
| `range_increment` | `int/float` | encoded: 2→2.5 km, 5→5 km, 10→10 km | → km |
| `number_of_heights` | `int` | 128 / 256 / 512 | — |
| `delay` | `int` | ×15 km (0–1500) | — |
| `base_gain` | `int` | 0–7 (0–42 dB), +8=auto gain | — |
| `frequency_search` | `int` | 0=no, 1=yes | — |
| `operating_mode` | `int` | 0=VI, 1=Drift Std, 2=Drift Auto, 3=Cal, 4=HRR, 5=Beam, 6=PGH, 7=Test | — |
| `data_format` | `int` | 1=MMM, 2=Drift, 3=PGH, **4=RSF**, 5=SBF, 6=BIT | — |
| `threshold` | `float` | 3 dB over MPA | → 3×(threshold−10) dB |
| `constant_gain` | `int` | 0=full, 1=−9 dB sw-low, 2=−9 dB tk-low, 3=−18 dB | — |
| `cit_length` | `int` | ms (0–40000) | — |
| `journal` | `str` | bit0=new gain, bit1=new height, bit2=new freq, bit3=new case | — |
| `bottom_height_window` | `int` | km (0–9999) | — |
| `top_height_window` | `int` | km (0–9999) | — |
| `number_of_heights_stored` | `int` | 1–512 | — |
| `number_of_frequency_groups` | `int` | count | — |
| `date` | `datetime` | — | synthesized from year/month/dom/hour/min/sec |

### RsfFrequencyGroup fields

| Field | Type | Raw units | After `setup()` |
|-------|------|-----------|-----------------|
| `pol` | `str` | O / X | — |
| `group_size` | `int` | encoded: 2=262, 3=504, 4=1008 | — |
| `frequency_reading` | `float` | ×10 kHz within ionogram range | → Hz (×10 000) |
| `offset` | `int/str` | 0=−20 kHz, 1=−10 kHz, 2=0, 3=+10 kHz, 4=+20 kHz, 5=search fail, E=forced, F=no Tx | → Hz or label |
| `additional_gain` | `float` | ×3 dB, range 0–15 | → dB (×3) |
| `seconds` | `int` | 00–59 | — |
| `mpa` | `float` | Most Probable Amplitude, 0–31 | — |
| `amplitude` | `np.array` | ×3 dB, 0–31 | → dB (×3); values < mpa set to 0 |
| `dop_num` | `np.array` | Doppler index 0–7 | — |
| `phase` | `np.array` | 0–31, ×11.25° or ×1 km | → degrees (×11.25) |
| `azimuth` | `np.array` | 0–7, ×60° | → degrees (×60) |
| `height` | `np.array` | km | `range_start + index × range_increment` |
| `azm_directions` | `list[str]` | — | N, NE, SE, S, SW, NW |

### Echo direction coding (Figure 3-8 style)

The `azimuth` field is quantized to 60° sectors from the RSF 3-bit field:

| Azimuth (°) | `azm_directions` label | Directogram sign |
|------------|------------------------|-----------------|
| 0 | N | + (eastward) |
| 60 | NE | + |
| 120 | SE | + |
| 180 | S | − (westward) |
| 240 | SW | − |
| 300 | NW | − |

Full 10-category echo classification used by `add_direction_ionogram()`:

| Category | Color | Description |
|----------|-------|-------------|
| NoVal | gray | Below amplitude threshold |
| NNE | royalblue | North/north-northeast |
| E | dodgerblue | East |
| W | gold | West |
| Vo− | darkred | Vertical, negative Doppler (downward layer) |
| Vo+ | lightcoral | Vertical, positive Doppler (upward layer) |
| SSW | orange | South/south-southwest |
| X− | darkgreen | X-mode, negative Doppler |
| X+ | lightgreen | X-mode, positive Doppler |
| NNW | midnightblue | North-northwest |

### Python usage

```python
from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.digi_plots import RsfIonogram

extractor = RsfExtractor("KR835_2023287000000.RSF", extract_time_from_name=True, extract_stn_from_name=True)
extractor.extract()
df = extractor.to_pandas()

r = RsfIonogram(figsize=(6, 5), font_size=10)
r.add_direction_ionogram(df, ylim=[80, 600], xlim=[1, 15], xticks=[1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
                         text="KR835  2023-10-14 00:00 UT", lower_plimit=5, ms=1.0)
r.save("rsf_direction_ionogram.png")
r.close()
```

### Directogram (Figure 3-11/3-12 style)

Y-axis = UT time, X-axis = West–East ground distance D_i (km):

```
D_i = sqrt(H_i² − H_v²)    (negative for westward arrivals)
```

```python
import glob, pandas as pd
from pynasonde.digisonde.digi_plots import RsfIonogram

frames = [ex.to_pandas() for ex in [RsfExtractor(f, True) for f in sorted(glob.glob("KR835_*.RSF"))]]
df_day = pd.concat(frames, ignore_index=True)
r = RsfIonogram(figsize=(6, 8), font_size=10)
r.add_directogram(df_day, dlim=[-800, 800], lower_plimit=5, ms=0.5, text="KR835  2023-10-14")
r.save("rsf_directogram_daily.png")
r.close()
```

---

## SBF — Single Byte Format

The SBF format (`data_format=5`) is a reduced binary ionogram that omits
interferometric direction data.  It uses the same 4096-byte block structure and
PREFACE header as RSF but stores compact single-byte amplitude values only,
making it suitable for stations with single-antenna configurations or for
rapid low-bandwidth archiving.

### SbfHeader fields

The SBF header is **identical to the RSF header** (see [RsfHeader fields](#rsfheader-fields)
above) with `data_format=5`.  After `__post_init__`:

- `start_frequency`, `stop_frequency` → Hz (×1e2)
- `coarse_frequency_step`, `fine_frequency_step` → Hz (×1e3)
- `range_increment` → km (2→2.5, 5→5, 10→10)
- `threshold` → dB: `3 × (threshold − 10)`
- `date` → synthesized `datetime` from year/month/dom/hour/min/sec

### SbfFrequencyGroup fields

| Field | Type | Raw units | After `setup()` |
|-------|------|-----------|-----------------|
| `pol` | `str` | O / X | — |
| `group_size` | `int` | encoded: 2=262, 3=504, 4=1008 | — |
| `frequency_reading` | `float` | ×10 kHz | → Hz (×10 000) |
| `offset` | `int/str` | 0=−20 kHz, 1=−10 kHz, 2=0, 3=+10 kHz, 4=+20 kHz, 5=search fail, E=forced, F=no Tx | → Hz or label |
| `additional_gain` | `float` | ×3 dB, 0–15 | → dB (×3) |
| `seconds` | `int` | 00–59 | — |
| `mpa` | `float` | Most Probable Amplitude, 0–31 | — |
| `amplitude` | `np.array` | ×3 dB, 0–31 | → dB (×3) |
| `dop_num` | `np.array` | Doppler index 0–7 | — |
| `phase` | `np.array` | 0–31, ×11.25° | → degrees (×11.25) |
| `azimuth` | `np.array` | 0–7, ×60° | → degrees (×60) |
| `height` | `np.array` | km | `range_start + index × range_increment` |

!!! note "SBF vs RSF"
    SBF (`data_format=5`) and RSF (`data_format=4`) share the same PREFACE
    header layout.  The key difference is that SBF records do **not** include
    interferometric phase data across multiple antennas, so echo direction
    cannot be resolved — only amplitude and single-antenna phase are stored.

---

## See Also

- [SAO Example](../examples/digisonde/sao.md)
- [RSF Direction Ionogram Example](../examples/digisonde/rsf_direction_ionogram.md)
- [SAO Isodensity + DFT Waterfall Example](../examples/digisonde/sao_dft.md)
- [Digisonde-4D Technical Manual](https://digisonde.com/pdf/Digisonde4DManual_LDI-web.pdf)
- [GIRO — Global Ionosphere Radio Observatory](https://giro.uml.edu/)
- [ULCAR / UMass Lowell Digisonde Center](https://ulcar.uml.edu/)
- [SAO-4 Format Specification](https://ulcar.uml.edu/~iag/SAO-4.htm)
- [SAOXML 5 Specification](https://ulcar.uml.edu/SAOXML/SAO.XML%205.0%20specification%20v1.0.pdf)
