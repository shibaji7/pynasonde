<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:

-->

<style>
tr:nth-child(even) {
  background-color: #b2b2b2!important;
  color: #f4f4f4!important;
}
</style>

# DIGISONDE: _Digital Ionospheric Goniometric IonoSONDE_
The Digisonde is a cutting-edge ionospheric radar designed for remote sensing of the ionosphere using high-frequency (HF) radio waves. This technology, rooted in the ionosonde innovation pioneered by Sir Edward Appleton in the late 1920s, stands for `Digital Ionospheric Goniometric IonoSONDE`.

The Digisonde captures a comprehensive range of ionospheric echo parameters, including:

* Amplitude and phase of signals
* Direction of arrival
* Virtual height
* Doppler frequency and spread
* Polarization (ordinary and extraordinary waves)

From these ionospheric echoes, the Digisonde provides:

* **Electron Density Profiles**: Near real-time ionospheric electron density profiles with error bars for each height.
* **Vertical Total Electron Content (VTEC)**: Precise measurements of ionospheric electron content.
* **Classical Ionospheric Characteristics**: Real-time parameters such as foF2, foF1, foE, foEs, MUF(3000)F2, hmF2, hmF1, hmE, and the IRI parameters B0 and B1.

Additionally outputs are following:

* E and F region drifts. 
* Near real-time radio skymaps of ionospheric reflection points for vertical and oblique Digisonde-to-Digisonde sounding.

The Digisonde is an indispensable tool for real-time ionospheric monitoring, enabling advanced research and operational forecasting of ionospheric conditions.

## DIGISONDE Data Outputs:
Outputs from a DIGISONDE are stored in `.SAO`, `.SKY`, `.DVL`, `.DFT` and `.RSF` formats. Each type of file stores different types datasets.

| File Extension              | File Type     | Note / Description  |
| :---------------- | :------: | ------: |
| `.SAO` | Structured ASCII / XML | Standard Archiving Output: Stores scaled ionosonde data in a compact, machine- and human-readable format. |
| `.SKY` | Structured ASCII | Skymap data files stores raw echo traces after inversion. |
| `.DVL` | Structured ASCII | Drift Velocity data files stores velocity estimates from the peak echoe locations. |
| `.DFT` | Structured Binary | Drift Fourier spectra files stores full Fourier transform spectra of Doppler-shifted signals (Hz bins, power values). |
| `.RSF` | Structured Binary | Raw Sounding File stores  |

All these files are either a custom binary format or structured ASCII formats. Structures of the datas is best understood through `C`, and `FORTRAN` definitions. However, we provide `Python`-version of the data structure.

### Standard Archiving Output (SAO) Format and SAOXML 5
This file is stored in a structured ASCII format, with each record representing a complete Digisonde sounding for a specific date and time, together with its corresponding parameters. A detailed description of these parameters is available at this [location](https://ulcar.uml.edu/~iag/SAO-4.htm).

To facilitate unified data exchange between ionosonde data producers and users of ionogram-derived characteristics, a new format has been introduced: **SAOXML 5**. This format reflects its heritage from the earlier Standard Archiving Output (SAO) version 4 while adopting the widely used XML language for general-purpose data exchange.

The **SAOXML 5 specification** serves as the authoritative reference for developing input and output interfaces in software projects that read and write ionogram-derived data. The associated paper presents the motivation for introducing this format, outlines the guiding principles of its design and use, and provides further details, which can be found at this [location](https://ulcar.uml.edu/SAOXML/SAO.XML%205.0%20specification%20v1.0.pdf).

### Skymap Datafile (SKY) Format
This file is in a structured **ASCII format**, with each file providing a complete **DIGISONDE sounding** for a specific date and time, along with the corresponding parameters. The following table provides an overview of the entries in the data file. `.SKY` files typically hold multiple entries of the following structure.

| Field Name             | Type         | Size (Bytes) | Note / Description                                                  |
|------------------------|--------------|--------------|----------------------------------------------------------------------|
| zenith_angle           | `float`      | 8            | Zenith angle of arrival of the echo (deg or rad)                     |
| sampl_freq             | `float`      | 8            | Sampling frequency used during the sounding (Hz)                     |
| group_range            | `float`      | 8            | Group path / virtual range of the echo (km)                          |
| gain_ampl              | `float`      | 8            | Receiver gain amplitude                                              |
| height_spctrum_ampl    | `array[float]` | variable   | Amplitude spectrum along height bins                                 |
| max_height_spctrum_ampl| `float`      | 8            | Maximum amplitude in the height spectrum                             |
| n_sources              | `int`        | 4            | Number of detected echo sources                                      |
| height_spctrum_cl_th   | `float`      | 8            | Threshold value for height spectrum classification                   |
| spect_line_cl_th       | `float`      | 8            | Threshold value for spectral line classification                     |
| polarization           | `string`     | variable     | Echo polarization (e.g., O or X mode)                                |
| x_coord                | `float`      | 8            | X-coordinate in the skymap (antenna array projection)                |
| y_coord                | `float`      | 8            | Y-coordinate in the skymap (antenna array projection)                |
| spect_amp              | `float`      | 8            | Amplitude of the spectral line                                       |
| spect_dop              | `int`        | 4            | Raw Doppler index or shift unit                                      |
| spect_dop_freq         | `float`      | 8            | Doppler shift frequency (Hz) derived from spect_dop                  |
| rms_error              | `float`      | 8            | Root-mean-square error of the estimate                               |
| datetime               | `datetime`   | variable     | UTC datetime of the sounding                                         |
| local_datetime         | `datetime`   | variable     | Local datetime of the sounding (station local time)                  |

### Differential Velocity (DVL) Format
This file is formatted in structured ASCII, with each entry corresponding to a specific date and time, along with the associated parameters.

| Field Name              | Type     | Size(Bytes) | Note / Description  |
| :---------------- | :------: | :------: | ----: |
| type             |  `str`        | 1 | Default value `DVL` |
| version             |  `str`        | 1 | Version of the data-processing |
| station_id       |  `int32`        | 4 | Station ID |
| ursi_tag       |  `str`        | 8 | URSI tag of the  |
| lat       |  `float64`        | 8 | Latitude of the station |
| lon       |  `float64`        | 8 | Longitude of the station |
| date       |  `dt.date`        | 16 | `python` date of the record |
| doy       |  `int32`        | 4 | Day of the year |
| time       |  `dt.time`        | 16 | `python` time of the record |
| Vx       |  `float64`        | 8 | Velocity in m/s along magnetic north direction |
| Vx_err       |  `float64`        | 8 | Error in `Vx` in m/s |
| Vy       |  `float64`        | 8 | Velocity in m/s along magnetic east direction |
| Vy_err       |  `float64`        | 8 | Error in `Vy` in m/s |
| Az       |  `float64`        | 8 | Velocity in m/s along counted clockwise from the magnetic north |
| Az_err       |  `float64`        | 8 | Error in `Az` in m/s |
| Vh       |  `float64`        | 8 | Velocity in m/s along geographic height |
| Vh_err       |  `float64`        | 8 | Error in `Vh` in m/s |
| Vz       |  `float64`        | 8 | Velocity in m/s along zenith |
| Vz_err       |  `float64`        | 8 | Error in `Vz` in m/s |
| Cord       |  `str`        | 1 | Coordinate: COM [Compass], GEO [Geographic], CGm [Corrected Geromagnetic]|
| Hb       |  `float64`        | 8 | Bottom (virtual) height of the velocity measurements, km |
| Ht       |  `float64`        | 8 | Top (virtual) height of the velocity measurements, km |
| Fl       |  `float64`        | 8 | Lower operating frequency, MHz |
| Fu       |  `float64`        | 8 | Upper operating frequency, MHz |

### Legacy Scinece Data Format: DFT
The **`.DFT` drift data file** consists of a variable number of **4,096-byte blocks**, where each block contains **8-bit amplitudes and phases** of the Doppler spectra.
#### Data Organization

- **Smallest entity**: a **sub-case**
  - A sub-case contains **four Doppler spectra** for each antenna
  - Obtained at:
    - one frequency
    - one height gate
    - one polarization setting

- **Doppler spectra length**:
  - Each Doppler spectrum is `2N` elements long
  - `N` is an operator-selected parameter

#### Storage Format

- The amplitudes of Doppler spectra are grouped into sets of **128 elements** for storage.
- Each set of 128 amplitudes may include data from **one to four antennas**, depending on the value of `N`.
- After 128 amplitudes, the block stores the corresponding **128 phase values** of the same Doppler spectrum (or spectra).

#### Summary Table

| Data Entity         | Description                                                     |
|---------------------|-----------------------------------------------------------------|
| Block size          | 4,096 bytes (fixed)                                            |
| Data type           | 8-bit integers (amplitude and phase values)                    |
| Smallest unit       | Sub-case (per frequency × height gate × polarization)           |
| Doppler spectra     | Four per antenna, each `2N` elements long                       |
| Grouping            | 128 amplitude values, followed by 128 phase values              |
| Antenna data        | Each 128-element set may include 1–4 antennas (depending on `N`) |

For details please check [Digisonde Manual](https://digisonde.com/pdf/Digisonde4DManual_LDI-web.pdf) section 5.166.

### Raw Sounding Data Files: RSF

The **RSF ionogram file** is composed of 4,096-byte blocks.  
Each block contains a **Header** and a variable number of **Frequency Groups**.

Refer to the *Digisonde Technical Manual*:  
- **Table 5C-37** — General Purpose PREFACE format  
- **Table 5C-38** — RSF Header structure  
- **Table 5C-39** — Frequency Group length settings  
- **Section 4** — Explanation of operator-selectable PREFACE parameters  

---

#### RSF File Layout

| Entity           | Size (Bytes)  | Description                                                                                     | Reference in Manual      |
|------------------|---------------|-------------------------------------------------------------------------------------------------|--------------------------|
| PREFACE          | variable      | General Purpose PREFACE containing operator-selectable parameters and defaults                  | Table 5C-37; Sec. 4      |
| Block            | 4,096         | Fundamental RSF storage unit, consisting of a Header + Frequency Groups                        | —                        |
| RSF Header       | variable      | Header structure for each block                                                                | Table 5C-38              |
| Frequency Group  | variable      | Unit containing PRELUDE + one height profile                                                    | Table 5C-39              |
| PRELUDE          | 6             | Fixed-size preamble for each Frequency Group                                                    | Table 5C-39              |
| Height Profile   | variable      | Virtual height data, length depends on operator-selected number of heights                      | Table 5C-39              |

---

#### Polarization Option

| Condition           | Storage Behavior                                                                 |
|---------------------|----------------------------------------------------------------------------------|
| `A < 8`             | Both **O- and X-mode** polarizations stored, each in its own Frequency Group     |
| `A ≥ 8`             | Only **O-mode** polarization stored for each frequency                          |

*(`A` = PREFACE Character #29 parameter)*

---
