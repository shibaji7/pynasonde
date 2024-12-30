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
| `.SAO` | Structured ASCII |  |
| `.SKY` | Structured ASCII |  |
| `.DVL` | Structured ASCII |  |
| `.DFT` | Structured Binary |  |
| `.RSF` | Structured Binary |  |

All these files are either a custom binary format or structured ASCII formats. Structures of the datas is best understood through `C`, and `FORTRAN` definitions. However, we provide `Python`-version of the data structure.

### Standard Archiving Output (SAO) Format
This file is in a structured ASCII format, with each file providing a complete DIGISONDE sounding for a specific date and time, along with the corresponding parameters.
### Skymap Datafile (SKY) Format
This file is in a structured ASCII format, with each file providing a complete DIGISONDE sounding for a specific date and time, along with the corresponding parameters.
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