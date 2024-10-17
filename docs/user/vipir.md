<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:

-->

# VIPIR: _Vertical Incidence Pulsed Ionospheric Radar_
The Vertical Incidence Pulsed Ionospheric Radar (VIPIR) was initially developed by Scion Associates under a Small Business Innovative Research grant from the U.S. Air Force Research Laboratory. The first installation of VIPIR took place at NASA's Wallops Island Flight Facility in 2008. Since then, 15 VIPIR instruments have been deployed in various locations.

In 2015, Scion Associates introduced Version 2 of the radar system, further enhancing its capabilities. This software suite is compatible with both the original and second-generation VIPIR systems, providing a robust tool for ionospheric research.

For a detailed technical description of the radar, please refer to Grubb et al.

## VIPIR Data Output: Raw In-phase and Quadrature (RIQ) Files
The output data from VIPIR is stored in Raw In-phase and Quadrature (RIQ) files. These files contain multiple range gate samples from the Digital Down Converter for each of the eight radar receive channels. Each range gate and receiver has an associated in-phase and quadrature sample.

In addition to the raw data blocks, each RIQ file includes metadata records such as the Sounding Control Table (SCT) and the Pulse Control Table (PCT), which define the instrument’s mode of operation. The metadata also provides site-specific details, including the station location and antenna configuration.

The RIQ file is a custom binary format. The structure of the data is best understood through `C`, and `FORTRAN` definitions. However, we provide `Python`-version of the data structure. 

## RIQ Data Structure (Pythonic)
An RIQ file is divided into blocks or records. Each record has the same format but can have varying length depending on the amount of scatter observed by the radar.

### Sounding Control Table (SCT)
Here is the SCT `Python` structure. However, `C` and `FORTRAN` both structure formats are also supported, and produce nearly idential files. The exception is for the user-defined text strings, where `C` prodices a null filled character string and `FORTRAN` produces a space filled character string. Both methods are supported. For 64 bit C code, it is necessary to define the structure as `packed`. This version defined here is 1.20.

| Field Name              | Type     | Size(Bytes) | Note / Description  |
| :---------------- | :------: | :------: | ----: |
| Magic             |  16 Bit-`Integer`        | 2 | Magic number `0x51495200` (/nullRIQ) Possibly Byte Reversed |
| sounding_table_size  |  16 Bit-`Integer`     | 2 | Bytes in sounder configuration structure |
| pulse_table_size  |  16 Bit-`Integer`     | 2 | Bytes in pulse configuration structure |
| raw_data_size  |  16 Bit-`Integer`     | 2 | Bytes in raw data block (one PRI) |
| struct_version  |  64 Bit-`Float`     | 8 | Format Version Number.  Currently 1.2 |