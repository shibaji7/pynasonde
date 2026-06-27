# CADI Format Notes

Reference files currently inspected:

- `tmp/CADI/6E131200.md4`
- `tmp/CADI/6E131200.txt`
- `tmp/CADI/6E131200.csv`
- `tmp/CADI/mdx2txt_cliV2.py`
- `tmp/CADI/mdx2csv_cliV2.py`

## Header Layout

The MD4 sample begins with mixed ASCII and little-endian binary fields:

- site: 3 bytes, e.g. `SHA`
- ASCII datetime: 22 bytes, e.g. `May 13 12:00:00 2026`
- filetype: 1 byte, e.g. `I`
- number of frequencies: uint16
- number of Doppler bins: uint8
- minimum height: uint16
- maximum height: uint16
- pulses per second: uint8
- number of pulses averaged: uint8
- base threshold x100: uint16
- noise threshold x100: uint16
- minimum Doppler for save: uint8
- time between measurements: uint16
- gain control: 1 byte
- signal processing: 1 byte
- number of receivers: uint8
- spares: 11 bytes

The sample header decodes to:

- site: `SHA`
- filetype: `I`
- `nfreqs`: 310
- `ndops`: 8
- min/max height: 90/1020 km
- number of receivers: 3

## Frequency Table

After the header, the file stores `nfreqs` little-endian float32 values. These are frequencies in Hz.

The extractor exposes both:

- `frequency_hz`
- `frequency_mhz`

## Payload Loop

Parsing follows the reference converter control flow:

1. Read `time_min`.
2. Continue until `time_min == 255`.
3. Read `time_sec`.
4. Read a flag byte that acts as the first gain/height control value.
5. Loop over each frequency index.
6. For each frequency, read:
   - noise flag
   - noise power x10
   - next flag
7. While `flag < 224`, decode height/Doppler detections:
   - `hflag = flag`
   - read `ndops_oneh`
   - if `ndops_oneh >= 128`, subtract 128 and add 200 to `hflag`
   - for each Doppler detection, read `doppler_flag`
   - read I/Q bytes for each receiver
8. Continue until record/file terminator logic matches the converter scripts.

## Height

The reference converter uses a fixed height spacing:

```text
height_km = height_flag * 3.0
```

This is captured in `CadiReader(dheight_km=3.0)` and propagated through `CadiExtractor`.

## Power and Phase

I/Q bytes are unsigned in the file. Product extraction converts them to signed values with:

```text
signed = value - 256 if value > 127 else value
```

Derived products:

- amplitude: `sqrt(I^2 + Q^2)`
- phase: `atan2(Q, I)`
- mean power: `20 * log10(mean(receiver_amplitudes))`

Zero mean amplitude maps to `0.0` dB to match the legacy converter style.

## Doppler

Current implementation exposes:

- `doppler_flag`
- `doppler_bin`

This is still an index/bin value, not calibrated Hz. A physical `doppler_hz` column needs calibration metadata:

- zero-bin convention
- bin spacing in Hz
- sign convention
- any CADI-specific FFT/integration details

## O/X Mode

Current CADI MD4 parsing does not expose explicit O/X mode labels. Any O/X separation must be treated as a future classification/inference layer, not as a directly decoded file field.
