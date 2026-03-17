# RSF — Parse and Inspect Raw Sounding File

<div class="hero">
  <h3>Low-Level RSF Walkthrough</h3>
  <p>
    Load a single DPS4D <code>.RSF</code> file, parse all blocks and frequency
    groups into structured Python dataclasses, and inspect headers and echo data
    programmatically.
  </p>
</div>

This page explains `examples/digisonde/rsf.py`.

## Call Flow

1. `RsfExtractor(filepath, ...)` opens the `.RSF` file and resolves optional
   metadata tables (station codes, frequency tables).
2. `.extract()` iterates over all 4096-byte blocks, decoding the RSF Header and
   Frequency Groups into `RsfDataUnit` objects stored in `rsf_data.rsf_data_units`.
3. Inspect `rsf_data_units[i].header` for block-level metadata (timestamp,
   frequency, height settings) and `frequency_groups[j]` for per-height echo data
   (amplitude, phase, Doppler number, direction bits).
4. `.to_pandas()` flattens all parsed units into a tidy DataFrame suitable for
   plotting or downstream analysis.

## Key Code

### 1) Load and Extract

```python
from pynasonde.digisonde.parsers.rsf import RsfExtractor

extractor = RsfExtractor(
    "/path/to/KR835_2023287000000.RSF",
    extract_time_from_name=True,   # parse timestamp from filename
    extract_stn_from_name=True,    # parse station URSI code from filename
)
extractor.extract()
```

### 2) Inspect Header and Frequency Groups

```python
# First block header: timestamp, frequency, height parameters
h = extractor.rsf_data.rsf_data_units[0].header
print(f"Date:      {h.date}")
print(f"Frequency: {h.frequency_group} MHz")

# First frequency group: one height profile of echoes
fg = extractor.rsf_data.rsf_data_units[0].frequency_groups[0]
print(f"Amplitude values: {fg.amplitude[:8]}")
print(f"Direction bits:   {fg.azimuth[:8]}")
```

### 3) Convert to DataFrame

```python
df = extractor.to_pandas()
print(df[["datetime", "frequency", "height_km", "amplitude", "azimuth"]].head(10))
```

## Run

```bash
cd /home/chakras4/Research/CodeBase/pynasonde
python examples/digisonde/rsf.py
```

## Related Files

- `examples/digisonde/rsf.py`
- `pynasonde/digisonde/parsers/rsf.py`
- `pynasonde/digisonde/datatypes/rsfdatatypes.py`

## See Also

- [RSF Direction Ionogram + Directogram](rsf_direction_ionogram.md)
- [DIGISONDE Format Guide](../../user/digisonde.md#rsf----raw-sounding-file)
- [RSF API Reference](../../dev/digisonde/parsers/rsf.md)
