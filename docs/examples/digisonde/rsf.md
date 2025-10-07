# DIGISONDE RSF Example

This short walkthrough explains how to ingest DIGISONDE `.RSF` files with
`pynasonde`. The sample uses data from the 14 October 2023 Great American Annular
Eclipse and highlights how to inspect the structured contents of an RSF product.
The runnable script lives at [`examples/digisonde/rsf.py`](https://github.com/shibaji7/pynasonde/examples/digisonde/rsf.py),
and the extractor implementation resides in
[`pynasonde/digisonde/parsers/rsf.py`](https://github.com/shibaji7/pynasonde/pynasonde/digisonde/parsers/rsf.py).

## Workflow overview

1. Instantiate `RsfExtractor` with a path to a `.RSF` file plus optional flags that
   load supplementary metadata tables.
2. Call `extract()` to populate `rsf_data_units`, which organize header information and
   frequency groups for each observation.
3. Inspect the parsed content (headers, frequency groups) to guide downstream processing.

## Example script

The snippet below mirrors `examples/digisonde/rsf.py`. Update the file path to target
your own `.RSF` dataset and print whichever elements you need to validate.

```python
"""Quick-start example for loading RSF files with `pynasonde`.

Follow the steps:

1. Point `RsfExtractor` at a `.RSF` file and enable optional table loading as needed.
2. Call `extract()` to parse the structured RSF product into `rsf_data_units`.
3. Inspect headers or frequency groups to drive quality control or visualization.

Update the sample file path to match your own dataset before running the script.
"""

from pynasonde.digisonde.parsers.rsf import RsfExtractor

# Configure the extractor with a target RSF file and optional frequency/angle tables.
extractor = RsfExtractor(
    "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/KR835_2023287000000.RSF",
    True,
    True,
)

# Parse the file to populate `rsf_data` with structured data units.
extractor.extract()

# Print the first header and frequency group for quick inspection/debugging.
print(extractor.rsf_data.rsf_data_units[0].header)
print(extractor.rsf_data.rsf_data_units[0].frequency_groups[0])
```

> For richer analysis, iterate through all `rsf_data_units` or convert them into pandas
> data structures. Pairing RSF outputs with simultaneous SAO or DVL products can help
> correlate spread-F activity with electron-density and drift-velocity signatures.
