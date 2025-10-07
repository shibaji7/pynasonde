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
