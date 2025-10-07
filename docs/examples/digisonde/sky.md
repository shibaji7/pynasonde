# DIGISONDE Sky Map Example

This example walks through the typical workflow for turning a DIGISONDE
`.SKY` file into a publishable sky map image with `pynasonde`. The runnable
script lives at [`examples/digisonde/sky.py`](../../../examples/digisonde/sky.py),
and the parsing logic it exercises is documented in
[`pynasonde/digisonde/parsers/sky.py`](../../../pynasonde/digisonde/parsers/sky.py).

## Steps overview

1. Instantiate `SkyExtractor` with the path to a `.SKY` file and optional flags
   that control whether frequency and angle tables are loaded.
2. Call `extract()` to parse the raw file, then convert the result to a pandas
   dataframe with `to_pandas()` for easier downstream use.
3. Create a `SkySummaryPlots` helper to render the sky map, tweak the display
   parameters, and save the plot to disk.

Replace the sample file path with one of your own datasets before running the
script. Any generated plots or temporary artifacts are written to the local
`tmp/` directory by default.

```python
from pynasonde.digisonde.digi_plots import SkySummaryPlots
from pynasonde.digisonde.parsers.sky import SkyExtractor

# Set up the extractor with your .SKY file and optional frequency/angle tables.
extractor = SkyExtractor(
    # "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286000915.SKY",
    "tmp/20250527/KW009_2025147000426.SKY",
    True,
    True,
)

# Parse the raw file and inspect the latest frequency header if needed.
extractor.extract().dataset[-1].freq_headers

# Convert structured output to a pandas dataframe for plotting.
df = extractor.to_pandas()

# Build the sky plot helper and render the Doppler frequency skymap.
skyplot = SkySummaryPlots()
skyplot.plot_skymap(
    df,
    zparam="spect_dop_freq",
    text=f"Skymap:\n {extractor.stn_code} / {extractor.date.strftime('%H:%M:%S UT, %d %b %Y')}",
    # cmap="jet",  # Uncomment to explore a different color map.
    clim=[-1, 1],
    rlim=6,
)

# Persist the figure and close underlying matplotlib resources.
skyplot.save("tmp/extract_sky.png")
skyplot.close()
```

The `clim` and `rlim` parameters shown above are good starting points for the
sample dataset; adjust them to match the dynamic range and physical extents of
your own observations.
