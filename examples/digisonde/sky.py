"""MkDocs example showing how to render a DIGISONDE sky map.

The workflow highlights the typical steps:

1. Provide the path to a `.SKY` file when constructing `SkyExtractor`.
2. Call `extract` to parse the raw data and convert the result to a
   pandas dataframe via `to_pandas`.
3. Use `SkySummaryPlots` to create a sky map, tweak plot options as needed,
   and save the image for later inspection.

Update the sample file path to point at your own data before running the
example; any generated artifacts (plot image, temporary files) are written
to the local `tmp/` directory.
"""

from pynasonde.digisonde.digi_plots import SkySummaryPlots
from pynasonde.digisonde.parsers.sky import SkyExtractor

# Build a parser instance with a target .SKY file plus flags controlling
# whether auxiliary frequency and angle tables are loaded.
extractor = SkyExtractor(
    # "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286000915.SKY",
    "tmp/20250527/KW009_2025147000426.SKY",
    True,
    True,
)
# Trigger the raw read/parsing routine; accessing `.dataset` demonstrates that
# metadata such as the most recent frequency header is now available.
extractor.extract().dataset[-1].freq_headers
# Convert the parsed structure into a tidy dataframe for plotting/pandas work.
df = extractor.to_pandas()
# Instantiate the plot helper that encapsulates matplotlib figure creation.
skyplot = SkySummaryPlots()
# Render a sky map using Doppler frequency as the color surface; adjust limits
# and labels to taste for your dataset.
skyplot.plot_skymap(
    df,
    zparam="spect_dop_freq",
    text=f"Skymap:\n {extractor.stn_code} / {extractor.date.strftime('%H:%M:%S UT, %d %b %Y')}",
    # cmap="jet",
    clim=[-1, 1],
    rlim=6,
)
# Persist the figure and release underlying matplotlib resources.
skyplot.save("tmp/extract_sky.png")
skyplot.close()
