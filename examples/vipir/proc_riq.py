"""Example script for converting VIPIR RIQ files into ionogram plots for MkDocs.

Steps covered:

1. Load a `.RIQ` file with `RiqDataset.create_from_file`, applying the correct VIPIR
   configuration and character encoding.
2. Apply the adaptive gain filter (baseline removal + optional median filter) to clean
   the raw ionogram.
3. Plot the resulting ionogram with `Ionogram` and save the figure into
   `docs/examples/figures/` so the documentation can embed the output.

Update `fname` to point at your own RIQ capture before running the example.
"""

import numpy as np

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.vipir.ngi.plotlib import Ionogram
from pynasonde.vipir.riq.parsers.read_riq import (
    VIPIR_VERSION_MAP,
    RiqDataset,
    adaptive_gain_filter,
)

# Path to the RIQ file to visualize; replace with your own VIPIR dataset.
font_size = 20
setsize(font_size)
fname = "examples/data/PL407_2024058061501.RIQ"

# Create a dataset object using the appropriate VIPIR configuration/encoding.
riq = RiqDataset.create_from_file(
    fname,
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[0],
)

# Generate an ionogram and suppress background noise via adaptive gain + median filter.
ion = adaptive_gain_filter(
    riq.get_ionogram(threshold=50, remove_baseline_noise=True),
    apply_median_filter=True,
    median_filter_size=3,
)

# Replace NaNs introduced by filtering with zero power for stable plotting.
ion.powerdB[np.isnan(ion.powerdB)] = 0.0

# Set up a single-panel ionogram canvas.
p = Ionogram(ncols=1, nrows=1, font_size=font_size, figsize=(7, 5))

# Render the ionogram using power (dB) as the color surface.
p.add_ionogram(
    frequency=ion.frequency,
    height=ion.height,
    value=ion.powerdB,
    mode="O/X",
    xlabel="Frequency, MHz",
    ylabel="Virtual Height, km",
    ylim=[70, 1000],
    xlim=[1.8, 22],
    add_cbar=True,
    cbar_label="Power, dB",
    prange=[0, 70],
    del_ticks=False,
)

# Persist the figure for inclusion in the documentation; adjust path as needed.
p.save("docs/examples/figures/ionogram_from_riq.png")
p.close()
