"""Example script for documenting DVL (drift velocity) plots in MkDocs.

The recipe:

1. Gather one or more directories containing `.DVL` files and load them with
   `DvlExtractor.load_DVL_files`.
2. Use `SkySummaryPlots.plot_dvl_drift_velocities` to build the stacked drift
   velocity figure.
3. Customize axes (time range, virtual height overlay, etc.), then export the
   plot into `docs/examples/figures/` so it can be embedded directly in the
   generated documentation.

Update the `data_dirs` list and the `date` variable to match your dataset
before running the example.
"""

import datetime as dt
import os

os.system("mkdir tmp/pynasondev1/")

import matplotlib.dates as mdates

from pynasonde.digisonde.digi_plots import SkySummaryPlots
from pynasonde.digisonde.digi_utils import setsize
from pynasonde.digisonde.parsers.dvl import DvlExtractor

date = dt.datetime(
    2023, 10, 14
)  # Baseline day for the time-axis window; adjust per dataset.
font_size = 18
setsize(font_size)
# Collect all DVL file records from each directory. Each DVL file is read in parallel
# using `n_procs` CPU cores. Files are typically holding one timestamp each, so to create
# a continuous time series, gather files from a single day or multiple days.
dvl_df = DvlExtractor.load_DVL_files(
    [
        "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/"
    ],
    n_procs=12,
)
dvlplot = SkySummaryPlots.plot_dvl_drift_velocities(  # Generate the stacked drift velocity panels.
    dvl_df, fname=None, draw_local_time=False, figsize=(7, 4.5), font_size=font_size
)

# Improve figure readability for publication (larger canvas, fonts, and markers).
# dvlplot.set_size_inches(7.5, 5.2)
for k, axis in enumerate(dvlplot.axes):
    axis.tick_params(axis="both", labelsize=font_size)
    axis.grid(alpha=0.25, which="both", linestyle="--")
    axis.text(
        0.95, 0.95, f"({chr(k+97)})", ha="right", va="center", transform=axis.transAxes
    )

ax = dvlplot.axes[0]
ax.xaxis.set_major_locator(
    mdates.HourLocator(interval=6)
)  # 6-hour ticks along the bottom panel.
ax.set_xlim([date, date + dt.timedelta(1)])  # Highlight a single-day interval.

ax = dvlplot.axes[1]
axt = ax.twinx()  # Overlay virtual height on a secondary axis.
axt.scatter(
    dvl_df.datetime,
    0.5 * (dvl_df.Hb + dvl_df.Ht),
    marker="D",
    s=15,
    color="m",
    linewidths=0.3,
)
axt.set_ylabel("Virtual Height, km", fontdict={"color": "m"})
axt.tick_params(axis="both", labelsize=font_size, colors="m")
axt.set_ylim(250, 500)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.set_xlim([date, date + dt.timedelta(1)])

ax = dvlplot.axes[2]
ax.set_ylim(-20, 20)  # Symmetric vertical velocity bounds for clarity.
ax = dvlplot.axes[2]
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.set_xlim([date, date + dt.timedelta(1)])

# dvlplot.tight_layout(pad=1.2)
dvlplot.fig.savefig(
    "tmp/pynasondev1/PynasondeV1-dvl - SoftwareX.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
)
dvlplot.save(
    f"docs/examples/figures/stackplots_dvl.png"
)  # Persist figure for documentation reuse.
dvlplot.close()
