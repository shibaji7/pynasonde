"""MkDocs example illustrating two SAO (scaled ionogram) workflows.

Height-profile panel
--------------------
1. Load one or more `.SAO` directories with `SaoExtractor.load_SAO_files`.
2. Derive electron density (`ed`) profiles and rescale them to 10^6 cm^-3 units.
3. Plot time–height density profiles and save the figure for documentation.

Scaled-parameter panel
----------------------
1. Reload the same files using the `scaled` extractor function.
2. Visualize `foF2` and `hmF2` on dual y-axes to capture F2-layer evolution.
3. Export the plot into `docs/examples/figures/` for MkDocs reuse.

Update the `folders` list and baseline `date` to match your campaign before
running the script.
"""

import datetime as dt

import matplotlib.dates as mdates

from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import setsize
from pynasonde.digisonde.parsers.sao import SaoExtractor

date = dt.datetime(2023, 10, 14)  # Reference day for constraining the time axis.
font_size = 16
setsize(font_size)
# Height-profile view: ingest SAO files and compute electron density profiles.
df = SaoExtractor.load_SAO_files(
    folders=[
        "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/"
    ],
    func_name="height_profile",
    n_procs=12,
)

# Convert electron density to units of 10^6 cm^-3 to simplify the colorbar.
df.ed = df.ed / 1e6

sao_plot = SaoSummaryPlots(
    figsize=(8, 4),
    fig_title="KR835 / Ne Profiles (derived parameters) during 2023 GAE",
    draw_local_time=False,
    font_size=font_size,
)
sao_plot.add_TS(
    df,
    zparam="ed",
    prange=[0, 1],
    zparam_lim=10,
    cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
    plot_type="scatter",
    scatter_ms=20,
)

# Tighten the x-axis to a single-day window with 6-hour ticks for readability.
ax = sao_plot.axes
ax.set_xlim([date, date + dt.timedelta(1)])
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

# Persist the figure alongside other documentation assets.
sao_plot.save("docs/examples/figures/stack_sao_ne.png")
sao_plot.fig.savefig(
    "tmp/pynasondev1/PynasondeV1-F2,01 - SoftwareX.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
)
sao_plot.close()


# F2-layer view: reload the SAO files using the `scaled` product to pull summary parameters.
df = SaoExtractor.load_SAO_files(
    folders=[
        "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/"
    ],
    func_name="scaled",
    n_procs=12,
)


sao_plot = SaoSummaryPlots(
    figsize=(8, 4),
    fig_title="KR835 / F2 (scaled) response during 14 Oct 2023 GAE",
    draw_local_time=False,
    font_size=font_size,
)
# Plot dual-axis F2 parameters (critical frequency and peak height).
sao_plot.plot_TS(
    df,
    right_yparams=["hmF2"],
    left_yparams=["foF2"],
    right_ylim=[100, 400],
    left_ylim=[1, 15],
)

# Tighten the x-axis to a single-day window with 6-hour ticks for readability.
ax = sao_plot.axes
ax.set_xlim([date, date + dt.timedelta(1)])
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

# Persist the figure alongside other documentation assets.
sao_plot.save("docs/examples/figures/stack_sao_F2.png")
sao_plot.fig.savefig(
    "tmp/pynasondev1/PynasondeV1-F2,02 - SoftwareX.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
)
sao_plot.close()
