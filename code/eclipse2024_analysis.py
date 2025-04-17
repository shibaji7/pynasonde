import sys

from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.digisonde.dvl import DvlExtractor
from pynasonde.digisonde.sao import SaoExtractor

sys.path.append("code/")
import datetime as dt

import matplotlib.dates as mdates
from eclipse_utils import create_eclipse_path_local


def generate_digisonde_pfh_profiles(
    folders,
    fig_file_name,
    fig_title="",
    draw_local_time=True,
    stns=[],
):
    dfs = [
        SaoExtractor.load_SAO_files(
            folders=[folder],
            func_name="height_profile",
            n_procs=12,
        )
        for folder in folders
    ]
    dfscs = [
        SaoExtractor.load_SAO_files(
            folders=[folder],
            func_name="scaled",
            n_procs=12,
        )
        for folder in folders
    ]
    N = len(folders)
    sao_plot = SaoSummaryPlots(
        font_size=10,
        figsize=(5, 3 * N),
        nrows=N,
        fig_title=fig_title,
        draw_local_time=draw_local_time,
    )
    for i in range(N):
        stn_info = get_digisonde_info(stns[i])
        df = dfs[i]
        print(df.datetime, df.local_datetime)
        df.ed = df.ed / 1e6
        dfsc = dfscs[i]
        ocl = create_eclipse_path_local(
            dfsc.datetime, stn_info["LAT"], stn_info["LONG"]
        )
        ax, _ = sao_plot.add_TS(
            df,
            zparam="ed",
            prange=[0, 1],
            zparam_lim=10,
            cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
            plot_type="scatter",
            title="Stn Code: " + stns[i],
            add_cbar=True,
            xlabel="Time, UT" if i == 2 else "",
        )
        ax.plot(
            dfsc.datetime,
            dfsc.hmF1,
            "+",
            color="lightgreen",
            ls="None",
            ms="5",
            zorder=4,
        )
        axt = ax.twinx()
        # axt.xaxis.set_major_locator(mdates.HourLocator())
        axt.plot(dfsc.datetime, 1 - ocl, color="k", ls="--", lw=1.2)
        axt.set_ylim(0, 1)
        axt.set_yticks([])
        ax.set_xlim([dt.datetime(2024, 4, 8, 10), dt.datetime(2024, 4, 8, 18)])

    sao_plot.save(fig_file_name)
    sao_plot.close()
    return


## Analyzing the dataset form 2024 Eclipse
stn = "AU930"
folders = [
    f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/{stn}/2024/098/scaled/",
]
# generate_digisonde_pfh_profiles(
#     folders,
#     "tmp/2024_pf.png",
#     fig_title="Digisondes / 08-09 April, 2024",
#     stns=[stn],
# )

# drift_dataset = DvlExtractor.load_DVL_files(folders=[])
drift_dataset = DvlExtractor.load_DVL_files(
    folders=[
        "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08/"
    ],
    n_procs=12,
)
from pynasonde.digisonde.digi_plots import SkySummaryPlots

SkySummaryPlots.plot_dvl_drift_velocities(
    drift_dataset, fname="tmp/extract_2024_04_08_dvl.png", draw_local_time=True
)


stn_info = get_digisonde_info("KR835")
ocl = create_eclipse_path_local(
    drift_dataset.datetime, stn_info["LAT"], stn_info["LONG"]
)
sao_plot = SaoSummaryPlots(
    figsize=(2.5, 7),
    nrows=3,
    fig_title="",
    draw_local_time=True,
)
ax = sao_plot.get_axes(False)
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.errorbar(
    drift_dataset.local_datetime,
    drift_dataset["Vx"],
    yerr=drift_dataset["Vx_err"],
    color="r",
    fmt="o",
    lw=0.8,
    alpha=0.9,
    zorder=3,
    capsize=1,
    capthick=1,
    ms=1,
)
ax.set_ylim(-100, 100)
ax.set_xlim([dt.datetime(2024, 4, 8, 10), dt.datetime(2024, 4, 8, 15)])
axt = ax.twinx()
axt.plot(drift_dataset.local_datetime, 1 - ocl, ls="--", color="k")
axt.set_yticks([])

ax = sao_plot.get_axes(False)
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.errorbar(
    drift_dataset.local_datetime,
    drift_dataset["Vy"],
    yerr=drift_dataset["Vy_err"],
    color="b",
    fmt="o",
    lw=0.8,
    alpha=0.9,
    zorder=3,
    capsize=1,
    capthick=1,
    ms=1,
)
ax.set_ylim(-100, 100)
ax.set_xlim([dt.datetime(2024, 4, 8, 10), dt.datetime(2024, 4, 8, 15)])
axt = ax.twinx()
axt.plot(drift_dataset.local_datetime, 1 - ocl, ls="--", color="k")
axt.set_yticks([])


ax = sao_plot.get_axes(False)
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.errorbar(
    drift_dataset.local_datetime,
    drift_dataset["Vz"],
    yerr=drift_dataset["Vz_err"],
    color="k",
    fmt="o",
    lw=0.8,
    alpha=0.9,
    zorder=3,
    capsize=1,
    capthick=1,
    ms=1,
)
ax.set_ylim(-100, 100)
ax.set_xlim([dt.datetime(2024, 4, 8, 10), dt.datetime(2024, 4, 8, 15)])
axt = ax.twinx()
axt.plot(drift_dataset.local_datetime, 1 - ocl, ls="--", color="k")
axt.set_yticks([])

sao_plot.save("tmp/extract_2024_04_08_dvl_eclipse.png")
sao_plot.close()
