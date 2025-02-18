import datetime as dt
import sys

import matplotlib.dates as mdates
import numpy as np

from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.dvl import DvlExtractor
from pynasonde.digisonde.sao import SaoExtractor

sys.path.append("Projects/eclipse/")
import utils


def generate_digisonde_pfh_profiles(
    folders, func_name, fig_file_name, fig_title="", draw_local_time=True
):
    df = SaoExtractor.load_SAO_files(
        folders=folders,
        func_name=func_name,
        n_procs=12,
    )
    df.ed = df.ed / 1e6
    sao_plot = SaoSummaryPlots(
        figsize=(6, 3), fig_title=fig_title, draw_local_time=draw_local_time
    )
    sao_plot.add_TS(
        df,
        zparam="ed",
        prange=[0, 1],
        zparam_lim=10,
        cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
        plot_type="scatter",
    )
    obs = utils.create_eclipse_path_local(
        df.datetime, df.lat.tolist()[0], df.lon.tolist()[0]
    )
    sao_plot.save(fig_file_name)
    sao_plot.close()
    return


## Analyzing the dataset form 2023 Eclipse
folders = [
    "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/"
]
func_name = "height_profile"
generate_digisonde_pfh_profiles(
    folders,
    func_name,
    "tmp/2023_Oct_14_KR835_pf.png",
    fig_title="AU930/13-14 Oct, 2023",
)

# ddf = DvlExtractor.load_DVL_files(
#     folders=[
#         "/media/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/"
#     ],
#     n_procs=12,
# )
# obs = utils.create_eclipse_path_local(
#     ddf.datetime,
#     ddf.lat.tolist()[0],
#     ddf.lon.tolist()[0]
# )
# from pynasonde.digisonde.digi_plots import SkySummaryPlots

# dvlplot = SkySummaryPlots.plot_dvl_drift_velocities(
#     ddf, fname=None, draw_local_time=True
# )

# ax = dvlplot.axes[0]
# axt = ax.twinx()
# axt.plot(
#     ddf.local_datetime,
#     1-obs, ls="--", lw=0.9,
#     color="k"
# )
# axt.set_ylabel("Obscuration")
# axt.set_ylim(0, 1)
# ax.set_xlim(
#     [
#         dt.datetime(2023,10,14,8),
#         dt.datetime(2023,10,14,16)
#     ]
# )
# ax.xaxis.set_major_locator(mdates.HourLocator())

# ax = dvlplot.axes[1]
# axt = ax.twinx()
# axt.scatter(
#     ddf.local_datetime,
#     0.5*(ddf.Hb+ddf.Ht),
#     marker="D", s=3,
#     color="m"
# )
# axt.set_ylabel("Virtual Height, km", fontdict={"color":"m"})
# axt.set_ylim(250, 500)
# ax.set_xlim(
#     [
#         dt.datetime(2023,10,14,8),
#         dt.datetime(2023,10,14,16)
#     ]
# )
# ax.xaxis.set_major_locator(mdates.HourLocator())

# ax = dvlplot.axes[2]
# ax.set_ylim(-20,20)
# ax = dvlplot.axes[2]
# axt = ax.twinx()
# axt.plot(
#     ddf.local_datetime,
#     1-obs, ls="--", lw=0.9,
#     color="k"
# )
# axt.set_ylabel("Obscuration")
# axt.set_ylim(0, 1)
# ax.set_xlim(
#     [
#         dt.datetime(2023,10,14,8),
#         dt.datetime(2023,10,14,16)
#     ]
# )
# ax.xaxis.set_major_locator(mdates.HourLocator())

# dvlplot.save("tmp/2023_dvl.png")
# dvlplot.close()
