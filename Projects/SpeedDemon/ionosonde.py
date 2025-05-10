import datetime as dt
import shutil

from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.digisonde.sao import SaoExtractor


def generate_digisonde_pfh_profiles(
    folders,
    fig_file_name,
    fig_title="",
    draw_local_time=False,
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
        df.ed = df.ed / 1e6
        dfsc = dfscs[i]
        ax, _ = sao_plot.add_TS(
            df,
            zparam="ed",
            prange=[0, 1],
            zparam_lim=10,
            cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
            plot_type="scatter",
            title="Stn Code: " + stns[i],
            add_cbar=True,
            xlabel="Time, UT" if i == N - 1 else "",
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

    sao_plot.save(fig_file_name)
    sao_plot.close()
    return


## Analyzing the dataset form Speed Deamon 2022
for doy in range(233, 238, 1):
    stn = "WP937"
    date = dt.datetime(2022, 1, 1) + dt.timedelta(days=doy - 1)
    fig_file_name = f"../../tmp/SAO.{stn}.2022.doy-{doy}.png"
    fig_title = f"Speed Demon / {date.strftime('%Y-%m-%d')}"

    shutil.rmtree(f"/tmp/{doy}/", ignore_errors=True)
    shutil.copytree(
        f"/media/chakras4/ERAU/SpeedDemon/WP937/individual/2022/{doy}/scaled/",
        f"/tmp/{doy}/scaled/",
    )
    generate_digisonde_pfh_profiles(
        folders=[f"/tmp/{doy}/scaled/"],
        fig_file_name=fig_file_name,
        fig_title=fig_title,
        draw_local_time=False,
        stns=[stn],
    )
    shutil.rmtree(f"/tmp/{doy}/", ignore_errors=True)
