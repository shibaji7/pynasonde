# from pynasonde.digisonde.dvl import DvlExtractor
from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.sao import SaoExtractor


def generate_digisonde_pfh_profiles(
    folders,
    func_name,
    fig_file_name,
    fig_title="",
    draw_local_time=True,
    stns=[],
):
    dfs = [
        SaoExtractor.load_SAO_files(folders=[folder], func_name=func_name)
        for folder in folders
    ]
    N = len(folders)
    sao_plot = SaoSummaryPlots(
        font_size=15,
        figsize=(4, 4 * N),
        nrows=N,
        fig_title=fig_title,
        draw_local_time=draw_local_time,
    )

    for i in range(N):
        df = dfs[i]
        df.ed = df.ed / 1e6
        ax, _ = sao_plot.add_TS(
            df,
            zparam="ed",
            prange=[0, 1],
            zparam_lim=10,
            cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
            plot_type="scatter",
            title="Stn Code: " + stns[i],
            add_cbar=i == 2,
            xlabel="Time, UT" if i == 2 else "",
        )
    sao_plot.save(fig_file_name)
    sao_plot.close()
    return


## Analyzing the dataset form 2023 Eclipse
folders = [
    f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/AU930/2024/098/scaled/",
    f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/AL945/2024/098/scaled/",
    f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/WP937/2024/098/scaled/",
]
func_name = "height_profile"
generate_digisonde_pfh_profiles(
    folders,
    func_name,
    "tmp/2024_pf.png",
    fig_title="Digisondes / 08-09 April, 2024",
    stns=["AU930", "AL945", "WP937"],
)

# drift_dataset = DvlExtractor.load_DVL_files(folders=[])
