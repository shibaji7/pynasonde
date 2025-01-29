# from pynasonde.digisonde.dvl import DvlExtractor
from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.sao import SaoExtractor


def generate_digisonde_pfh_profiles(
    folders, func_name, fig_file_name, fig_title="", draw_local_time=True
):
    df = SaoExtractor.load_SAO_files(folders=folders, func_name=func_name)
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
    )
    sao_plot.save(fig_file_name)
    sao_plot.close()
    return


## Analyzing the dataset form 2023 Eclipse
folders = [
    f"/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/request/Eclipse/AU930/2023/286/scaled/"
]
func_name = "height_profile"
generate_digisonde_pfh_profiles(
    folders, func_name, "tmp/2023_AU930_pf.png", fig_title="AU930/13-14 Oct, 2023"
)

# drift_dataset = DvlExtractor.load_DVL_files(folders=[])
