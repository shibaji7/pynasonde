from pynasonde.sao_extractor import SaoExtractor
from pynasonde.sao_summary_plots import SaoSummaryPlots
import datetime as dt
import numpy as np

# Example Usage
if __name__ == "__main__":
    filename = "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/*.SAO"
    import glob

    files = glob.glob(filename)
    files.sort()
    for i, f in enumerate(files):
        extractor = SaoExtractor(f)
        extractor.extract()
        print(extractor.SAOstruct["Scaled"])
        if i == 3:
            break
    # coll1 = SaoExtractor.load_SAO_files(
    #     folders=["tmp/SKYWAVE_DPS4D_2023_10_14"],
    #     func_name="height_profile",
    # )
    # coll1.ed = coll1.ed / 1e6
    # sao_plot = SaoSummaryPlots(
    #     figsize=(6, 3), fig_title="KR835/13-14 Oct, 2023", draw_local_time=True
    # )
    # sao_plot.add_TS(
    #     coll1,
    #     zparam="ed",
    #     prange=[0, 1],
    #     zparam_lim=10,
    #     cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
    # )
    # sao_plot.save("tmp/example_pf.png")
    # sao_plot.close()
    # coll2 = SaoExtractor.load_SAO_files(
    #     folders=["/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2023_10_14/"], func_name="scaled"
    # )
    # print(coll2[["datetime", "foEs"]], coll2.head())
    # coll2 = coll2[(coll2.datetime >= dt.datetime(2023, 10, 14, 14))
    #               & (coll2.datetime >= dt.datetime(2023, 10, 14, 18))]
    # print(coll2[["datetime", "foEs"]])
    # sao_plot = SaoSummaryPlots(
    #     figsize=(6, 3), fig_title="KR835/13-14 Oct, 2023", draw_local_time=True
    # )
    # sao_plot.plot_TS(coll2, left_yparams=["foF1"], left_ylim=[1, 15])
    # sao_plot.save("tmp/example_ts.png")
    # sao_plot.close()
    # SaoSummaryPlots.plot_isodensity_contours(
    #     coll1,
    #     xlim=[dt.datetime(2023, 10, 13, 12), dt.datetime(2023, 10, 14)],
    #     fname="tmp/example_id.png",
    # )
    # extractor = SaoExtractor("tmp/20250527/KW009_2025147120000_SAO.XML", True, True)
    # extractor.extract_xml()
    # print(extractor.get_scaled_datasets_xml())
    # print(extractor.get_height_profile_xml())
    # # sao_plot = SaoSummaryPlots(
    # #     figsize=(3, 3), fig_title="kw009/27 May, 2025", draw_local_time=False
    # # )
    # # sao_plot.save("tmp/kw_ion.png")
    # # sao_plot.close()
    # col = SaoExtractor.load_XML_files(["tmp/20250527/"], func_name="scaled")
    # sao_plot = SaoSummaryPlots(
    #     figsize=(6, 3), fig_title="kw009/27 May, 2025", draw_local_time=False
    # )
    # col["foFs"], col["hmFs"] = (
    #     np.nanmax([col.foF1.tolist(), col.foF2.tolist()], axis=0),
    #     np.nanmax([col.hmF1.tolist(), col.hmF2.tolist()], axis=0),
    # )
    # print(col.head())
    # sao_plot.plot_TS(
    #     col,
    #     left_yparams=["foEs"],
    #     right_yparams=["h`Es"],
    #     right_ylim=[80, 150],
    #     left_ylim=[0, 6],
    #     seed=6,
    # )
    # sao_plot.save("tmp/example_ts.png")
    # #
    # # print(col.head())
    # # sao_plot.add_TS(
    # #     col,
    # #     zparam="pf",
    # #     prange=[2, 5],
    # #     zparam_lim=np.nan,
    # #     cbar_label=r"$f_0$, MHz",
    # #     scatter_ms=40,
    # #     plot_type="scatter",
    # #     ylim=[90, 150],
    # # )
    # # sao_plot.save("tmp/example_pf.png")
    # sao_plot.close()
