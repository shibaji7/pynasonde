import datetime as dt

from pynasonde.digisonde.digi_plots import SaoSummaryPlots, search_color_schemes
from pynasonde.digisonde.parsers.sao import SaoExtractor

if __name__ == "__main__":
    for d in range(14, 15):
        date = dt.datetime(2025, 5, 19) + dt.timedelta(days=d)
        col = SaoExtractor.load_XML_files(
            [f"tmp/Digisonde/{date.strftime('%Y%m%d')}"], func_name="scaled"
        )
        sao_plot = SaoSummaryPlots(
            figsize=(6, 3),
            fig_title=f"kw009/{date.strftime('%b %d, %Y')}",
            draw_local_time=False,
            nrows=2,
        )
        sao_plot.plot_TS(
            col,
            left_yparams=["foEs"],
            right_yparams=["h`Es"],
            right_ylim=[80, 150],
            left_ylim=[0, 6],
            seed=4721,
            ylabels=["Frequencies, MHz", "Virtual Height, km"],
            xlabel="",
        )
        sao_plot.plot_TS(
            col,
            left_yparams=["foF1"],
            right_yparams=["hmF1"],
            right_ylim=[80, 600],
            left_ylim=[0, 10],
            seed=4721,
        )
        sao_plot.save(f"tmp/seed/ts_{date.strftime('%Y%m%d')}.png")
