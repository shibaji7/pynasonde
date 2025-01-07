import datetime as dt

import numpy as np

from pynasonde.model.point import Point

if __name__ == "__main__":
    f_sweep = np.linspace(2, 12, 101)
    p = Point(dt.datetime(2017, 5, 27, 15), 42.6233, -71.4882, np.arange(50, 500))
    p._load_profile_()
    p.calculate_collision_freqs()
    p.calculate_absorptions(f_sweep=f_sweep)
    df_o = p.find_ionogram_trace_max_height(f_sweep=f_sweep)
    # df_x = p.find_ionogram_trace_max_height(f_sweep=f_sweep, mode="X")
    # print(df_x.head(30))
    # print(p.H, p.edens.shape)
    from pynasonde.model.plots import AnalysisPlots

    ap = AnalysisPlots(ncols=2, figsize=(6, 2))
    ax = ap.get_axes(del_ticks=False)
    for f in f_sweep:
        ap.plot_profile(p.get_absoption_profiles(fo=f), f, ax=ax)
    ap.plot_ionogram_trace(
        df_o.f_sweep,
        df_o.max_ret_heights,
        lcolor="r",
        # df.absorption,
        # df.alts,
    )
    # ap.plot_ionogram_trace(df_x.f_sweep, df_x.max_ret_heights, lcolor="g", ax=ax)
    # ax = ap.get_axes(False)
    # for c, fo in zip(["k", "b", "r", "c"], [2, 3, 4, 5]):
    #     df = p.find_ionogram_trace(fo)
    #     ap.plot_ionogram_trace(
    #         p.get_absoption_profiles(fo).ravel(), p.alts,
    #         # df.absorption,
    #         # df.alts,
    #         lcolor=c,
    #         ax=ax,
    #     )

    ap.save("tmp/example_trace.png")
    ap.close()
