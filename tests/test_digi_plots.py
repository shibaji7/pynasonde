"""Tests covering plotting utilities built on Matplotlib."""

import datetime as dt

import numpy as np
import pandas as pd

from pynasonde.digisonde.digi_plots import SaoSummaryPlots, SkySummaryPlots


def _sample_timeheight_dataframe():
    base = dt.datetime(2024, 4, 8, 16, 0, 0)
    records = []
    for hour in range(2):
        for height in (120, 180, 240):
            records.append(
                {
                    "datetime": base + dt.timedelta(hours=hour),
                    "th": float(height),
                    "pf": float(5 + hour + height / 200.0),
                }
            )
    return pd.DataFrame.from_records(records)


def test_sao_summary_plots_add_ts(tmp_path):
    df = _sample_timeheight_dataframe()

    plotter = SaoSummaryPlots(fig_title="SAO", figsize=(3, 3))
    plotter.add_TS(df, xparam="datetime", yparam="th", zparam="pf", prange=[5, 8])

    outfile = tmp_path / "sao_ts.png"
    plotter.save(outfile)
    plotter.close()

    assert outfile.exists() and outfile.stat().st_size > 0


def test_sao_summary_plots_plot_ts(tmp_path):
    base = dt.datetime(2024, 4, 8, 16, 0, 0)
    df = pd.DataFrame(
        {
            "datetime": [base + dt.timedelta(hours=h) for h in range(4)],
            "hmF2": np.linspace(200, 260, 4),
            "foF2": np.linspace(5, 8, 4),
        }
    )

    plotter = SaoSummaryPlots(fig_title="SAO Scalar", figsize=(3, 3))
    plotter.plot_TS(
        df,
        right_yparams=["hmF2"],
        left_yparams=["foF2"],
        right_ylim=[180, 280],
        left_ylim=[4, 9],
        xlim=[df.datetime.min(), df.datetime.max()],
    )

    outfile = tmp_path / "sao_scalar.png"
    plotter.save(outfile)
    plotter.close()

    assert outfile.exists() and outfile.stat().st_size > 0


def test_sky_summary_plots_skymap(tmp_path):
    df = pd.DataFrame(
        {
            "x_coord": [0.1, 0.5, 0.9],
            "y_coord": [0.2, 0.4, 0.6],
            "spect_dop_freq": [0.01, -0.02, 0.03],
            "zenith_angle": [10, 20, 30],
            "sampl_freq": [3, 3, 3],
            "group_range": [100, 150, 200],
            "gain_ampl": [1, 1, 1],
            "height_spctrum_ampl": [1, 1, 1],
            "max_height_spctrum_ampl": [1, 1, 1],
            "n_sources": [1, 1, 1],
            "height_spctrum_cl_th": [1, 1, 1],
            "spect_line_cl_th": [1, 1, 1],
            "polarization": [0, 1, 0],
        }
    )

    plotter = SkySummaryPlots()
    plotter.plot_skymap(df, zparam="spect_dop_freq", text="Test")
    outfile = tmp_path / "sky.png"
    plotter.save(outfile)
    plotter.close()
    assert outfile.exists() and outfile.stat().st_size > 0
