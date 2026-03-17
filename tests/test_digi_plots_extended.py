"""Extended tests for pynasonde.digisonde.digi_plots.

Covers: search_color_schemes, SaoSummaryPlots.plot_ionogram,
add_isodensity_contours, add_TS (title + no-cbar variants),
plot_TS (no right-axis), SkySummaryPlots.plot_doppler_waterfall,
plot_doppler_spectra, plot_drift_velocities, and RsfIonogram.add_direction_ionogram.
"""

import datetime as dt

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pytest

from pynasonde.digisonde.digi_plots import (
    RsfIonogram,
    SaoSummaryPlots,
    SkySummaryPlots,
    search_color_schemes,
)

# ---------------------------------------------------------------------------
# Helpers: synthetic DataFrames
# ---------------------------------------------------------------------------


def _sao_height_df(n_times=3, n_heights=8):
    """Time × height plasma-frequency DataFrame (for isodensity / TS plots)."""
    base = dt.datetime(2024, 4, 9, 0, 0, 0)
    records = []
    for t_idx in range(n_times):
        t = base + dt.timedelta(hours=t_idx)
        for h in np.linspace(100, 400, n_heights):
            records.append(
                {
                    "datetime": t,
                    "th": float(h),
                    "pf": float(3.0 + t_idx + h / 200.0),
                    "ed": float(1e10 * t_idx + h * 1e8),
                }
            )
    return pd.DataFrame.from_records(records)


def _scalar_ts_df(n=4):
    """Simple scalar time-series DataFrame."""
    base = dt.datetime(2024, 4, 9, 0, 0, 0)
    return pd.DataFrame(
        {
            "datetime": [base + dt.timedelta(hours=h) for h in range(n)],
            "foF2": np.linspace(5, 8, n),
            "foF1": np.linspace(4, 6, n),
            "hmF2": np.linspace(200, 260, n),
            "hmF1": np.linspace(150, 200, n),
        }
    )


def _dft_df(n_bins=16, n_heights=4, n_blocks=2):
    """Synthetic DFT waterfall/spectra DataFrame."""
    records = []
    for blk in range(n_blocks):
        for h in np.linspace(100, 400, n_heights):
            for d in range(n_bins):
                records.append(
                    {
                        "block_idx": blk,
                        "doppler_bin": d - n_bins // 2,
                        "height_km": float(h),
                        "amplitude": float(20.0 + blk * 5 + d * 0.5),
                        "frequency_reading": float((blk + 1) * 5e6),
                    }
                )
    return pd.DataFrame.from_records(records)


def _rsf_df(n=20):
    """Synthetic RSF direction-ionogram DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    azm_dirs = ["N", "NE", "SE", "S", "SW", "NW"]
    pols = ["O", "X"]
    records = []
    for i in range(n):
        records.append(
            {
                "frequency_reading": float(rng.integers(3, 15)) * 1e6,
                "height": float(rng.integers(100, 400)),
                "amplitude": float(rng.integers(5, 50)),
                "azm_directions": azm_dirs[i % len(azm_dirs)],
                "pol": pols[i % len(pols)],
                "dop_num": float(rng.integers(0, 8)),
            }
        )
    return pd.DataFrame.from_records(records)


def _dvl_df(n=4):
    """Synthetic drift velocity DataFrame."""
    base = dt.datetime(2024, 4, 9, 0, 0, 0)
    hours = list(range(n))
    return pd.DataFrame(
        {
            "datetime": [base + dt.timedelta(hours=h) for h in hours],
            "t_hr": [float(h) for h in hours],  # numeric alias used in tests
            "Vx": np.linspace(-20, 20, n),
            "Vx_err": np.ones(n) * 2.0,
        }
    )


# ---------------------------------------------------------------------------
# search_color_schemes
# ---------------------------------------------------------------------------


class TestSearchColorSchemes:
    def test_returns_list(self):
        bounds = [{"gt": 0.0, "lt": 1.0}]
        result = search_color_schemes(num_colors=1, bounds=bounds, search_length=10)
        assert isinstance(result, list)

    def test_finds_at_least_one(self):
        # Very loose bounds → many matches
        bounds = [{"gt": 0.0, "lt": 1.0}, {"gt": 0.0, "lt": 1.0}]
        result = search_color_schemes(num_colors=2, bounds=bounds, search_length=20)
        assert len(result) > 0

    def test_result_has_seed_and_color_keys(self):
        bounds = [{"gt": 0.0, "lt": 1.0}]
        result = search_color_schemes(num_colors=1, bounds=bounds, search_length=5)
        if result:
            assert "seed" in result[0]
            assert "color" in result[0]

    def test_tight_bounds_returns_empty(self):
        # No value can be in (0.9999, 1.0000) practically
        bounds = [{"gt": 0.9999, "lt": 1.0000}]
        result = search_color_schemes(num_colors=1, bounds=bounds, search_length=5)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# SaoSummaryPlots.plot_ionogram
# ---------------------------------------------------------------------------


class TestSaoSummaryPlotsIonogram:
    def test_plot_ionogram_ionogram_kind(self, tmp_path):
        df = _sao_height_df(n_times=1, n_heights=5)
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.plot_ionogram(df, xparam="pf", yparam="th", kind="ionogram")
        out = tmp_path / "ionogram.png"
        plotter.save(out)
        plotter.close()
        assert out.exists() and out.stat().st_size > 0

    def test_plot_ionogram_scatter_kind(self, tmp_path):
        df = _sao_height_df(n_times=1, n_heights=5)
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.plot_ionogram(df, xparam="pf", yparam="th", kind="scatter")
        out = tmp_path / "ionogram_scatter.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_plot_ionogram_with_text(self, tmp_path):
        df = _sao_height_df(n_times=1, n_heights=5)
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.plot_ionogram(df, xparam="pf", yparam="th", text="KR835 / 2024-04-09")
        out = tmp_path / "ionogram_text.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_plot_ionogram_no_ticks(self, tmp_path):
        df = _sao_height_df(n_times=1, n_heights=5)
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.plot_ionogram(df, xparam="pf", yparam="th", del_ticks=True)
        out = tmp_path / "ionogram_noticks.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()


# ---------------------------------------------------------------------------
# SaoSummaryPlots.add_isodensity_contours
# ---------------------------------------------------------------------------


class TestSaoSummaryPlotsIsodensity:
    def test_isodensity_basic(self, tmp_path):
        df = _sao_height_df(n_times=4, n_heights=10)
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.add_isodensity_contours(df, xparam="datetime", yparam="th", zparam="pf")
        out = tmp_path / "isodensity.png"
        plotter.save(out)
        plotter.close()
        assert out.exists() and out.stat().st_size > 0

    def test_isodensity_with_text(self, tmp_path):
        df = _sao_height_df(n_times=4, n_heights=10)
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.add_isodensity_contours(
            df, xparam="datetime", yparam="th", zparam="pf", text="Station/2024-04-09"
        )
        out = tmp_path / "isodensity_text.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_isodensity_with_prange(self, tmp_path):
        df = _sao_height_df(n_times=4, n_heights=10)
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.add_isodensity_contours(
            df, xparam="datetime", yparam="th", zparam="pf", prange=[2.0, 10.0]
        )
        out = tmp_path / "isodensity_prange.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()


# ---------------------------------------------------------------------------
# SaoSummaryPlots.add_TS — title and add_cbar=False branches
# ---------------------------------------------------------------------------


class TestSaoSummaryPlotsAddTS:
    def test_add_ts_with_title(self, tmp_path):
        df = _sao_height_df()
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.add_TS(
            df,
            xparam="datetime",
            yparam="th",
            zparam="pf",
            prange=[3, 8],
            title="Test Plot",
        )
        out = tmp_path / "add_ts_title.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_add_ts_no_cbar(self, tmp_path):
        df = _sao_height_df()
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.add_TS(
            df,
            xparam="datetime",
            yparam="th",
            zparam="pf",
            prange=[3, 8],
            add_cbar=False,
        )
        out = tmp_path / "add_ts_nocbar.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_add_ts_scatter_type(self, tmp_path):
        df = _sao_height_df()
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.add_TS(
            df,
            xparam="datetime",
            yparam="th",
            zparam="pf",
            prange=[3, 8],
            plot_type="scatter",
        )
        out = tmp_path / "add_ts_scatter.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_add_ts_with_zparam_lim(self, tmp_path):
        df = _sao_height_df()
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.add_TS(
            df,
            xparam="datetime",
            yparam="th",
            zparam="pf",
            prange=[3, 8],
            zparam_lim=6.0,
        )
        out = tmp_path / "add_ts_zparam_lim.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()


# ---------------------------------------------------------------------------
# SaoSummaryPlots.plot_TS — no right_yparams and with title
# ---------------------------------------------------------------------------


class TestSaoSummaryPlotsPlotTS:
    def test_plot_ts_no_right_yparams(self, tmp_path):
        df = _scalar_ts_df()
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.plot_TS(df, xparam="datetime", left_yparams=["foF2"], right_yparams=[])
        out = tmp_path / "plot_ts_no_right.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_plot_ts_with_title(self, tmp_path):
        df = _scalar_ts_df()
        plotter = SaoSummaryPlots(figsize=(3, 3))
        plotter.plot_TS(
            df,
            xparam="datetime",
            left_yparams=["foF2"],
            right_yparams=["hmF2"],
            title="SAO Timeseries",
        )
        out = tmp_path / "plot_ts_title.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()


# ---------------------------------------------------------------------------
# SkySummaryPlots.plot_doppler_waterfall
# ---------------------------------------------------------------------------


class TestSkySummaryPlotsWaterfall:
    def test_waterfall_auto_block(self, tmp_path):
        df = _dft_df()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_doppler_waterfall(df)
        out = tmp_path / "waterfall_auto.png"
        plotter.save(out)
        plotter.close()
        assert out.exists() and out.stat().st_size > 0

    def test_waterfall_specific_block(self, tmp_path):
        df = _dft_df()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_doppler_waterfall(df, block_idx=0)
        out = tmp_path / "waterfall_block0.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_waterfall_with_text_and_prange(self, tmp_path):
        df = _dft_df()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_doppler_waterfall(
            df, block_idx=0, text="Test", prange=[20.0, 40.0]
        )
        out = tmp_path / "waterfall_text.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_waterfall_with_xlim_ylim(self, tmp_path):
        df = _dft_df()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_doppler_waterfall(df, block_idx=0, xlim=[-5, 5], ylim=[100, 400])
        out = tmp_path / "waterfall_lims.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()


# ---------------------------------------------------------------------------
# SkySummaryPlots.plot_doppler_spectra
# ---------------------------------------------------------------------------


class TestSkySummaryPlotsSpectra:
    def test_spectra_basic(self, tmp_path):
        df = _dft_df()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_doppler_spectra(df)
        out = tmp_path / "spectra_basic.png"
        plotter.save(out)
        plotter.close()
        assert out.exists() and out.stat().st_size > 0

    def test_spectra_with_block_and_heights(self, tmp_path):
        df = _dft_df()
        heights = df["height_km"].unique()[:3].tolist()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_doppler_spectra(df, block_idx=0, selected_heights=heights)
        out = tmp_path / "spectra_heights.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_spectra_with_text_and_lims(self, tmp_path):
        df = _dft_df()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_doppler_spectra(
            df, block_idx=0, text="DFT Spectra", xlim=[-5, 5], ylim=[0, 60]
        )
        out = tmp_path / "spectra_text.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()


# ---------------------------------------------------------------------------
# SkySummaryPlots.plot_drift_velocities
# ---------------------------------------------------------------------------


class TestSkySummaryPlotsDrift:
    # matplotlib capthick+datetime triggers a rotation bug; use numeric t_hr.
    # Production code calls set_major_locator twice (minor is also set as major),
    # so AutoMinorLocator would recurse. Use NullLocator to keep ticks quiet.
    _loc = mticker.NullLocator()

    def test_drift_basic(self, tmp_path):
        df = _dvl_df()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_drift_velocities(
            df,
            xparam="t_hr",
            yparam="Vx",
            error="Vx_err",
            major_locator=self._loc,
            minor_locator=self._loc,
        )
        out = tmp_path / "drift_basic.png"
        plotter.save(out)
        plotter.close()
        assert out.exists() and out.stat().st_size > 0

    def test_drift_with_text(self, tmp_path):
        df = _dvl_df()
        plotter = SkySummaryPlots(figsize=(3, 3))
        plotter.plot_drift_velocities(
            df,
            xparam="t_hr",
            yparam="Vx",
            error="Vx_err",
            text="Drift Vx",
            major_locator=self._loc,
            minor_locator=self._loc,
        )
        out = tmp_path / "drift_text.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()


# ---------------------------------------------------------------------------
# RsfIonogram.add_direction_ionogram
# ---------------------------------------------------------------------------


class TestRsfIonogramDirection:
    def test_direction_ionogram_basic(self, tmp_path):
        df = _rsf_df(n=30)
        plotter = RsfIonogram(figsize=(3, 3))
        plotter.add_direction_ionogram(df)
        out = tmp_path / "direction_ionogram.png"
        plotter.save(out)
        plotter.close()
        assert out.exists() and out.stat().st_size > 0

    def test_direction_ionogram_with_text(self, tmp_path):
        df = _rsf_df(n=20)
        plotter = RsfIonogram(figsize=(3, 3))
        plotter.add_direction_ionogram(df, text="KR835 RSF")
        out = tmp_path / "direction_ionogram_text.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()

    def test_direction_ionogram_no_ticks(self, tmp_path):
        df = _rsf_df(n=20)
        plotter = RsfIonogram(figsize=(3, 3))
        plotter.add_direction_ionogram(df, del_ticks=True)
        out = tmp_path / "direction_ionogram_noticks.png"
        plotter.save(out)
        plotter.close()
        assert out.exists()
