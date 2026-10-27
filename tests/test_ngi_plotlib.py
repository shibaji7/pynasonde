"""Tests for the VIPIR NGI ``Ionogram`` plotting wrapper on small synthetic data."""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pynasonde.vipir.ngi.plotlib import COLOR_MAPS, Ionogram

DATES = [dt.datetime(2024, 4, 9, 12, 30)]


@pytest.fixture
def iono():
    fig = Ionogram(dates=DATES, fig_title="TITLE", nrows=2, ncols=2, figsize=(2, 1.5))
    yield fig
    fig.close()


def synthetic_power():
    freq = np.linspace(1.5, 20.0, 40)
    height = np.linspace(60.0, 700.0, 60)
    power = np.full((freq.size, height.size), 10.0)
    power[10:20, 20:25] = 60.0  # a bright trace
    return freq, height, power


def test_constructor_builds_a_grid_of_axes(iono):
    assert iono.axes.shape == (4,) and iono.fig_title == "TITLE"
    single = Ionogram(nrows=1, ncols=1, figsize=(2, 1.5))
    assert not hasattr(single.axes, "ravel")  # a lone Axes, not an array
    single.close()


def test_colormap_runs_black_to_pale_yellow():
    cm = COLOR_MAPS.Inferno
    assert cm(0.0)[:3] == (0.0, 0.0, 0.0)
    assert cm(1.0)[0] == pytest.approx(1.0)


def test_add_ionogram_sets_axes_and_draws_the_mesh(iono):
    f, h, p = synthetic_power()
    ax = iono.add_ionogram(f, h, p, text="O-mode", ylim=[60, 700], xlim=[1.5, 20])
    assert ax.get_xlim() == pytest.approx(tuple(np.log10([1.5, 20])))
    assert ax.get_ylim() == (60, 700)
    assert ax.get_xlabel() == "Frequency, MHz"
    mesh = ax.collections[0]
    assert mesh.get_clim() == (5, 70)
    assert any(t.get_text() == "O-mode" for t in ax.texts)
    # the first panel also carries the figure title
    assert any(t.get_text() == "TITLE" for t in ax.texts)
    assert ax.get_xticks().size == 0  # ticks are stripped by default


def test_add_ionogram_with_ticks_and_colorbar(iono):
    f, h, p = synthetic_power()
    ax = iono.add_ionogram(f, h, p, del_ticks=False, add_cbar=True, mode="X")
    assert [t.get_text() for t in ax.get_xticklabels()][:3] == ["1.5", "2.0", "3.0"]
    labels = [a.get_ylabel() for a in iono.fig.axes]
    assert "X-mode Power, dB" in labels


def test_second_panel_has_no_title(iono):
    f, h, p = synthetic_power()
    iono.add_ionogram(f, h, p)
    ax2 = iono.add_ionogram(f, h, p)
    assert not any(t.get_text() == "TITLE" for t in ax2.texts)


def test_add_ionogram_traces_default_and_custom_text(iono):
    ax = iono.add_ionogram_traces(np.array([2.0, 4.0]), np.array([200.0, 300.0]))
    assert ax.texts[-1].get_text() == "O-mode/20240409 1230 UT"
    ax2 = iono.add_ionogram_traces(
        np.array([2.0]), np.array([250.0]), text="mine", del_ticks=False, ax=ax
    )
    assert ax2 is ax and ax.texts[-1].get_text() == "mine"
    assert len(ax.lines) == 2


def interval_frame(n_t=6, n_r=8):
    t0 = pd.Timestamp("2024-04-09 00:00")
    rows = []
    for ti in range(n_t):
        for ri in range(n_r):
            rows.append(
                dict(
                    time=t0 + pd.Timedelta(minutes=10 * ti),
                    range=100.0 + 20.0 * ri,
                    O_mode_power=30.0 + 3 * ri,
                    O_mode_noise=10.0,
                )
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("kind", ["pcolormesh", "contourf"])
def test_add_interval_plots(iono, kind):
    df = interval_frame()
    ax = iono.add_interval_plots(
        df, kind=kind, ylim=[100, 240], prange=[20, 60], add_cbar=True
    )
    assert ax.get_ylim() == (100, 240)
    assert ax.get_xlabel() == "Time, UT"
    assert len(ax.collections) >= 1
    assert "O-mode Power, dB" in [a.get_ylabel() for a in iono.fig.axes]


def test_interval_plot_masks_power_below_the_noise_floor(iono):
    df = interval_frame()
    df["O_mode_power"] = 5.0  # everything under 1.2 x noise
    iono.add_interval_plots(
        df, add_cbar=False, xdate_lims=[df.time.min(), df.time.max()]
    )
    assert df["O_mode_power"].isna().all()


def test_add_time_series(iono):
    t = pd.date_range("2024-04-09", periods=5, freq="h")
    ax = iono.add_TS(t, np.array([3.0, 4.0, 5.0, 4.5, 3.5]))
    assert ax.get_ylim() == (3.0, 5.0)
    assert ax.get_ylabel() == r"$foF_2$, MHz"
    ax2 = iono.add_TS(t, np.arange(5.0), ylim=[0, 10], xlim=[t[0], t[-1]])
    assert ax2.get_ylim() == (0, 10)
    assert ax2.lines[0].get_marker() == "."


def test_save_writes_a_png(iono, tmp_path):
    f, h, p = synthetic_power()
    iono.add_ionogram(f, h, p)
    out = tmp_path / "ion.png"
    iono.save(out)
    assert out.stat().st_size > 1000
