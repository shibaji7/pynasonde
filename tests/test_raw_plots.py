"""Smoke tests for raw plotting helpers."""

import numpy as np

from pynasonde.digisonde.raw.raw_plots import RawPlots


def test_raw_plots_basic(tmp_path):
    plotter = RawPlots(fig_title="Raw", figsize=(2, 2))
    ax = plotter.get_axes(del_ticks=False)
    im = ax.imshow(np.arange(9).reshape(3, 3))
    # plotter._add_colorbar(im, plotter.fig, ax, label="Test")
    outfile = tmp_path / "raw.png"
    plotter.save(outfile)
    plotter.close()
    assert outfile.exists() and outfile.stat().st_size > 0
