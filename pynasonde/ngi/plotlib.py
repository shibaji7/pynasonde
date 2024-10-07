import datetime as dt
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import pynasonde.ngi.utils as utils


class Ionogram(object):

    def __init__(
        self,
        dates: dt.datetime = None,
        fig_title: str = "",
        nrows: int = 2,
        ncols: int = 3,
        font_size: float = 10,
    ):
        self.dates = dates
        self.ncols = ncols
        self.nrows = nrows
        self.fig, self.axes = plt.subplots(
            figsize=(6 * nrows, 3 * ncols), dpi=300, nrows=nrows, ncols=ncols
        )  # Size for website
        self.axes = self.axes.ravel()
        self.fig_title = fig_title
        utils.setsize(font_size)
        self._num_subplots_created = 0
        return

    def add_ionogram(
        self,
        frequency: np.array,
        height: np.array,
        value: np.array,
        mode: str = "O",
        xlabel: str = "Frequency, MHz",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [50, 800],
        xlim: List[float] = [1, 22],
        add_cbar: bool = False,
        cbar_label: str = "{}-mode Power, dB",
        cmap: str = "Greens",
        prange: List[float] = [5, 70],
        xticks: List[float] = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
        text: str = None,
        del_ticks: bool = True,
    ) -> None:
        ax = self._add_axis(del_ticks)
        ax.set_xlim(np.log10(xlim))
        ax.set_xlabel(xlabel, fontdict={"size": 12})
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        im = ax.pcolormesh(
            np.log10(frequency),
            height,
            value.T,
            lw=0.01,
            edgecolors="None",
            cmap=cmap,
            vmax=prange[1],
            vmin=prange[0],
            zorder=3,
        )
        if np.logical_not(del_ticks):
            ax.set_xticks(np.log10(xticks))
            ax.set_xticklabels(xticks)
        if text:
            ax.text(
                0.05,
                0.9,
                text,
                ha="left",
                va="center",
                transform=ax.transAxes,
            )
        if add_cbar:
            self._add_colorbar(im, self.fig, ax, label=cbar_label.format(mode))
        return ax

    def _add_axis(self, del_ticks=True):
        ax = self.axes[self._num_subplots_created]
        if del_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if self._num_subplots_created == 0:
            ax.text(
                0.01,
                1.05,
                self.fig_title,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict={"size": 12},
            )
        self._num_subplots_created += 1
        return ax

    def _add_colorbar(self, im, fig, ax, label=""):
        """
        Add a colorbar to the right of an axis.
        """
        pos = ax.get_position()
        cpos = [
            pos.x1 + 0.025,
            pos.y0 + 0.0125,
            0.015,
            pos.height * 0.9,
        ]  # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        cb = fig.colorbar(im, ax=ax, cax=cax)
        cb.set_label(label)
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return
