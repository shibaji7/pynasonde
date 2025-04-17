import datetime as dt
from typing import List

import matplotlib.pyplot as plt
import numpy as np

DATE_FORMAT: str = r"$%H^{%M}$"

import pynasonde.digisonde.digi_utils as utils


class RawPlots(object):
    """
    A class to plot a summary stack plots using the data obtained from .bin
    """

    def __init__(
        self,
        fig_title: str = "",
        nrows: int = 1,
        ncols: int = 1,
        font_size: float = 10,
        figsize: tuple = (3, 3),
        date: dt.datetime = None,
        date_lims: List[dt.datetime] = [],
        subplot_kw: dict = None,
        draw_local_time: bool = False,
    ):
        self.fig_title = fig_title
        self.nrows = nrows
        self.ncols = ncols
        self.date = date
        self.font_size = font_size
        self.figsize = figsize
        self.date_lims = date_lims
        self.subplot_kw = subplot_kw
        self.n_sub_plots = 0
        self.draw_local_time = draw_local_time
        self.fig, self.axes = plt.subplots(
            figsize=(figsize[0] * nrows, figsize[1] * ncols),
            dpi=300,
            nrows=nrows,
            ncols=ncols,
            subplot_kw=self.subplot_kw,
        )  # Size for website
        if type(self.axes) == list or type(self.axes) == np.ndarray:
            self.axes = self.axes.ravel()
        return

    def get_axes(self, del_ticks=True):
        utils.setsize(self.font_size)
        ax = (
            self.axes[self.n_sub_plots]
            if type(self.axes) == np.ndarray or type(self.axes) == list
            else self.axes
        )
        if del_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if self.n_sub_plots == 0:
            ax.text(
                0.01,
                1.05,
                self.fig_title,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict={"size": self.font_size},
            )
        self.n_sub_plots += 1
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

    def add_colorbar(
        self,
        im,
        fig,
        ax,
        label: str = "",
        mpos: List[float] = [0.025, 0.0125, 0.015, 0.5],
    ):
        """
        Add a colorbar to the right of an axis.
        """
        pos = ax.get_position()
        cpos = [
            pos.x1 + mpos[0],
            pos.y0 + mpos[1],
            mpos[2],
            pos.height * mpos[3],
        ]  # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        cb = fig.colorbar(im, ax=ax, cax=cax)
        cb.set_label(label)
        return

    def add_pcolor(
        self,
        ax,
        X: np.array,
        Y: np.array,
        zz: np.array,
        add_cbar: bool = True,
        label: str = "",
        cmap: str = "jet",
        prange: List[float] = [1, 15],
        ylabel: str = "",
        xlabel: str = "",
        ylim: List = [],
        xlim: List = [],
        cbar_label: str = "",
    ):
        im = ax.pcolormesh(
            X,
            Y,
            zz.T,
            lw=0.01,
            edgecolors="None",
            cmap=cmap,
            # vmax=prange[1],
            # vmin=prange[0],
            zorder=3,
        )
        if add_cbar:
            self.add_colorbar(im, self.fig, ax, label=cbar_label)
        return


class AFRLPlots(RawPlots):

    def __init__(
        self,
        fig_title: str = "",
        nrows: int = 1,
        ncols: int = 1,
        font_size: float = 10,
        figsize: tuple = (3, 3),
        date: dt.datetime = None,
        date_lims: List[dt.datetime] = [],
        subplot_kw: dict = None,
        draw_local_time: bool = False,
    ):
        super().__init__(
            fig_title,
            nrows,
            ncols,
            font_size,
            figsize,
            date,
            date_lims,
            subplot_kw,
            draw_local_time,
        )
        return

    def draw_psd(
        self,
        f: np.array,
        psd: np.array,
        color: str = "r",
        ylabel: str = "Power Spectral Density",
        xlabel: str = "Frequency, MHz",
        ylim: List = [],
        xlim: List = [],
        ls: str = "-",
        lw: float = 0.6,
        alpha: float = 0.7,
        title: str = None,
        ax=None,
    ):
        utils.setsize(self.font_size)
        ax = ax if ax else self.get_axes(del_ticks=False)
        xlim = xlim if len(xlim) == 2 else [f.min(), f.max()]
        ylim = ylim if len(ylim) == 2 else [psd.min(), psd.max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        if title:
            ax.text(0.95, 1.05, title, ha="right", va="center", transform=ax.transAxes)
        ax.plot(
            f,
            psd,
            ls=ls,
            color=color,
            lw=lw,
            alpha=alpha,
        )
        ax.set_yscale("log")
        return

    def draw_psd_scan(
        self,
        f: np.array,
        t_spec: np.array,
        psd: np.array,
        color: str = "r",
        ylabel: str = "Time, us",
        xlabel: str = "Frequency, MHz",
        ylim: List = [],
        xlim: List = [],
        alpha: float = 0.7,
        title: str = None,
        ax=None,
    ):
        utils.setsize(self.font_size)
        ax = ax if ax else self.get_axes(del_ticks=False)
        xlim = xlim if len(xlim) == 2 else [f.min(), f.max()]
        ylim = ylim if len(ylim) == 2 else [t_spec.min(), t_spec.max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        if title:
            ax.text(0.95, 1.05, title, ha="right", va="center", transform=ax.transAxes)
        # self.add_pcolor(
        #     ax,
        #     f,
        #     t_spec,
        #     psd,
        # )
        im = ax.pcolormesh(
            f,
            t_spec,
            psd.T,
            lw=0.01,
            edgecolors="None",
            cmap="jet",
            # vmax=prange[1],
            # vmin=prange[0],
            zorder=3,
        )
        # if add_cbar:
        self.add_colorbar(im, self.fig, ax, label="")
        print(t_spec.min(), t_spec.max())
        return
