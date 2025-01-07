import datetime as dt
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setsize(size=8):

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use(["science", "ieee"])
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]
    mpl.rcParams.update(
        {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
    )
    return


class ModelPlots(object):
    """
    A class to plot a summary stack plots using the data obtained from SAO
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
    ):
        self.fig_title = fig_title
        self.nrows = nrows
        self.ncols = ncols
        self.date = date
        self.font_size = font_size
        self.figsize = figsize
        self.date_lims = date_lims
        self.n_sub_plots = 0
        self.fig, self.axes = plt.subplots(
            figsize=(figsize[0] * nrows, figsize[1] * ncols),
            dpi=300,
            nrows=nrows,
            ncols=ncols,
        )  # Size for website
        if type(self.axes) == list or type(self.axes) == np.ndarray:
            self.axes = self.axes.ravel()
        return

    def get_axes(self, del_ticks=True):
        setsize(self.font_size)
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

    def _add_colorbar(
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


class AnalysisPlots(ModelPlots):
    """ """

    def __init__(
        self,
        fig_title: str = "",
        nrows: int = 1,
        ncols: int = 1,
        font_size: float = 10,
        figsize: tuple = (3, 3),
        date: dt.datetime = None,
        date_lims: List[dt.datetime] = [],
    ):
        super().__init__(fig_title, nrows, ncols, font_size, figsize, date, date_lims)
        return

    def plot_ionogram_trace(
        self,
        f_sweep: np.array,
        max_ret_heights: np.array,
        xlabel: str = "Frequency, MHz",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [0, 400],
        xlim: List[float] = [1, 22],
        xticks: List[float] = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
        text: str = None,
        del_ticks: bool = False,
        lcolor: str = "k",
        lw: float = 0.7,
        zorder: int = 2,
        ax=None,
    ):
        setsize(self.font_size)
        ax = ax if ax else self.get_axes(del_ticks=del_ticks)
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylim(ylim)
        ax.set_xlim(np.log10(xlim))
        ax.set_ylabel(
            ylabel,
            fontdict={"size": self.font_size},
        )
        ax.plot(
            np.log10(f_sweep),
            max_ret_heights,
            ls="None",
            zorder=zorder,
            marker="s",
            ms=0.6,
            color=lcolor,
        )
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
                fontdict={"size": self.font_size},
            )
        return ax

    def plot_profile(
        self,
        df: pd.DataFrame,
        f_sweep: float,
        xlabel: str = "Absorption, dB",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [0, 400],
        xlim: List[float] = [0, 3],
        text: str = None,
        del_ticks: bool = False,
        lcolor: str = "k",
        lw: float = 0.7,
        ls: str = "-",
        zorder: int = 2,
        ax=None,
    ):
        setsize(self.font_size)
        ax = ax if ax else self.get_axes(del_ticks=del_ticks)
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        # ax.set_ylim(ylim)
        # ax.set_xlim(xlim)
        ax.set_ylabel(
            ylabel,
            fontdict={"size": self.font_size},
        )
        ax.plot(
            np.log10(df.absorption.cumsum()),
            df.alts,
            ls=ls,
            lw=lw,
            zorder=zorder,
            color=lcolor,
        )
        if text:
            ax.text(
                0.05,
                0.9,
                text,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict={"size": self.font_size},
            )
        return ax
