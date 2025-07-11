import datetime as dt
from types import SimpleNamespace
from typing import List, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter

import pynasonde.vipir.ngi.utils as utils

COLOR_MAPS = SimpleNamespace(
    **dict(
        Inferno=LinearSegmentedColormap.from_list(
            "inferno",
            [
                (0.0, "#000000"),  # black
                (0.2, "#55007F"),  # dark purple
                (0.4, "#AA00FF"),  # magenta
                (0.6, "#FF4500"),  # reddish orange
                (0.8, "#FFFF00"),  # yellow
                (1.0, "#FFFFAA"),  # pale yellow
            ],
        )
    )
)


class Ionogram(object):

    def __init__(
        self,
        dates: dt.datetime = None,
        fig_title: str = "",
        nrows: int = 2,
        ncols: int = 3,
        font_size: float = 10,
        figsize: tuple = (6, 3),
    ):
        self.dates = dates
        self.ncols = ncols
        self.nrows = nrows
        self.fig, self.axes = plt.subplots(
            figsize=(figsize[0] * ncols, figsize[1] * nrows),
            dpi=300,
            nrows=nrows,
            ncols=ncols,
        )  # Size for website
        if type(self.axes) == list or type(self.axes) == np.ndarray:
            self.axes = self.axes.ravel()
        self.fig_title = fig_title
        self.font_size = font_size
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
        cmap: Union[str, LinearSegmentedColormap] = COLOR_MAPS.Inferno,
        prange: List[float] = [5, 70],
        xticks: List[float] = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
        text: str = None,
        del_ticks: bool = True,
    ) -> None:
        ax = self._add_axis(del_ticks)
        ax.set_xlim(np.log10(xlim))
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylim(ylim)
        ax.set_ylabel(
            ylabel,
            fontdict={"size": self.font_size},
        )
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
                fontdict={"size": self.font_size},
            )
        if add_cbar:
            self._add_colorbar(im, self.fig, ax, label=cbar_label.format(mode))
        return ax

    def add_ionogram_traces(
        self,
        frequency: np.array,
        height: np.array,
        mode: str = "O",
        xlabel: str = "Frequency, MHz",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [50, 400],
        xlim: List[float] = [1, 22],
        xticks: List[float] = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
        text: str = None,
        del_ticks: bool = True,
        alpha: float = 0.8,
        ms: float = 0.7,
        color: str = "r",
        ax=None,
    ) -> None:
        ax = ax if ax else self._add_axis(del_ticks)
        ax.set_xlim(np.log10(xlim))
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylim(ylim)
        ax.set_ylabel(
            ylabel,
            fontdict={"size": self.font_size},
        )
        ax.plot(np.log10(frequency), height, color + ".", ms=ms, alpha=alpha)
        if np.logical_not(del_ticks):
            ax.set_xticks(np.log10(xticks))
            ax.set_xticklabels(xticks)
        text = (
            text if text else f"{mode}-mode/{self.dates[0].strftime('%Y%m%d %H%M')} UT"
        )
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

    def _add_axis(self, del_ticks=True):
        ax = (
            self.axes[self._num_subplots_created]
            if type(self.axes) == np.ndarray or type(self.axes) == list
            else self.axes
        )
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
                fontdict={"size": self.font_size},
            )
        self._num_subplots_created += 1
        return ax

    def add_interval_plots(
        self,
        df: pd.DataFrame,
        mode: str = "O",
        xlabel: str = "Time, UT",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [50, 800],
        xlim: List[dt.datetime] = None,
        add_cbar: bool = True,
        cbar_label: str = "{}-mode Power, dB",
        cmap: str = "Spectral",
        prange: List[float] = [5, 70],
        noise_scale: float = 1.2,
        date_format: str = r"$%H^{%M}$",
        del_ticks: bool = False,
        xtick_locator: mdates.HourLocator = mdates.HourLocator(interval=4),
        xdate_lims: List[dt.datetime] = None,
    ):
        xlim = xlim if xlim is not None else [df.time.min(), df.time.max()]
        ax = self._add_axis(del_ticks=del_ticks)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontdict={"size": self.font_size})
        ax.xaxis.set_major_locator(xtick_locator)
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        Zval, lims = (
            np.array(df[f"{mode}_mode_power"]),
            np.array(df[f"{mode}_mode_noise"]),
        )
        Zval[Zval < lims * noise_scale] = np.nan
        df[f"{mode}_mode_power"] = Zval
        X, Y, Z = utils.get_gridded_parameters(
            df,
            xparam="time",
            yparam="range",
            zparam=f"{mode}_mode_power",
            rounding=False,
        )
        Z[Z<prange[0]] = prange[0]
        levels = np.linspace(prange[0], prange[1], 5)
        # Overlay filled contours for the same data
        im = ax.contourf(
            X,
            Y,
            Z.T,
            levels=levels,
            cmap=cmap,
            alpha=0.4,
            zorder=4,
        )

        # Overlay contour lines
        cs = ax.contour(
            X,
            Y,
            Z.T,
            levels=levels,
            colors="k",
            linewidths=0.5,
            zorder=5,
        )
        # Optionally label the contour lines
        ax.clabel(cs, inline=True, fontsize=self.font_size * 0.5)
        ax.text(
            0.01, 1.05, self.fig_title, ha="left", va="center", transform=ax.transAxes
        )
        if xdate_lims is not None:
            ax.set_xlim(xdate_lims)
        if add_cbar:
            self._add_colorbar(im, self.fig, ax, label=cbar_label.format(mode))
        return ax

    def add_TS(
        self,
        time: List,
        ys: np.array,
        ms=0.6,
        alpha=0.7,
        ylim: List = None,
        xlim: List[dt.datetime] = None,
        ylabel: str = r"$foF_2$, MHz",
        xlabel: str = "Time, UT",
        color: str = "r",
        marker: str = ".",
        major_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 4)),
        minor_locator: mdates.RRuleLocator = mdates.MinuteLocator(
            byminute=range(0, 60, 30)
        ),
    ):
        ylim = ylim if ylim else [np.min(ys), np.max(ys)]
        ax = self._add_axis(del_ticks=False)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        ax.plot(time, ys, marker=marker, color=color, ms=ms, alpha=alpha, ls="None")
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
