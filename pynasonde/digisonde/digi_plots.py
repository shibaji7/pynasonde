import datetime as dt
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

import pynasonde.digisonde.digi_utils as utils


class DigiPlots(object):
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
        subplot_kw: dict = None,
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


class SaoSummaryPlots(DigiPlots):
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

    def add_TS(
        self,
        df: pd.DataFrame,
        xparam: str = "date",
        yparam: str = "th",
        zparam: str = "pf",
        cbar_label: str = r"$foF_2$, MHz",
        cmap: str = "Spectral",
        prange: List[float] = [1, 15],
        ylabel: str = "Height, km",
        xlabel: str = "Time, UT",
        major_locator: mdates.RRuleLocator = mdates.HourLocator(
            byhour=range(0, 24, 12)
        ),
        minor_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 4)),
        ylim: List = [80, 800],
        xlim: List[dt.datetime] = None,
        add_cbar: bool = True,
        zparam_lim: float = 15.0,
    ):
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks=False)
        xlim = xlim if xlim is not None else [df[xparam].min(), df[xparam].max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        df = df[df[zparam] <= zparam_lim]
        X, Y, Z = utils.get_gridded_parameters(
            df,
            xparam=xparam,
            yparam=yparam,
            zparam=zparam,
            rounding=False,
        )
        im = ax.pcolor(
            X,
            Y,
            Z.T,
            lw=0.01,
            edgecolors="None",
            cmap=cmap,
            vmax=prange[1],
            vmin=prange[0],
            zorder=3,
        )
        ax.text(
            0.01, 1.05, self.fig_title, ha="left", va="center", transform=ax.transAxes
        )
        if add_cbar:
            self._add_colorbar(im, self.fig, ax, label=cbar_label)
        return

    def plot_TS(
        self,
        df: pd.DataFrame,
        xparam: str = "date",
        right_yparams: List[str] = ["hmF1"],
        left_yparams: List[str] = ["foF1", "foF1p", "foE"],
        colors: List[str] = ["r", "b", "k"],
        ylabels: List[str] = ["Frequencies, MHz", "Height, km"],
        xlabel: str = "Time, UT",
        marker: str = ".",
        major_locator: mdates.RRuleLocator = mdates.HourLocator(
            byhour=range(0, 24, 12)
        ),
        minor_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 4)),
        right_ylim: List = [100, 400],
        left_ylim: List = [1, 15],
        xlim: List[dt.datetime] = None,
        ms: float = 0.6,
        alpha: float = 0.7,
    ):
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks=False)
        xlim = xlim if xlim is not None else [df[xparam].min(), df[xparam].max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabels[0])
        ax.set_ylim(left_ylim)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        for y, col in zip(left_yparams, colors):
            ax.plot(
                df[xparam],
                df[y],
                marker=marker,
                color=col,
                ms=ms,
                alpha=alpha,
                ls="None",
                label=y,
            )
        ax.legend(loc=2)
        ax = ax.twinx()
        ax.set_ylabel(ylabels[1])
        ax.set_ylim(right_ylim)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        return

    def plot_ionogram(
        self,
        df: pd.DataFrame,
        xparam: str = "pf",
        yparam: str = "th",
        xlabel: str = "Frequency, MHz",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [0, 600],
        xlim: List[float] = [1, 22],
        xticks: List[float] = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
        text: str = None,
        del_ticks: bool = False,
        ls: str = "-",
        lcolor: str = "k",
        lw: float = 0.7,
        zorder: int = 2,
    ):
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)
        ax.set_xlim(np.log10(xlim))
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylim(ylim)
        ax.set_ylabel(
            ylabel,
            fontdict={"size": self.font_size},
        )
        ax.plot(
            np.log10(df[xparam]), df[yparam], ls=ls, zorder=zorder, lw=lw, color=lcolor
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
        return

    def plot_isodensity_contours(self):
        return


class SkySummaryPlots(DigiPlots):
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
        subplot_kw: dict = dict(projection="polar"),
    ):
        super().__init__(
            fig_title, nrows, ncols, font_size, figsize, date, date_lims, subplot_kw
        )
        return

    def convert_to_rt(self, row, xparam, yparam, zparam):
        """Convert to r/theta coordinate"""

        row["r"], row["theta"] = (
            np.sqrt(row[xparam] ** 2 + row[yparam] ** 2),
            np.arctan2(row[yparam], row[xparam]),
        )
        row["marker"] = "+" if row[zparam] > 0 else "o"
        return row

    def plot_skymap(
        self,
        df: pd.DataFrame,
        xparam: str = "x_coord",
        yparam: str = "y_coord",
        zparam: str = "spect_dop",
        theta_lim: List[float] = [0, 360],
        rlim: float = 5,
        text: str = None,
        del_ticks: bool = True,
        cmap: str = "Spectral",
        cbar: bool = True,
        clim: List[float] = [-5, 5],
        cbar_label: str = "Doppler, Hz",
        ms: float = 1.5,
        zorder: int = 2,
    ):
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)
        ax.set_thetamin(theta_lim[0])
        ax.set_thetamax(theta_lim[1])
        ax.set_rmax(rlim)
        df = df.apply(self.convert_to_rt, args=(xparam, yparam, zparam), axis=1)
        im = ax.scatter(
            df["theta"],
            df["r"],
            c=df[zparam],
            cmap=cmap,
            s=ms,
            marker="D",
            zorder=zorder,
            vmax=clim[1],
            vmin=clim[0],
        )
        for xtick in np.linspace(0, rlim, 5):
            ax.axhline(xtick, ls="--", lw=0.4, alpha=0.6, color="k")
        for ytick in [0, np.pi / 2, np.pi, 1.5 * np.pi]:
            ax.axvline(ytick, ls="-", lw=0.4, alpha=0.6, color="k")
        ax.text(
            1.01,
            0.5,
            "East",
            ha="left",
            va="center",
            transform=ax.transAxes,
            rotation=90,
        )
        ax.text(
            0.5,
            1.01,
            "North",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
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
        if cbar:
            self._add_colorbar(im, self.fig, ax, cbar_label, [0.05, 0.0125, 0.015, 0.5])
        return

    def plot_doppler_waterfall(self):
        return

    def plot_drift_velocities(
        self,
        df: pd.DataFrame,
        xparam: str = "datetime",
        yparam: str = "Vx",
        color: str = "r",
        error: str = "Vx_err",
        ylabel: str = r"Velocity ($V_x$), m/s",
        xlabel: str = "Time, UT",
        text: str = None,
        del_ticks: bool = False,
        fmt: str = "o",
        lw: float = 0.7,
        alpha: float = 0.8,
        zorder: int = 4,
        major_locator: mdates.RRuleLocator = mdates.HourLocator(
            byhour=range(0, 24, 12)
        ),
        minor_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 4)),
        ylim: List = [-100, 100],
        xlim: List[dt.datetime] = None,
    ):
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)
        xlim = xlim if xlim is not None else [df[xparam].min(), df[xparam].max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        ax.errorbar(
            df[xparam],
            df[yparam],
            yerr=df[error],
            color=color,
            fmt=fmt,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
            capsize=1,
            capthick=1,
            ms=1,
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

        return

    @staticmethod
    def plot_dvl_drift_velocities(
        df: pd.DataFrame,
        xparam: str = "datetime",
        yparams: List[str] = ["Vx", "Vy", "Vz"],
        colors: List[str] = ["r", "b", "k"],
        errors: List[str] = ["Vx_err", "Vy_err", "Vz_err"],
        labels: List[str] = ["$V_x$", "$V_y$", "$V_z$"],
        text: str = None,
        del_ticks: bool = False,
        fmt: str = "o",
        lw: float = 0.7,
        alpha: float = 0.8,
        zorder: int = 4,
        major_locator: mdates.RRuleLocator = mdates.HourLocator(
            byhour=range(0, 24, 12)
        ),
        minor_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 4)),
        ylim: List = [-100, 100],
        xlim: List[dt.datetime] = None,
        fname: str = None,
        figsize: tuple = (2.5, 7),
    ):
        dvlplot = SkySummaryPlots(figsize=figsize, nrows=3, ncols=1, subplot_kw=None)
        for i, y, col, err, lab in zip(
            range(len(yparams)), yparams, colors, errors, labels
        ):
            ylabel = rf"Velocity({lab}), m/s"
            xlabel = "Time, UT" if i == 2 else ""
            dvlplot.plot_drift_velocities(
                df,
                xparam,
                y,
                col,
                err,
                ylabel,
                xlabel,
                text,
                del_ticks,
                fmt,
                lw,
                alpha,
                zorder,
                major_locator,
                minor_locator,
                ylim,
                xlim,
            )
        if fname:
            dvlplot.save(fname)
        dvlplot.close()
        return
