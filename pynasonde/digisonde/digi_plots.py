import datetime as dt
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

import pynasonde.digisonde.digi_utils as utils

DATE_FORMAT: str = r"$%H^{%M}$"


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

    def add_TS(
        self,
        df: pd.DataFrame,
        xparam: str = "datetime",
        yparam: str = "th",
        zparam: str = "pf",
        cbar_label: str = r"$f_0$, MHz",
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
        title: str = None,
        add_cbar: bool = True,
        zparam_lim: float = 15.0,
        plot_type: str = "pcolor",
        scatter_ms: float = 4,
    ):
        xparam = "local_" + xparam if self.draw_local_time else xparam
        xlabel = xlabel.replace("UT", "LT") if self.draw_local_time else xlabel
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks=False)
        xlim = xlim if xlim is not None else [df[xparam].min(), df[xparam].max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(DATE_FORMAT))
        df = df[df[zparam] <= zparam_lim]
        if plot_type == "pcolor":
            X, Y, Z = utils.get_gridded_parameters(
                df,
                xparam=xparam,
                yparam=yparam,
                zparam=zparam,
                rounding=False,
            )
            im = ax.pcolormesh(
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
        else:
            im = ax.scatter(
                df[xparam],
                df[yparam],
                c=df[zparam],
                zorder=3,
                cmap=cmap,
                vmax=prange[1],
                vmin=prange[0],
                s=scatter_ms,
                marker="s",
            )
        if title:
            ax.text(0.95, 1.05, title, ha="right", va="center", transform=ax.transAxes)
        if add_cbar:
            self._add_colorbar(im, self.fig, ax, label=cbar_label)
        return (ax, im)

    def plot_TS(
        self,
        df: pd.DataFrame,
        xparam: str = "datetime",
        right_yparams: List[str] = ["hmF1"],
        left_yparams: List[str] = ["foF1", "foF1p", "foEs"],
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
        title: str = None,
    ):
        xparam = "local_" + xparam if self.draw_local_time else xparam
        xlabel = xlabel.replace("UT", "LT") if self.draw_local_time else xlabel
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks=False)
        xlim = xlim if xlim is not None else [df[xparam].min(), df[xparam].max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabels[0])
        ax.set_ylim(left_ylim)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(DATE_FORMAT))
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
        if title:
            ax.text(0.95, 1.05, title, ha="right", va="center", transform=ax.transAxes)
        tax = None
        if len(right_yparams) > 0:
            tax = ax.twinx()
            tax.set_ylabel(ylabels[1])
            tax.set_ylim(right_ylim)
            colors.reverse()
            for y, col in zip(right_yparams, colors):
                tax.plot(
                    df[xparam],
                    df[y],
                    marker="D",
                    color=col,
                    ms=ms,
                    alpha=alpha,
                    ls="None",
                    label=y,
                )
            tax.xaxis.set_major_locator(major_locator)
            tax.xaxis.set_major_locator(minor_locator)
            tax.xaxis.set_major_formatter(DateFormatter(DATE_FORMAT))
        return (ax, tax)

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

    @staticmethod
    def plot_isodensity_contours(
        df: pd.DataFrame,
        xparam: str = "date",
        yparam: str = "th",
        zparam: str = "pf",
        xlabel: str = "Time, UT",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [50, 300],
        major_locator: mdates.RRuleLocator = mdates.HourLocator(
            byhour=range(0, 24, 12)
        ),
        minor_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 4)),
        xlim: List[dt.datetime] = None,
        fbins: List[float] = [1, 2, 3, 4, 5, 6, 7, 8],
        text: str = None,
        del_ticks: bool = False,
        fname: str = None,
        figsize: tuple = (5, 3),
        lw: float = 0.7,
        alpha: float = 0.8,
        zorder: int = 4,
        cmap="Spectral",
    ):
        plot = SaoSummaryPlots(figsize=figsize, nrows=1, ncols=1)
        ax = plot.get_axes(del_ticks)
        xlim = xlim if xlim is not None else [df[xparam].min(), df[xparam].max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(DATE_FORMAT))
        for i in range(len(fbins) - 1):
            f_max, f_min = fbins[i + 1], fbins[i]
            o = df[(df[zparam] >= f_min) & (df[zparam] <= f_max)]
            im = ax.scatter(
                o[xparam],
                o[yparam],
                c=o[zparam],
                marker="s",
                s=1.5,
                cmap=cmap,
                vmax=f_max,
                vmin=f_min,
                zorder=zorder,
                alpha=alpha,
            )
        if text:
            ax.text(
                0.05,
                0.9,
                text,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict={"size": plot.font_size},
            )
        if fname:
            plot.save(fname)
        plot.close()
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
        ax.xaxis.set_major_formatter(DateFormatter(DATE_FORMAT))
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
        draw_local_time: bool = False,
    ):
        xparam = "local_" + xparam if draw_local_time else xparam
        dvlplot = SkySummaryPlots(
            figsize=figsize,
            nrows=3,
            ncols=1,
            subplot_kw=None,
            draw_local_time=draw_local_time,
        )
        for i, y, col, err, lab in zip(
            range(len(yparams)), yparams, colors, errors, labels
        ):
            text = text if text else ""
            text = (
                text
                + f"{df[xparam].iloc[0].strftime('%d %b')}-{df[xparam].iloc[-1].strftime('%d %b, %Y')}"
                if i == 0
                else None
            )
            ylabel = rf"Velocity({lab}), m/s"
            xlabel = "Time, UT" if i == 2 else ""
            xlabel = "Time, LT" if xlabel == "Time, UT" and draw_local_time else xlabel
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
        else:
            return dvlplot
