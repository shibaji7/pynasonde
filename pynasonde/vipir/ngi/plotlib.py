"""Plotting helpers for visualizing VIPIR NGI ionograms and time-series data."""

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
    """Convenience wrapper around a Matplotlib figure for VIPIR ionograms.

    Attributes:
        dates: Optional list of timestamps associated with the plotted data.
        ncols: Number of subplot columns requested at construction.
        nrows: Number of subplot rows requested at construction.
        fig: Matplotlib figure hosting the subplots.
        axes: Flattened array of Matplotlib axes corresponding to each slot.
        fig_title: Title drawn on the first subplot.
        font_size: Base font size used throughout the figure.
    """

    def __init__(
        self,
        dates: dt.datetime = None,
        fig_title: str = "",
        nrows: int = 2,
        ncols: int = 3,
        font_size: float = 10,
        figsize: tuple = (6, 3),
    ):
        """Initialize the figure canvas and subplot grid.

        Args:
            dates: Optional timestamp(s) used when labeling plots.
            fig_title: Title added above the first subplot.
            nrows: Number of subplot rows.
            ncols: Number of subplot columns.
            font_size: Base font size applied through `utils.setsize`.
            figsize: Base width/height (in inches) of a single subplot.
        """
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
        txt_color: str = "w",
    ) -> None:
        """Render a single ionogram heatmap into the next subplot slot.

        Args:
            frequency: Plasma frequency axis (MHz).
            height: Virtual height axis (km).
            value: 2-D power array aligned with `frequency` × `height`.
            mode: Descriptor used in the colorbar label (e.g., O or X).
            xlabel: X-axis label text.
            ylabel: Y-axis label text.
            ylim: Y-axis limits (km).
            xlim: Frequency limits (MHz).
            add_cbar: Whether to append a colorbar for this axes.
            cbar_label: Format string used for the colorbar title.
            cmap: Matplotlib colormap or name.
            prange: Min/max bounds (dB) applied to the power field.
            xticks: Explicit tick positions shown when `del_ticks` is False.
            text: Optional annotation placed inside the axes.
            del_ticks: Remove axis ticks for a cleaner grid layout.

        Returns:
            Matplotlib axes instance that received the plot.
        """
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
                fontdict={"size": self.font_size, "color": txt_color},
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
        """Overlay traced ionogram echoes as point markers.

        Args:
            frequency: Frequency coordinates (MHz) of the traced points.
            height: Virtual heights (km) for each trace sample.
            mode: Descriptor used in on-plot text (e.g., O or X).
            xlabel: X-axis label text.
            ylabel: Y-axis label text.
            ylim: Y-axis limits (km).
            xlim: Frequency limits (MHz).
            xticks: Tick positions when `del_ticks` is False.
            text: Optional annotation inside the axes; defaults to timestamp.
            del_ticks: Remove ticks for cleaner multipanel layouts.
            alpha: Marker transparency.
            ms: Marker size.
            color: Base color (combined with "." marker).
            ax: Optional axes to draw on (defaults to the next subplot).

        Returns:
            Matplotlib axes instance that received the plot.
        """
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
        """Return the next available subplot axis, optionally stripping ticks."""
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
        kind: str = "pcolormesh",
    ):
        """Plot mode-specific interval statistics on a time/height grid.

        Args:
            df: DataFrame containing time, range, and power columns.
            mode: Mode prefix (O/X/etc.) used to select DataFrame columns.
            xlabel: X-axis label text.
            ylabel: Y-axis label text.
            ylim: Limits for the height axis.
            xlim: Optional datetime bounds passed to `set_xlim`.
            add_cbar: Whether to include a colorbar for the filled contour.
            cbar_label: Format string applied to the colorbar label.
            cmap: Matplotlib colormap name.
            prange: Minimum/maximum dB values shown in the contour.
            noise_scale: Multiplier applied to the noise floor when masking.
            date_format: Matplotlib datetime formatter string.
            del_ticks: Remove ticks before plotting, if desired.
            xtick_locator: Locator used for primary x-axis ticks.
            xdate_lims: Optional override for the x-axis limits.

        Returns:
            Matplotlib axes instance containing the interval plot.
        """
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
        Z[Z < prange[0]] = prange[0]
        if kind == "pcolormesh":
            im = ax.pcolormesh(
                X,
                Y,
                Z.T,
                cmap=cmap,
                vmax=prange[1],
                vmin=prange[0],
                zorder=3,
            )
        elif kind == "contourf":
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
        """Plot a time-series curve (e.g., foF2) in the next subplot slot.

        Args:
            time: Sequence of timestamps corresponding to `ys`.
            ys: Values to plot.
            ms: Marker size.
            alpha: Marker transparency.
            ylim: Optional y-axis limits; defaults to data min/max.
            xlim: Optional x-axis limits.
            ylabel: Y-axis label text.
            xlabel: X-axis label text.
            color: Matplotlib color specification.
            marker: Marker style for the scatter plot.
            major_locator: Locator for major ticks on the time axis.
            minor_locator: Locator for minor ticks on the time axis.

        Returns:
            Matplotlib axes instance that received the plot.
        """
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
        """Attach a vertical colorbar to the right of the supplied axis.

        Args:
            im: Mappable returned by a Matplotlib plotting call.
            fig: Parent figure.
            ax: Axis the colorbar aligns with.
            label: Text label applied to the colorbar.
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
        """Persist the assembled figure to disk."""
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        """Release Matplotlib resources associated with the figure."""
        self.fig.clf()
        plt.close()
        return
