"""Plotting helpers for the Digisonde subpackage.

This module provides higher-level plotting utilities used by Digisonde
parsers and analyses. The main classes are:

- ``DigiPlots``: base figure/axes manager and small conveniences.
- ``SaoSummaryPlots``: time-height and ionogram-style visualizations.
- ``SkySummaryPlots``: polar/skymap plotting helpers and drift plots.
- ``RsfIonogram``: RSF-format ionogram renderer.

Each class accepts pandas DataFrames produced by the library parsers and
offers simple, documented methods for common visualizations.
"""

import datetime as dt
from types import SimpleNamespace
from typing import List, Optional

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter

import pynasonde.digisonde.digi_utils as utils

DATE_FORMAT: str = r"$%H^{%M}$"

COLOR_MAPS = SimpleNamespace(
    **dict(
        RedBlackBlue=LinearSegmentedColormap.from_list(
            "RedBlackBlue",
            [
                (0.0, "#FF0000"),  # reddish
                (0.5, "#000000"),  # black
                (1.0, "#131DE3E8"),  # black
            ],
        ),
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
        ),
    )
)


def search_color_schemes(
    num_colors: int,
    bounds: List[dict],
    search_length: int = 10000,
    color_map: LinearSegmentedColormap = COLOR_MAPS.RedBlackBlue,
) -> List:
    """Search for random color combinations sampled from a colormap that satisfy bounds.

    This helper tries a number of random seeds and samples `num_colors` values
    from `color_map` for each seed. Each sampled value is tested against the
    corresponding `bounds` entry which should be a dict with `gt` and `lt`
    (greater-than and less-than) thresholds. Seeds that produce values within
    all bounds are recorded and returned.

    Args:
        num_colors: Number of colors to sample per seed.
        bounds: List of dicts, one per color, with numeric keys `gt` and `lt`.
        search_length: Number of seeds to try (default 10000). Higher values
            increase the chance of finding acceptable combinations.
        color_map: A Matplotlib colormap used to convert sampled fractions to
            RGBA colors.

    Returns:
        A list of dicts with keys: `seed` (int), `decimals` (list of floats),
            and `color` (list of RGBA tuples).
    """
    cmap = []
    for se in range(search_length):
        np.random.seed(se)
        rands = [np.random.rand() for _ in range(num_colors)]
        check_count = 0
        for r, b in zip(rands, bounds):
            if r >= b["gt"] and r <= b["lt"]:
                check_count += 1
        if check_count == num_colors:
            cmap.append(
                dict(seed=se, decimals=rands, color=[color_map(r) for r in rands])
            )
    return cmap


class DigiPlots(object):
    """Base plotting helper.

    DigiPlots wraps common Matplotlib figure and axis management used across
    the Digisonde plotting helpers. It centralizes figure creation, sizing,
    colorbar placement and provides small convenience helpers like
    `get_axes`, `save` and `close`.

    Attributes:
        fig_title str: Title shown on the first subplot.
        nrows int: Subplot grid(row) layout.
        ncols int: Subplot grid(col) layout.
        font_size float: Base font size applied via `utils.setsize`.
        figsize tuple(int): Per-subplot size (width, height).
        date datetime: Optional reference date used by some plotters.
        date_lims list(datetime): Optional x-axis limits as datetimes.
        subplot_kw dict: Passed to `plt.subplots` for e.g. polar projections.
        draw_local_time bool: If True, certain methods will use local time columns
            (prefixed with `local_`) instead of UTC.
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
            figsize=(figsize[0] * ncols, figsize[1] * nrows),
            dpi=300,
            nrows=nrows,
            ncols=ncols,
            subplot_kw=self.subplot_kw,
        )  # Size for website
        if type(self.axes) == list or type(self.axes) == np.ndarray:
            self.axes = self.axes.ravel()
        return

    def get_axes(self, del_ticks=True):
        """Return the next axes for plotting and optionally remove ticks.

        This advances an internal subplot counter so subsequent calls will
        return the next axis in the grid. When the first subplot is returned
        the figure title is drawn.
        Args:
            del_ticks: If True, remove x/y ticks from the returned axis.
        """
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

    def save(self, filepath: str):
        """Save current figure to `filepath` using tight bounding box.
        Args:
            filepath: Full path (including filename) to save the figure.
        """
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        """Clear and close the current Matplotlib figure.

        This frees memory held by the figure and closes the associated
        Matplotlib window/backend. Call this when the plot is no longer
        needed (for example after saving to disk).
        """
        self.fig.clf()
        plt.close()
        return

    def _add_colorbar(
        self,
        im: plt.cm.ScalarMappable,
        fig: plt.Figure,
        ax: plt.axes,
        label: str = "",
        mpos: List[float] = [0.025, 0.0125, 0.015, 0.5],
    ):
        """Add a colorbar to the right of an axis.

        Parameters mirror common Matplotlib `colorbar` usage but provide a
        simple positioning interface via `mpos` describing relative offsets
        and size of the colorbar axes.

        Args:
            im: The Matplotlib ScalarMappable (e.g. QuadMesh) to use for
                colorbar generation.
            fig: The Matplotlib figure containing the axis.
            ax: The axis to which the colorbar applies.
            label: Colorbar label text.
            mpos: List of 4 floats describing the colorbar axes position
                relative to `ax`. The list contains [left, bottom, width,
                height] where left/bottom are offsets from `ax` and height is
                a fraction of `ax` height. For example, the default value
                places a thin colorbar to the right of `ax` with half its
                height.
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
    """Plotting helpers for SAO summary time-series and ionogram-style plots.

    This class provides higher-level plotting methods built on top of
    `DigiPlots` such as `add_TS` (time-series rasterized frequency plots) and
    `plot_TS` (time-series of scalar parameters). The methods accept pandas
    DataFrames produced by `pynasonde.digisonde.parsers` and use column names
    to select x/y/z parameters.

    This method inherits from `DigiPlots` and thus also provides methods and attributes including
    `get_axes`, `save`, `close`, and attributes like `fig_title`, `nrows`, `ncols`, `font_size`,
    `figsize`, `date`, `date_lims`, `subplot_kw`, and `draw_local_time`.
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
        cmap: str | LinearSegmentedColormap = COLOR_MAPS.Inferno,
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
        zparam_lim: float = np.nan,
        plot_type: str = "pcolor",
        scatter_ms: float = 4,
    ):
        """Create a time-height colored grid (pcolor) or scatter showing the
        parameter `zparam` as a color over time (`xparam`) and height/level
        (`yparam`).

        Args:
            df: Input DataFrame containing `xparam`, `yparam` and `zparam` columns.
            xparam: Column name for the x-axis (time).
            yparam: Column name for the y-axis (height).
            zparam: Column name for the color parameter.
            cbar_label: Colorbar label text.
            cmap: Matplotlib colormap or name used for coloring.
            prange: List of two floats defining the color range (vmin, vmax).
            ylabel: Y-axis label text.
            xlabel: X-axis label text.
            major_locator: Matplotlib date locator for major x-axis ticks.
            minor_locator: Matplotlib date locator for minor x-axis ticks.
            ylim: List of two floats defining the y-axis limits.
            xlim: List of two datetimes defining the x-axis limits. If None,
                the full range of `df[xparam]` is used.
            title: Optional title text shown above the plot.
            add_cbar: If True, add a colorbar to the right of the plot.
            zparam_lim: If provided, filter out rows where `zparam` exceeds
                this value.
            plot_type: Either 'pcolor' (default) or 'scatter' to choose the
                rendering method.
            scatter_ms: If `plot_type=='scatter'`, use this marker size.

        Returns:
            Tuple (ax, im) where `im` is the Matplotlib QuadMesh or PathCollection
                used for colorbar generation.
        """
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
        if not np.isnan(zparam_lim):
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
        ylabels: List[str] = ["Frequencies, MHz", "Height, km"],
        xlabel: str = "Time, UT",
        marker: str = "s",
        major_locator: mdates.RRuleLocator = mdates.HourLocator(
            byhour=range(0, 24, 12)
        ),
        minor_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 4)),
        right_ylim: List = [100, 400],
        left_ylim: List = [1, 15],
        xlim: List[dt.datetime] = None,
        ms: float = 1,
        alpha: float = 1.0,
        title: str = None,
        right_axis_color: str = None,
        left_axis_color: str = None,
        color_map: LinearSegmentedColormap = COLOR_MAPS.RedBlackBlue,
        seed: int = 5,
    ):
        """Plot multiple scalar timeseries from `df` on left and optional
        right axes.

        The left axis shows the `left_yparams` series and the right axis
        (if `right_yparams` provided) shows additional series using a twin
        y-axis. Colors are picked from a color_map seeded for reproducible
        plots.

        Args:
            df: Input DataFrame containing `xparam`, `left_yparams` and
                `right_yparams` columns.
            xparam: Column name for the x-axis (time).
            right_yparams: List of column names to plot on the right y-axis.
            left_yparams: List of column names to plot on the left y-axis.
            ylabels: List of two strings for the left and right y-axis labels.
            xlabel: X-axis label text.
            marker: Matplotlib marker style for all series.
            major_locator: Matplotlib date locator for major x-axis ticks.
            minor_locator: Matplotlib date locator for minor x-axis ticks.
            right_ylim: List of two floats defining the right y-axis limits.
            left_ylim: List of two floats defining the left y-axis limits.
            xlim: List of two datetimes defining the x-axis limits. If None,
                the full range of `df[xparam]` is used.
            ms: Marker size for all series.

        Returns:
            Tuple (ax, tax) where `ax` is the left axis and `tax` is the
                right axis (or None if no right_yparams provided).
        """
        np.random.seed(seed)
        xparam = "local_" + xparam if self.draw_local_time else xparam
        xlabel = xlabel.replace("UT", "LT") if self.draw_local_time else xlabel
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks=False)
        xlim = xlim if xlim is not None else [df[xparam].min(), df[xparam].max()]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylim(left_ylim)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(DATE_FORMAT))
        colors = [color_map(np.random.rand()) for _ in range(len(left_yparams))]
        left_axis_color = colors[0] if left_axis_color is None else left_axis_color
        ax.set_ylabel(ylabels[0], color=left_axis_color)
        ax.tick_params(axis="y", colors=left_axis_color)
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
            tax.set_ylim(right_ylim)
            colors = [color_map(np.random.rand()) for _ in range(len(left_yparams))]
            right_axis_color = (
                colors[0] if right_axis_color is None else right_axis_color
            )
            tax.set_ylabel(ylabels[1], color=right_axis_color)
            tax.tick_params(axis="y", colors=right_axis_color)
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
        ax: plt.axes = None,
        kind: str = "ionogram",
    ):
        """Plot an ionogram-style trace: frequency (log-scaled) vs virtual
        height.

        If `kind=='ionogram'` lines are drawn, otherwise individual points are
        plotted. The x-axis is log10-scaled but tick labels are shown in
        linear frequency values for readability.

        Args:
            df: Input DataFrame containing `xparam` and `yparam` columns.
            xparam: Column name for the x-axis (frequency).
            yparam: Column name for the y-axis (height).
            xlabel: X-axis label text.
            ylabel: Y-axis label text.
            ylim: List of two floats defining the y-axis limits.
            xlim: List of two floats defining the x-axis limits.
            xticks: List of floats defining the x-axis tick locations.
            text: Optional text shown in the upper left of the plot.
            del_ticks: If True, remove x/y ticks from the axis.
            ls: Line style (if `kind=='ionogram'`).
            lcolor: Line/marker color.
            lw: Line width or marker size.
            zorder: Matplotlib z-order for layering.
            ax: If provided, use this axis instead of creating a new one.
            kind: Either 'ionogram' (default) or 'scatter' to choose the
                rendering method.
        """
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)
        ax.set_xlim(np.log10(xlim))
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylim(ylim)
        ax.set_ylabel(
            ylabel,
            fontdict={"size": self.font_size},
        )
        if kind == "ionogram":
            ax.plot(
                np.log10(df[xparam]),
                df[yparam],
                ls=ls,
                zorder=zorder,
                lw=lw,
                color=lcolor,
            )
        else:
            ax.scatter(
                np.log10(df[xparam]),
                df[yparam],
                marker="s",
                zorder=zorder,
                s=lw,
                color=lcolor,
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

    def add_isodensity_contours(
        self,
        df: pd.DataFrame,
        xparam: str = "datetime",
        yparam: str = "th",
        zparam: str = "pf",
        xlabel: str = "Time, UT",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [50, 500],
        xlim: List[dt.datetime] = None,
        fbins: List[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        cmap: str = "plasma",
        prange: List[float] = None,
        contour_colors: str = "k",
        contour_lw: float = 0.5,
        major_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 2)),
        minor_locator: mdates.RRuleLocator = mdates.HourLocator(byhour=range(0, 24, 1)),
        text: str = None,
        del_ticks: bool = False,
        cbar_label: str = "Plasma Frequency, MHz",
        zorder: int = 4,
    ):
        """Plot daily isodensity contours from stacked SAO height profiles.

        Replicates the Digisonde *Isodensity* visualization (see
        `digisonde.com/images/Digisonde-Isodensity.gif`_): a time–height
        colour map of plasma frequency with frequency-labelled contour lines
        overlaid.

        Each row in *df* must represent one (time, height, plasma-frequency)
        triplet — i.e. the concatenated output of
        :meth:`SaoExtractor.get_height_profile` over many ionograms.
        The data are scatter-plotted using :func:`~matplotlib.axes.Axes.scatter`
        with a continuous colourmap, then :func:`~matplotlib.axes.Axes.tricontour`
        is overlaid at the requested frequency levels for clear contour lines.

        Args:
            df: DataFrame with at least ``xparam``, ``yparam``, and ``zparam``
                columns.  Produced by :meth:`SaoExtractor.get_height_profile`
                (or its XML variant) concatenated across files.
            xparam: Column for the time (x) axis.  Must be datetime-like.
            yparam: Column for the virtual height (y) axis, in km.
            zparam: Column for the frequency (z) axis, in MHz.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            ylim: ``[min, max]`` height limits in km.
            xlim: ``[t_start, t_end]`` datetime limits.  Auto-derived if
                ``None``.
            fbins: Frequency levels (MHz) drawn as contour lines and used for
                the colour scale.
            cmap: Matplotlib colormap name for the scatter fill.
            prange: ``[vmin, vmax]`` colour range in MHz.  Defaults to
                ``[fbins[0], fbins[-1]]``.
            contour_colors: Colour of the overlaid contour lines.
            contour_lw: Line width of the contour lines.
            major_locator: Major tick locator for the time axis.
            minor_locator: Minor tick locator for the time axis.
            text: Optional annotation in the upper-left corner.
            del_ticks: Suppress all axis ticks when ``True``.
            cbar_label: Label on the colour bar.
            zorder: Matplotlib z-order for scatter points.
        """
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)

        df = df.dropna(subset=[xparam, yparam, zparam]).copy()
        df[zparam] = pd.to_numeric(df[zparam], errors="coerce")
        df = df.dropna(subset=[zparam])
        # Filter to valid height range and realistic plasma frequencies
        df = df[(df[yparam] >= ylim[0]) & (df[yparam] <= ylim[1])]
        df = df[(df[zparam] > 0) & (df[zparam] <= (fbins[-1] + 2))]
        df[xparam] = pd.to_datetime(df[xparam])

        xlim = xlim if xlim is not None else [df[xparam].min(), df[xparam].max()]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylabel(ylabel, fontdict={"size": self.font_size})
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_formatter(DateFormatter(DATE_FORMAT))

        vmin, vmax = prange if prange is not None else [fbins[0], fbins[-1]]

        # ── Build a regular time × height grid for pcolormesh ─────────────────
        # Bin time into ionogram slots (detect cadence from unique timestamps)
        times = np.sort(df[xparam].unique())
        heights = np.arange(ylim[0], ylim[1] + 1, 5.0)  # 5 km bins

        df["_h_bin"] = pd.cut(df[yparam], bins=heights, labels=heights[:-1])
        df["_t_bin"] = df[xparam]  # keep exact ionogram timestamp

        pivot = (
            df.groupby(["_t_bin", "_h_bin"], observed=True)[zparam]
            .mean()
            .unstack("_h_bin")
        )
        # Convert index to matplotlib date numbers for pcolormesh
        t_num = mdates.date2num(np.array(pivot.index.to_pydatetime()))
        h_edges = pivot.columns.astype(float).values
        Z = pivot.values  # shape (n_times, n_heights)

        if len(t_num) > 1 and len(h_edges) > 1:
            im = ax.pcolormesh(
                t_num,
                h_edges,
                Z.T,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
                zorder=zorder,
            )

            # Contour lines at each fbins level
            try:
                # Fill NaN with 0 for contouring
                Z_fill = np.where(np.isnan(Z.T), 0, Z.T)
                T_grid, H_grid = np.meshgrid(t_num, h_edges)
                cs = ax.contour(
                    T_grid,
                    H_grid,
                    Z_fill,
                    levels=fbins,
                    colors=contour_colors,
                    linewidths=contour_lw,
                    zorder=zorder + 1,
                )
                ax.clabel(cs, fmt="%g", fontsize=self.font_size - 2, inline=True)
            except Exception:
                pass

            cbar = self.fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label(cbar_label, fontsize=self.font_size - 1)
            cbar.set_ticks(fbins)

        if text:
            ax.text(
                0.02,
                0.96,
                text,
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontdict={"size": self.font_size - 1},
            )
        if not del_ticks:
            ax.tick_params(axis="both", labelsize=self.font_size - 1)
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
        """Plot frequency bands as isodensity contours over time and height.

        This convenience function bins the input DataFrame by `fbins` and
        renders small square markers for points whose `zparam` (frequency)
        falls in each frequency bin. It is useful to visualize the
        distribution of echo frequencies over time/height.

        If `fname` is provided the figure will be saved to disk.

        Args:
            df: Input DataFrame containing `xparam`, `yparam` and `zparam` columns.
            xparam: Column name for the x-axis (time).
            yparam: Column name for the y-axis (height).
            zparam: Column name for the color parameter (frequency).
            xlabel: X-axis label text.
            ylabel: Y-axis label text.
            ylim: List of two floats defining the y-axis limits.
            major_locator: Matplotlib date locator for major x-axis ticks.
            minor_locator: Matplotlib date locator for minor x-axis ticks.
            xlim: List of two datetimes defining the x-axis limits. If None,
                the full range of `df[xparam]` is used.
            fbins: List of frequency bin edges used to group points.
            text: Optional text shown in the upper left of the plot.
            del_ticks: If True, remove x/y ticks from the axis.
            fname: If provided, save the figure to this path.
            figsize: Figure size (width, height) in inches.
            lw: Line width for marker edges.
        """
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
    """Polar/sky plotting utilities.

    Used to create skymaps, drift velocity plots, and other polar visualizations
    from Digisonde-derived coordinate data.
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
        """Convert cartesian x/y coordinates to polar (r, theta).

        This is an `apply` helper used by `plot_skymap`. It computes radial
        distance and angle (theta). Marker style is chosen based on the
        sign of `zparam` (useful to differentiate doppler sign).
        """

        # Radius and angle (note arctan2 returns angle in radians)
        row["r"], row["theta"] = (
            np.sqrt(row[xparam] ** 2 + row[yparam] ** 2),
            -np.arctan2(row[yparam], row[xparam]),
        )
        # Choose a marker to visually separate positive/negative zparam
        row["marker"] = "+" if row[zparam] > 0 else "o"
        return row

    def plot_skymap(
        self,
        df: pd.DataFrame,
        xparam: str = "x_coord",
        yparam: str = "y_coord",
        zparam: str = "spect_dop",
        theta_lim: List[float] = [0, 360],
        rlim: float = 21,
        text: str = None,
        del_ticks: bool = True,
        cmap: str | LinearSegmentedColormap = COLOR_MAPS.RedBlackBlue,
        cbar: bool = True,
        clim: List[float] = [-5, 5],
        cbar_label: str = "Doppler, Hz",
        ms: float = 1.5,
        zorder: int = 2,
        nrticks: int = 5,
        txt_loc: tuple = (0.05, 0.9),
        txt_fontsize: float = 10,
    ):
        """Render a polar skymap of measured points.

        The DataFrame should contain x/y coordinates in same units and a
        color parameter `zparam`. The method converts coords to (r,theta) and
        plots them on a polar projection with optional colorbar.
        """
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)
        ax.set_thetamin(theta_lim[0])
        ax.set_thetamax(theta_lim[1])
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
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
        for xtick in np.linspace(0, rlim - 1, nrticks):
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
                txt_loc[0],
                txt_loc[1],
                text,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict={"size": txt_fontsize},
            )
        ax.set_rmax(rlim)
        if cbar:
            self._add_colorbar(im, self.fig, ax, cbar_label, [0.05, 0.0125, 0.015, 0.5])
        return

    def plot_doppler_waterfall(
        self,
        df: pd.DataFrame,
        xparam: str = "doppler_bin",
        yparam: str = "height_km",
        zparam: str = "amplitude",
        block_idx: int = None,
        xlabel: str = "Doppler Bin (− approach / + recede)",
        ylabel: str = "Virtual Height, km",
        cbar_label: str = "Amplitude, dB",
        xlim: List[float] = None,
        ylim: List[float] = None,
        prange: List[float] = None,
        cmap: str = "inferno",
        text: str = None,
        del_ticks: bool = False,
    ):
        """Plot a Doppler spectra waterfall from a DFT DataFrame.

        Creates a 2-D colour map of amplitude versus Doppler bin (x) and
        virtual height (y) for a single DFT block.  This matches the
        "Doppler Spectra" panel described in the Digisonde-4D manual and the
        DPS4D DFT-format documentation.

        Use :meth:`DftExtractor.to_pandas` to produce *df*.

        Args:
            df: DataFrame from :meth:`DftExtractor.to_pandas` containing at
                minimum ``doppler_bin``, ``height_km``, ``amplitude``, and
                ``block_idx`` columns.
            xparam: Column for the Doppler-bin axis (signed, centred at 0).
            yparam: Column for the height axis (km).
            zparam: Column for amplitude values.
            block_idx: Which DFT block (frequency step) to plot.  If ``None``
                the block with the highest peak amplitude is chosen
                automatically.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            cbar_label: Colour-bar label.
            xlim: ``[min, max]`` Doppler-bin limits.  Auto if ``None``.
            ylim: ``[min, max]`` height limits in km.  Auto if ``None``.
            prange: ``[vmin, vmax]`` amplitude colour range in dB.  If
                ``None`` the range is derived from the 2nd and 98th percentile
                of the selected block's amplitude.
            cmap: Matplotlib colormap name.
            text: Optional annotation in the upper-left corner.
            del_ticks: Suppress axis ticks when ``True``.
        """
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)

        # Auto-select block with strongest signal if not specified
        if block_idx is None:
            block_idx = int(df.groupby("block_idx")[zparam].max().idxmax())

        sub = df[df["block_idx"] == block_idx].copy()
        if sub.empty:
            return

        # Build 2-D amplitude grid: rows=heights, cols=Doppler bins
        heights = np.sort(sub[yparam].unique())
        dbins = np.sort(sub[xparam].unique())
        grid = (
            sub.pivot_table(index=yparam, columns=xparam, values=zparam, aggfunc="mean")
            .reindex(index=heights, columns=dbins)
            .values
        )

        # Auto amplitude range: 2nd–98th percentile of the block
        if prange is None:
            vmin = float(np.nanpercentile(grid, 2))
            vmax = float(np.nanpercentile(grid, 98))
        else:
            vmin, vmax = prange

        X, Y = np.meshgrid(dbins, heights)
        im = ax.pcolormesh(
            X,
            Y,
            grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylabel(ylabel, fontdict={"size": self.font_size})
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.axvline(0, color="w", lw=0.7, ls="--")

        cbar = self.fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=self.font_size - 1)

        if text:
            ax.text(
                0.02,
                0.96,
                text,
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontdict={"size": self.font_size - 1},
                color="w",
            )
        if not del_ticks:
            ax.tick_params(axis="both", labelsize=self.font_size - 1)
        return

    def plot_doppler_spectra(
        self,
        df: pd.DataFrame,
        xparam: str = "doppler_bin",
        yparam: str = "amplitude",
        height_col: str = "height_km",
        block_idx: int = None,
        selected_heights: List[float] = None,
        n_heights: int = 6,
        cmap: str = "viridis",
        xlabel: str = "Doppler Bin (− approach / + recede)",
        ylabel: str = "Amplitude, dB",
        ylim: List[float] = None,
        xlim: List[float] = None,
        text: str = None,
        lw: float = 1.0,
        del_ticks: bool = False,
    ):
        """Plot Doppler amplitude spectra for selected heights from a DFT block.

        Each selected height is rendered as a separate line, coloured by
        height from a continuous colourmap.  A legend identifies which
        colour corresponds to which height.  The Doppler bin axis is
        signed so that bin 0 = zero velocity (centre of the 128-bin range).

        Args:
            df: DataFrame from :meth:`DftExtractor.to_pandas`.
            xparam: Column for the Doppler-bin axis.
            yparam: Column for amplitude values.
            height_col: Column holding height in km.
            block_idx: DFT block (frequency step) to plot.
            selected_heights: Explicit list of height values (km) to plot.
                If ``None``, ``n_heights`` evenly spaced heights are chosen.
            n_heights: Number of heights to sample when ``selected_heights``
                is ``None``.
            cmap: Matplotlib colormap used to colour each height trace.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            ylim: ``[min, max]`` amplitude range.  Auto if ``None``.
            xlim: ``[min, max]`` Doppler-bin range.  Auto if ``None``.
            text: Optional annotation in the upper-left corner.
            lw: Line width.
            del_ticks: Suppress axis ticks when ``True``.
        """
        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)

        # Auto-select block with strongest signal if not specified
        if block_idx is None:
            block_idx = int(df.groupby("block_idx")[yparam].max().idxmax())

        sub = df[df["block_idx"] == block_idx].copy()
        if sub.empty:
            return

        # Annotate with frequency from header if available
        if "frequency_hz" in sub.columns:
            freq_mhz = sub["frequency_hz"].iloc[0] / 1e6
            freq_label = f"  {freq_mhz:.3f} MHz"
        else:
            freq_label = ""

        all_heights = np.sort(sub[height_col].unique())
        if selected_heights is None:
            idx = np.round(np.linspace(0, len(all_heights) - 1, n_heights)).astype(int)
            selected_heights = all_heights[idx]

        cmap_fn = plt.get_cmap(cmap)
        h_min, h_max = all_heights.min(), all_heights.max()

        for h in selected_heights:
            row = sub[sub[height_col] == h].sort_values(xparam)
            if row.empty:
                continue
            norm_h = (h - h_min) / max(h_max - h_min, 1)
            color = cmap_fn(norm_h)
            ax.plot(
                row[xparam],
                row[yparam],
                color=color,
                lw=lw,
                label=f"{h:.0f} km",
            )

        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylabel(ylabel, fontdict={"size": self.font_size})
        ax.set_title(f"Block {block_idx}{freq_label}", fontsize=self.font_size - 1)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.legend(
            title="Height",
            fontsize=self.font_size - 2,
            framealpha=0.7,
            title_fontsize=self.font_size - 2,
        )

        if text:
            ax.text(
                0.02,
                0.96,
                text,
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontdict={"size": self.font_size - 1},
            )
        if not del_ticks:
            ax.tick_params(axis="both", labelsize=self.font_size - 1)
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
        """Plot drift velocities with error bars.

        Expects `df` with `xparam` datetime and `yparam` + `error` columns. This
        is a simple helper commonly used by higher-level plotting wrappers.

        Args:
            df: Input DataFrame containing `xparam`, `yparam` and `error` columns.
            xparam: Column name for the x-axis (time).
            yparam: Column name for the y-axis (velocity).
            color: Matplotlib color for the series.
            error: Column name for the y-errors.
            ylabel: Y-axis label text.
            xlabel: X-axis label text.
            text: Optional text shown in the upper left of the plot.
            del_ticks: If True, remove x/y ticks from the axis.
            fmt: Matplotlib marker style for the series.
            lw: Line width for markers.
            alpha: Matplotlib alpha (transparency) for the series.
            zorder: Matplotlib z-order for layering.
            major_locator: Matplotlib date locator for major x-axis ticks.
            minor_locator: Matplotlib date locator for minor x-axis ticks.
            ylim: List of two floats defining the y-axis limits.
            xlim: List of two datetimes defining the x-axis limits. If None,
                the full range of `df[xparam]` is used.
        """
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
        font_size: float = 18,
    ):
        """Create a 3-row DVL velocity plot (Vx, Vy, Vz) with optional save.

        This static helper builds a SkySummaryPlots container and uses
        `plot_drift_velocities` to populate each subplot. It returns the
        plot object unless `fname` is provided (in which case it saves and
        closes the figure).

        Args:
            df: Input DataFrame containing `xparam`, `yparams` and `errors` columns.
            xparam: Column name for the x-axis (time).
            yparams: List of three column names for the y-axes (Vx, Vy, Vz).
            colors: List of three colors for the yparams.
            errors: List of three column names for the y-errors.
            labels: List of three strings for the y-axis labels.
            text: Optional text shown in the upper left of the first plot.
            del_ticks: If True, remove x/y ticks from the axes.
            fmt: Matplotlib marker style for all series.
            lw: Line width for markers.
            alpha: Matplotlib alpha (transparency) for all series.
            zorder: Matplotlib z-order for layering.
            major_locator: Matplotlib date locator for major x-axis ticks.
            minor_locator: Matplotlib date locator for minor x-axis ticks.
            ylim: List of two floats defining the y-axis limits.
            xlim: List of two datetimes defining the x-axis limits. If None,
                the full range of `df[xparam]` is used.
            fname: If provided, save the figure to this path.
            figsize: Figure size (width, height) in inches.
            draw_local_time: If True, convert `xparam` to local time.
        Returns:
            If `fname` is None, returns the `SkySummaryPlots` object containing
                the figure and axes. Otherwise, saves the figure to `fname` and
                returns None.
        """

        xparam = "local_" + xparam if draw_local_time else xparam
        dvlplot = SkySummaryPlots(
            figsize=figsize,
            nrows=3,
            ncols=1,
            subplot_kw=None,
            draw_local_time=draw_local_time,
            font_size=font_size,
        )
        for i, y, col, err, lab in zip(
            range(len(yparams)), yparams, colors, errors, labels
        ):
            text = text if text else ""
            date_txt = (
                f"{df[xparam].iloc[0].strftime('%d %b')}-{df[xparam].iloc[-1].strftime('%d %b, %Y')}"
                if df[xparam].iloc[0].day != df[xparam].iloc[-1].day
                else f"{df[xparam].iloc[-1].strftime('%d %b, %Y')}"
            )
            text = text + date_txt if i == 0 else None
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


class RsfIonogram(DigiPlots):
    """Plotting helpers for RSF-format ionograms.

    Provides methods to render RSF-style ionograms (frequency vs height) and
    convenience helpers for filtering and visualizing RSF amplitude data.

    This method inherits from `DigiPlots` and thus also provides methods and attributes including
    `get_axes`, `save`, `close`, and attributes like `fig_title`, `nrows`, `ncols`, `font_size`,
    `figsize`, `date`, `date_lims`, `subplot_kw`, and `draw_local_time`.
    """

    # Color palette matching Figure 3-8 legend in the DPS4D manual.
    # Order determines legend order.
    DIRECTION_COLORS = {
        "Vo-": "#8B0000",  # dark red      – O-pol, vertical, neg Doppler
        "Vo+": "#FF6B6B",  # salmon/pink   – O-pol, vertical, pos Doppler
        "X-": "#006400",  # dark green    – X-pol, vertical, neg Doppler
        "X+": "#90EE90",  # light green   – X-pol, vertical, pos Doppler
        "NNE": "#4169E1",  # royal blue    – NE azimuth direction
        "E": "#1E90FF",  # dodger blue   – SE azimuth direction
        "W": "#FFD700",  # gold          – SW azimuth direction
        "SSW": "#FFA500",  # orange        – S  azimuth direction
        "NNW": "#191970",  # midnight blue – NW azimuth direction
        "NoVal": "#808080",  # gray          – undetermined amplitude
    }

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

    def add_direction_ionogram(
        self,
        df: pd.DataFrame,
        xparam: str = "frequency_reading",
        yparam: str = "height",
        xlabel: str = "Frequency, MHz",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [80, 600],
        xlim: List[float] = [1, 22],
        xticks: List[float] = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
        text: str = None,
        del_ticks: bool = False,
        ms: float = 0.5,
        marker: str = "s",
        zorder: int = 2,
        lower_plimit: float = 5,
        dop_split: int = 4,
    ):
        """Add an RSF ionogram with echo directions color coded, mimicking Figure 3-8.

        Echoes are classified into ten direction/polarization categories and
        rendered with the color convention used in the Digisonde-4D manual
        (Figure 3-8):

        * **Vo- / Vo+** – O-polarization, vertical incidence, negative / positive
          Doppler (dark-red / salmon).
        * **X- / X+** – X-polarization, vertical incidence, negative / positive
          Doppler (dark-green / light-green).
        * **NNE / E / NNW** – northward / eastward / north-west directional echoes
          (blue shades).
        * **W / SSW** – westward / south-south-west directional echoes
          (yellow / orange).
        * **NoVal** – zero or below-threshold amplitude (gray).

        Vertical-incidence echoes correspond to rows where ``azm_directions``
        equals ``"N"`` (azimuth = 0°). Doppler sign is determined by comparing
        ``dop_num`` against ``dop_split`` (default 4, i.e. dop_num < 4 → negative).

        Args:
            df: DataFrame produced by :meth:`RsfExtractor.to_pandas`.
            xparam: Column name for frequency (Hz, converted internally to MHz).
            yparam: Column name for virtual height (km).
            xlabel: X-axis label text.
            ylabel: Y-axis label text.
            ylim: ``[min, max]`` y-axis limits in km.
            xlim: ``[min, max]`` x-axis limits in MHz (log10 scale applied).
            xticks: Tick locations in MHz displayed on the log10 x-axis.
            text: Optional annotation placed in the upper-left of the axis.
            del_ticks: If ``True``, suppress x/y axis ticks.
            ms: Scatter marker size.
            marker: Matplotlib marker style character.
            zorder: Matplotlib z-order for layering scatter points.
            lower_plimit: Minimum amplitude (dB) threshold; rows below are dropped.
            dop_split: ``dop_num`` threshold separating negative (``< dop_split``)
                from positive (``>= dop_split``) Doppler shift.
        """

        # Map the six azm_directions labels produced by RsfFreuencyGroup.setup()
        # to Figure 3-8 category names.  "N" (azimuth 0°) is vertical incidence;
        # pol + Doppler sign determine Vo±/X±.
        _AZM_TO_CATEGORY = {
            "NE": "NNE",
            "SE": "E",
            "SW": "W",
            "S": "SSW",
            "NW": "NNW",
        }

        def _classify(row):
            if row["amplitude"] <= 0:
                return "NoVal"
            azm = row["azm_directions"]
            if azm == "N":
                if row["pol"] == "O":
                    return "Vo-" if row["dop_num"] < dop_split else "Vo+"
                else:
                    return "X-" if row["dop_num"] < dop_split else "X+"
            return _AZM_TO_CATEGORY.get(azm, "NoVal")

        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)
        ax.set_xlim(np.log10(xlim))
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size})
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontdict={"size": self.font_size})

        df = df.copy()
        df[xparam] = df[xparam].astype(float) / 1e6  # Hz → MHz
        df = df[df[xparam] != 0.0]
        df["amplitude"] = df["amplitude"].replace(0, np.nan)
        df = df[df["amplitude"] >= lower_plimit]

        df["_category"] = df.apply(_classify, axis=1)

        legend_handles = []
        for label, color in RsfIonogram.DIRECTION_COLORS.items():
            subset = df[df["_category"] == label]
            if subset.empty:
                continue
            ax.scatter(
                np.log10(subset[xparam]),
                subset[yparam],
                c=color,
                s=ms,
                marker=marker,
                zorder=zorder,
            )
            legend_handles.append(
                mpatches.Patch(facecolor=color, edgecolor="none", label=label)
            )

        if not del_ticks:
            ax.set_xticks(np.log10(xticks))
            ax.set_xticklabels(xticks)

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=self.font_size - 1,
                framealpha=0.8,
                markerscale=4,
                handlelength=1.0,
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

    def add_directogram(
        self,
        df: pd.DataFrame,
        hparam: str = "height",
        dlim: List[float] = [-800, 800],
        ylim: Optional[List] = None,
        ylabel: str = "Time, UT",
        time_format: str = "%H:%M",
        lower_plimit: float = 5,
        dop_split: int = 4,
        ms: float = 1.5,
        marker: str = "s",
        zorder: int = 2,
        text: str = None,
        del_ticks: bool = False,
    ):
        """Add a directogram: ground distance (km) vs UT time, Figure 3-12 style.

        For each above-threshold oblique echo the signed ground distance to the
        plasma reflection point is computed per ionogram (grouped by timestamp):

        .. math::

            D_i = \\pm\\sqrt{H_i^2 - H_v^2}

        where *H_i* is the echo virtual height, *H_v* is the representative
        F-layer height estimated from the peak-amplitude vertical echo in that
        ionogram, and the sign is negative for west/northwest/south-southwest
        echoes and positive for northeast/east echoes.

        The result matches the Figure 3-12 layout:

        * **X-axis**: ground distance in km (← WEST … 0 … EAST →)
        * **Y-axis**: UT time, increasing downward (``ax.invert_yaxis()``)
        * **Centre line** (D = 0): vertical-incidence Vo± / X± echoes
        * **Left** (D < 0): W, NNW, SSW direction echoes
        * **Right** (D > 0): NNE, E direction echoes

        Works for a single ionogram (one timestamp) or a full day of ionograms
        concatenated from multiple :meth:`RsfExtractor.to_pandas` calls.

        Args:
            df: DataFrame from :meth:`RsfExtractor.to_pandas` (one or more
                ionograms concatenated). Must contain ``date``, ``height``,
                ``amplitude``, ``azm_directions``, ``pol``, ``dop_num``.
            hparam: Column name for virtual height (km) used to compute D_i.
            dlim: ``[min, max]`` x-axis limits for ground distance in km.
            ylim: Optional ``[t_min, t_max]`` datetime limits for the y-axis.
                  If ``None``, determined automatically from the data.
            ylabel: Y-axis label.
            time_format: ``strftime`` format for y-axis tick labels.
            lower_plimit: Minimum amplitude (dB) threshold; rows below dropped.
            dop_split: Doppler index threshold (< → neg Doppler, >= → pos).
            ms: Scatter marker size.
            marker: Matplotlib marker style character.
            zorder: Matplotlib z-order.
            text: Optional annotation placed in the upper-left of the axis.
            del_ticks: If ``True``, suppress axis ticks.
        """
        DIRECTION_COLORS = {
            "Vo-": "#8B0000",
            "Vo+": "#FF6B6B",
            "X-": "#006400",
            "X+": "#90EE90",
            "NNE": "#4169E1",
            "E": "#1E90FF",
            "W": "#FFD700",
            "SSW": "#FFA500",
            "NNW": "#191970",
        }
        _WEST = {"W", "SSW", "NNW"}
        _AZM_TO_CATEGORY = {"NE": "NNE", "SE": "E", "SW": "W", "S": "SSW", "NW": "NNW"}

        def _classify(row):
            if row["amplitude"] <= 0:
                return None
            azm = row["azm_directions"]
            if azm == "N":
                if row["pol"] == "O":
                    return "Vo-" if row["dop_num"] < dop_split else "Vo+"
                else:
                    return "X-" if row["dop_num"] < dop_split else "X+"
            return _AZM_TO_CATEGORY.get(azm)

        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks)
        ax.set_xlim(dlim)
        ax.set_xlabel(
            "← WEST          [km]          EAST →", fontdict={"size": self.font_size}
        )
        ax.set_ylabel(ylabel, fontdict={"size": self.font_size})
        ax.axvline(0, color="k", lw=0.8, ls="--", zorder=1)

        df = df.copy()
        df["amplitude"] = df["amplitude"].replace(0, np.nan)
        df = df[df["amplitude"] >= lower_plimit]
        df["_category"] = df.apply(_classify, axis=1)
        df = df[df["_category"].notna()]

        # Convert date to matplotlib float for y-axis
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["_t"] = mdates.date2num(df["date"].dt.to_pydatetime())

        # Per-ionogram H_v: group by timestamp, find H_v from peak vertical echo
        def _compute_D(grp):
            vert = grp[grp["azm_directions"] == "N"]
            H_v = (
                vert.loc[vert["amplitude"].idxmax(), hparam]
                if not vert.empty
                else grp[hparam].median()
            )
            H_i = grp[hparam].values
            D = np.sqrt(np.maximum(H_i**2 - H_v**2, 0))
            # Negate for west-side directions
            mask_west = grp["_category"].isin(_WEST)
            D = np.where(mask_west, -D, D)
            # Vertical echoes stay at D=0
            mask_vert = grp["_category"].isin({"Vo-", "Vo+", "X-", "X+"})
            D = np.where(mask_vert, 0.0, D)
            grp = grp.copy()
            grp["_D"] = D
            return grp

        df = df.groupby("_t", group_keys=False).apply(_compute_D)

        legend_handles = []
        for label, color in DIRECTION_COLORS.items():
            sub = df[df["_category"] == label]
            if sub.empty:
                continue
            ax.scatter(
                sub["_D"],
                sub["_t"],
                c=color,
                s=ms,
                marker=marker,
                zorder=zorder,
                alpha=0.8,
            )
            legend_handles.append(
                mpatches.Patch(facecolor=color, edgecolor="none", label=label)
            )

        # Y-axis: datetime formatting, inverted (time increases downward)
        ax.yaxis.set_major_formatter(mdates.DateFormatter(time_format))
        ax.yaxis.set_major_locator(mdates.AutoDateLocator())
        if ylim is not None:
            ax.set_ylim(
                mdates.date2num(pd.to_datetime(ylim[0]).to_pydatetime()),
                mdates.date2num(pd.to_datetime(ylim[1]).to_pydatetime()),
            )
        ax.invert_yaxis()

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=self.font_size - 1,
                framealpha=0.8,
            )

        ax.text(
            0.25,
            0.98,
            "WEST",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontdict={"size": self.font_size - 1},
            color="gray",
        )
        ax.text(
            0.75,
            0.98,
            "EAST",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontdict={"size": self.font_size - 1},
            color="gray",
        )

        if not del_ticks:
            ax.tick_params(axis="both", labelsize=self.font_size - 1)

        if text:
            ax.text(
                0.02,
                0.05,
                text,
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                fontdict={"size": self.font_size - 1},
            )
        return

    def add_sky_directogram(
        self,
        df: pd.DataFrame,
        xparam: str = "frequency_reading",
        yparam: str = "height",
        zparam: str = "amplitude",
        rlim: List[float] = [80, 600],
        cmap: str = "plasma",
        prange: List[float] = [5, 60],
        cbar_label: str = "Amplitude, dB",
        ms: float = 1.5,
        lower_plimit: float = 5,
        text: str = None,
    ):
        """Add a polar sky-map directogram of RSF echo arrival directions.

        The sky map uses a polar axis where the angular position (θ) is the
        echo arrival azimuth and the radial position (r) is the virtual height.
        Points are colored by amplitude. Compass-bearing convention: North at
        the top, clockwise increasing.

        Requires ``RsfIonogram(subplot_kw={'projection': 'polar'}, ...)``::

            r = RsfIonogram(subplot_kw={'projection': 'polar'}, ...)
            r.add_sky_directogram(df, ...)

        Args:
            df: DataFrame produced by :meth:`RsfExtractor.to_pandas`.
            xparam: Frequency column (Hz) – used for zero-frequency filtering.
            yparam: Virtual height column (km), mapped to the radial axis.
            zparam: Amplitude column (dB), mapped to color.
            rlim: ``[min, max]`` radial (height) limits in km.
            cmap: Matplotlib colormap for amplitude coloring.
            prange: ``[vmin, vmax]`` amplitude color range in dB.
            cbar_label: Colorbar label text.
            ms: Scatter marker size.
            lower_plimit: Minimum amplitude (dB) threshold.
            text: Optional title placed above the polar axes.
        """
        _AZM_TO_DEG = {"N": 0, "NE": 60, "SE": 120, "S": 180, "SW": 240, "NW": 300}
        _AZM_LABELS = {
            0: "N\n(Vert.)",
            60: "NNE",
            120: "E",
            180: "S",
            240: "W",
            300: "NNW",
        }

        utils.setsize(self.font_size)
        ax = self.get_axes(del_ticks=False)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(rlim)
        ax.set_ylabel(
            "Virtual Height, km",
            labelpad=30,
            fontdict={"size": self.font_size},
        )

        df = df.copy()
        df[xparam] = df[xparam].astype(float) / 1e6
        df = df[df[xparam] != 0.0]
        df[zparam] = df[zparam].replace(0, np.nan)
        df = df[df[zparam] >= lower_plimit]
        df["_theta_rad"] = np.deg2rad(df["azm_directions"].map(_AZM_TO_DEG).fillna(0))

        sc = ax.scatter(
            df["_theta_rad"],
            df[yparam],
            c=df[zparam],
            s=ms,
            cmap=cmap,
            vmin=prange[0],
            vmax=prange[1],
            alpha=0.7,
        )

        for deg, label in _AZM_LABELS.items():
            ax.text(
                np.deg2rad(deg),
                rlim[1] * 1.18,
                label,
                ha="center",
                va="center",
                fontsize=self.font_size - 2,
            )

        cbar = self.fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.7)
        cbar.set_label(cbar_label, fontsize=self.font_size - 1)
        ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=self.font_size - 2)

        if text:
            ax.set_title(text, pad=20, fontsize=self.font_size)
        return
