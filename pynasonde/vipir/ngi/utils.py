"""Shared utilities for VIPIR NGI processing (time conversion, smoothing, etc.)."""

import datetime as dt
import importlib.resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytz
import toml
from loguru import logger
from timezonefinder import TimezoneFinder


class TimeZoneConversion:
    """Convert timestamps between UTC and a station's local timezone."""

    def __init__(
        self,
        local_tz: str = None,
        lat: float = 37.8815,
        long: float = -75.4374,
    ):
        """Initialize the converter with either a timezone name or coordinates.

        Args:
            local_tz: Explicit timezone name; inferred from lat/lon when None.
            lat: Station latitude used when deriving the timezone.
            long: Station longitude used when deriving the timezone.
        """
        self.local_tz = local_tz
        tf = TimezoneFinder()
        if (long is not None) and (lat is not None):
            self.local_tz = tf.timezone_at(lng=long, lat=lat)
        logger.info(f"Local Time: {self.local_tz}")
        self.utc_zone, self.local_zone = pytz.timezone("UTC"), pytz.timezone(
            self.local_tz
        )
        return

    def utc_to_local_time(self, dates):
        """Translate an iterable of UTC datetimes to the configured local zone.

        Args:
            dates: Iterable of naive or timezone-aware datetime objects.

        Returns:
            List of localized datetimes adjusted to `self.local_zone`.
        """
        date = pd.to_datetime(dates[0])
        tdiff_hr = (
            self.local_zone.localize(date)
            - self.utc_zone.localize(date).astimezone(self.local_zone)
        ).seconds / 3600
        dates = [d - dt.timedelta(hours=tdiff_hr) for d in dates]
        return dates


def to_local_time(dates: list, tz1, tz2):
    """Adjust naive datetimes from timezone `tz1` into `tz2`.

    Args:
        dates: Iterable of datetime objects.
        tz1: Origin timezone (pytz timezone).
        tz2: Destination timezone (pytz timezone).

    Returns:
        List of datetime objects converted to `tz2`.
    """
    date = pd.to_datetime(dates[0])
    tdiff_hr = (tz2.localize(date) - tz1.localize(date).astimezone(tz2)).seconds / 3600
    dates = [d - dt.timedelta(hours=tdiff_hr) for d in dates]
    return dates


def remove_outliers(o: pd.DataFrame, pname: str, quantiles=[0.05, 0.95]):
    """Trim rows where `pname` falls outside the provided quantile window.

    Args:
        o: Input dataframe.
        pname: Column name used to evaluate quantiles.
        quantiles: Lower and upper quantile thresholds.

    Returns:
        Filtered dataframe restricted to the quantile window.
    """
    lower_bound = o[pname].quantile(quantiles[0])
    upper_bound = o[pname].quantile(quantiles[1])
    o = o[(o[pname] >= lower_bound) & (o[pname] <= upper_bound)]
    return o


def running_median(arr, window=21):
    """Compute a moving median with the given window size.

    Args:
        arr: Sequence of values.
        window: Sliding window length.

    Returns:
        List of median values aligned with the input sequence.
    """
    return pd.Series(arr).rolling(window=window, min_periods=1).median().tolist()


def smooth(x, window_len=11, window="hanning"):
    """Apply a windowed smoothing convolution to a 1-D array.

    Args:
        x: Input NumPy array (1-D).
        window_len: Length of the smoothing window.
        window: Window function name (flat, hanning, hamming, bartlett, blackman).

    Returns:
        Smoothed NumPy array with edge handling.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    if window == "flat":
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")
    y = np.convolve(w / w.sum(), s, mode="valid")
    d = window_len - 1
    y = y[int(d / 2) : -int(d / 2)]
    return y


def setsize(size=8):
    """Configure matplotlib/scienceplots defaults for a consistent style.

    Args:
        size: Base font size applied to the plot configuration.
    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use(["science", "ieee"])
    plt.rcParams.update(
        {
            "text.usetex": False,
        }
    )
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


def to_namespace(d: object) -> SimpleNamespace:
    """Recursively convert dicts/lists into `SimpleNamespace` instances.

    Args:
        d: Arbitrary nested structure of dicts/lists/primitives.

    Returns:
        Equivalent structure with dicts converted to `SimpleNamespace`.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_namespace(v) for v in d]
    else:
        return d


def load_toml(fpath: str = None) -> SimpleNamespace:
    """Load a TOML configuration file into nested `SimpleNamespace` objects.

    Args:
        fpath: Optional explicit path to a TOML file; defaults to bundled config.

    Returns:
        SimpleNamespace representation of the parsed TOML.
    """
    if fpath:
        logger.info(f"Loading from {fpath}")
        cfg = to_namespace(toml.load(fpath))
    else:
        with importlib.resources.path("pynasonde", "config.toml") as config_path:
            logger.info(f"Loading from {config_path}")
            cfg = to_namespace(toml.load(config_path))
    return cfg


def get_color_by_index(index, total_indices, cmap_name="viridis"):
    """Pick a color from a colormap using an index within ``[0, total)``.

    Args:
        index: Position within the available color slots.
        total_indices: Total number of available slots.
        cmap_name: Matplotlib colormap name.

    Returns:
        RGBA tuple sampled from the requested colormap.
    """
    import matplotlib.pyplot as plt

    # Normalize the index to be between 0 and 1
    norm_index = index / total_indices

    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Return the color for the given index
    return cmap(norm_index)


def get_gridded_parameters(q, xparam, yparam, zparam, r=1, rounding=True):
    """Reshape scattered parameter samples onto an evenly-spaced grid.

    Args:
        q: Dataframe containing the source columns.
        xparam: Column name used for the X dimension.
        yparam: Column name used for the Y dimension.
        zparam: Column name containing the values to grid.
        r: Rounding precision applied before grouping.
        rounding: Whether to round values prior to pivoting.

    Returns:
        Tuple of `(X, Y, Z)` NumPy arrays suitable for contour/mesh plots.
    """
    import numpy as np

    plotParamDF = q[[xparam, yparam, zparam]]
    if rounding:
        if xparam != "time":
            plotParamDF[xparam] = np.round(plotParamDF[xparam], r)
        plotParamDF[yparam] = np.round(plotParamDF[yparam], r)
    plotParamDF = plotParamDF.groupby([xparam, yparam]).mean().reset_index()
    plotParamDF = plotParamDF[[xparam, yparam, zparam]].pivot(
        index=xparam, columns=yparam
    )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y = np.meshgrid(x, y)
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
        np.isnan(plotParamDF[zparam].values), plotParamDF[zparam].values
    )
    return X, Y, Z
