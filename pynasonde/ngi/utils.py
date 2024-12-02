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

    def __init__(
        self,
        local_tz: str = None,
        lat: float = 37.8815,
        long: float = -75.4374,
    ):
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
        date = pd.to_datetime(dates[0])
        tdiff_hr = (
            self.local_zone.localize(date)
            - self.utc_zone.localize(date).astimezone(self.local_zone)
        ).seconds / 3600
        dates = [d - dt.timedelta(hours=tdiff_hr) for d in dates]
        return dates


def to_local_time(dates: list, tz1, tz2):
    date = pd.to_datetime(dates[0])
    tdiff_hr = (tz2.localize(date) - tz1.localize(date).astimezone(tz2)).seconds / 3600
    dates = [d - dt.timedelta(hours=tdiff_hr) for d in dates]
    return dates


def remove_outliers(o: pd.DataFrame, pname: str, quantiles=[0.05, 0.95]):
    lower_bound = o[pname].quantile(quantiles[0])
    upper_bound = o[pname].quantile(quantiles[1])
    o = o[(o[pname] >= lower_bound) & (o[pname] <= upper_bound)]
    return o


def running_median(arr, window=21):
    return pd.Series(arr).rolling(window=window, min_periods=1).median().tolist()


def smooth(x, window_len=11, window="hanning"):
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


def to_namespace(d: object) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_namespace(v) for v in d]
    else:
        return d


def load_toml(fpath: str = None) -> SimpleNamespace:
    if fpath:
        logger.info(f"Loading from {fpath}")
        cfg = to_namespace(toml.load(fpath))
    else:
        with importlib.resources.path("pynasonde", "config.toml") as config_path:
            logger.info(f"Loading from {config_path}")
            cfg = to_namespace(toml.load(config_path))
    return cfg


def get_color_by_index(index, total_indices, cmap_name="viridis"):
    import matplotlib.pyplot as plt

    # Normalize the index to be between 0 and 1
    norm_index = index / total_indices

    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Return the color for the given index
    return cmap(norm_index)


def get_gridded_parameters(q, xparam, yparam, zparam, r=1, rounding=True):
    """ """
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
