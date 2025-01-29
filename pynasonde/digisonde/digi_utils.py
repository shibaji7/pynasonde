import importlib.resources
from types import SimpleNamespace

import numpy as np
from loguru import logger


def to_namespace(d: object) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_namespace(v) for v in d]
    else:
        return d


def setsize(size=8):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scienceplots

    logger.info(f"Invoking scienceplots: {scienceplots.__str__()}")
    # plt.style.use(["science", "ieee"])
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


def get_gridded_parameters(
    q, xparam, yparam, zparam, r=1, rounding=True, xparam_invalid="date"
):
    """ """
    import numpy as np

    plotParamDF = q[[xparam, yparam, zparam]]
    if rounding:
        if xparam != xparam_invalid:
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


def get_digisonde_info(code: str, fpath: str = None, long_key: str = "LONG") -> dict:
    stations = load_station_csv(fpath)
    stations[long_key] = np.where(
        stations[long_key] > 180, stations[long_key] - 360, stations[long_key]
    )
    station = stations[stations.URSI == code]
    station = station.to_dict("records")
    station = station[0] if len(station) > 0 else station
    return station


def load_station_csv(fpath: str = None) -> SimpleNamespace:
    import pandas as pd

    if fpath:
        logger.info(f"Loading from {fpath}")
        stations = pd.read_csv(fpath)
    else:
        with importlib.resources.path(
            "pynasonde", "digisonde_station_codes.csv"
        ) as fpath:
            logger.info(f"Loading from {fpath}")
            stations = pd.read_csv(fpath)
    return stations
