import importlib.resources
from types import SimpleNamespace

import toml
from loguru import logger


def setsize(size=8):
    pass

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
