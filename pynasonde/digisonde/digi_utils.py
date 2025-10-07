"""Utilities for the Digisonde subpackage.

This module contains small, frequently used helpers for the Digisonde
parsers and plotting code. Key functions include:

- ``to_namespace``: recursively convert dicts/lists to objects for
    attribute-style access.
- ``setsize``: apply consistent Matplotlib rcParams used by plotting
    helpers in this package.
- ``get_gridded_parameters``: aggregate DataFrame rows onto an X/Y grid
    suitable for pcolormesh rendering.
- ``load_station_csv`` / ``get_digisonde_info``: load bundled station
    metadata and look up station information.
- ``load_dtd_file``: helper to construct an lxml XMLParser optionally
    configured for DTD validation.

The functions here are lightweight and intended to be used by the
parsers and plotting helpers under ``pynasonde.digisonde``.
"""

import importlib.resources
from types import SimpleNamespace

import numpy as np
from loguru import logger
from lxml import etree


def to_namespace(d: object) -> SimpleNamespace:
    """Recursively convert mapping/list structures to SimpleNamespace.

    This helper turns nested dictionaries into `SimpleNamespace` objects so
    fields can be accessed with attribute syntax (``ns.field``) instead of
    dictionary indexing (``d['field']``). Lists are preserved but their
    elements are converted recursively.

    Args:
        d: A dict, list, or primitive value.

    Returns:
        A SimpleNamespace for dicts, a list with converted elements for
        lists, or the original value for primitives.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_namespace(v) for v in d]
    else:
        return d


def setsize(size=8):
    """Set plotting font sizes and family for Matplotlib (scienceplots).

    This convenience function applies a small set of rcParam changes used
    across the Digisonde plotting helpers to keep figures consistent. It
    also logs the invocation of the `scienceplots` package (if available).

    Args:
        size: Base font size for labels and ticks.
    """
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
    """Aggregate DataFrame parameters onto a 2D grid for pcolormesh.

    The function groups `q` by (`xparam`, `yparam`), averages `zparam`, and
    pivots the result into a 2-D grid suitable for use with
    ``matplotlib.axes.Axes.pcolormesh``. NaN values are masked so rendering
    is stable.

    Args:
        q: pandas DataFrame containing x/y/z columns.
        xparam: Column name to use for the x axis (typically datetime).
        yparam: Column name for the vertical axis (height).
        zparam: Column name for the colored parameter (e.g., frequency).
        r: Rounding precision applied to x/y when `rounding` is True.
        rounding: If True, round coordinates before grouping to reduce
            cardinality and produce a regular grid.
        xparam_invalid: Column name that should not be rounded (default
            'date' for datetime-type columns).

    Returns:
        X, Y, Z where X/Y are 2D meshgrid arrays and Z is a masked array of
        z-values with the same shape as X/Y.
    """
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
    """Lookup station metadata by URSI code.

    Reads the station CSV (via ``load_station_csv``), normalizes longitudes
    to the [-180, 180] range (if needed) and returns the first matching
    station record as a dictionary.

    Args:
        code: URSI station code to look up.
        fpath: Optional path to a custom station CSV. If omitted the bundled
            `digisonde_station_codes.csv` resource is used.
        long_key: Column name used for longitude values (default 'LONG').

    Returns:
        A dict representing the station metadata for `code`, or an empty
        list/dict if no station was found (matching previous behavior).
    """
    stations = load_station_csv(fpath)
    stations[long_key] = np.where(
        stations[long_key] > 180, stations[long_key] - 360, stations[long_key]
    )
    station = stations[stations.URSI == code]
    station = station.to_dict("records")
    station = station[0] if len(station) > 0 else station
    return station


def load_station_csv(fpath: str = None) -> SimpleNamespace:
    """Load Digisonde station metadata CSV into a pandas DataFrame.

    If `fpath` is omitted the function loads the packaged
    `digisonde_station_codes.csv` resource. The function logs which file it
    loaded and returns the resulting DataFrame.

    Args:
        fpath: Optional path to a CSV file with station metadata.

    Returns:
        A pandas DataFrame containing station metadata.
    """
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


def load_dtd_file(fpath: str = None) -> etree.XMLParser:
    """Create an lxml XMLParser optionally configured with DTD validation.

    If `fpath` is None the packaged `saoxml.dtd` resource is used. The
    returned parser will have `dtd_validation` enabled if a path is provided
    (or the resource resolves to a real path).

    Args:
        fpath: Optional path to a DTD file.

    Returns:
        An instance of ``lxml.etree.XMLParser`` configured for DTD
        validation.
    """
    if fpath is None:
        fpath = importlib.resources.path("pynasonde", "saoxml.dtd")
    logger.info(f"Loading from {str(fpath)}")
    parser = etree.XMLParser(dtd_validation=bool(fpath))
    return parser


def is_valid_xml_data_string(text):
    """Checks if a string contains only numbers, periods, and spaces.

    Args:
    text: The string to check.

    Returns:
    True if the string is valid, False otherwise.
    """
    for char in text:
        if not (char.isdigit() or char == "." or char == " "):
            return False
    return True
