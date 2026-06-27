"""Utilities for the Digisonde subpackage.

This module contains small, frequently used helpers for the Digisonde
parsers and plotting code. Key functions include:

- ``to_namespace``: recursively convert dicts/lists to objects for
    attribute-style access.
- ``merge_dicts_selected_keys`` / ``flatten_dict``: small dictionary helpers
    used when converting parsed nested products to tabular rows.
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

import glob
import importlib.resources
import datetime as dt
import os
import re
import struct
from functools import partial
from multiprocessing import Pool
from types import SimpleNamespace
from collections.abc import Mapping
from typing import BinaryIO, Callable, Iterable, Sequence, Union

import numpy as np
import pandas as pd
from loguru import logger
from lxml import etree
from tqdm import tqdm


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


def merge_dicts_selected_keys(
    base: Mapping | None,
    update: Mapping | None,
    keys: Sequence[str] | None = None,
) -> dict:
    """Return a shallow merge of two mappings with optional key selection.

    Args:
        base: Base mapping. ``None`` is treated as an empty mapping.
        update: Mapping to merge into ``base``. ``None`` is treated as empty.
        keys: Optional ordered list of keys to copy from ``update``.

    Returns:
        A new dictionary. Inputs are not modified.

    Raises:
        KeyError: If ``keys`` contains a name missing from ``update``.
    """
    merged = dict(base or {})
    source = dict(update or {})
    if keys is None:
        merged.update(source)
    else:
        merged.update({key: source[key] for key in keys})
    return merged


def flatten_dict(
    payload: Mapping,
    parent_key: str = "",
    sep: str = "_",
) -> dict:
    """Flatten a nested mapping into one dictionary.

    Nested keys are joined with ``sep``. Lists and other non-mapping values are
    preserved as values because parser flattening should not silently expand
    array-valued science products into ambiguous column names.
    """
    flattened = {}
    for key, value in payload.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value
    return flattened


def setsize(size=8):
    """Set plotting font sizes and family for Matplotlib.

    This convenience function applies a small set of rcParam changes used
    across the Digisonde plotting helpers to keep figures consistent. The
    ``scienceplots`` package is used when available but is not required.

    Args:
        size: Base font size for labels and ticks.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401

        logger.info(
            f"scienceplots available: {getattr(scienceplots, '__version__', 'unknown')}"
        )
    except ImportError:
        logger.debug("scienceplots not installed; skipping style import.")
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

    plotParamDF = q[[xparam, yparam, zparam]].copy()
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


def collect_files(
    folders: Iterable[str],
    exts: Union[str, Sequence[str]],
    unique: bool = True,
) -> list[str]:
    """Collect files matching one or more glob patterns under folders.

    Args:
        folders: Directories to search.
        exts: One glob pattern or a sequence of glob patterns.
        unique: If True, de-duplicate and sort matches.

    Returns:
        Sorted file paths matching the requested patterns.
    """
    patterns = [exts] if isinstance(exts, str) else list(exts)
    files: list[str] = []
    for folder in folders:
        found: list[str] = []
        for pattern in patterns:
            found.extend(glob.glob(os.path.join(folder, pattern)))
        found = sorted(set(found) if unique else found)
        logger.info(f"Searching {folder} for {patterns}: found {len(found)} files")
        files.extend(found)
    return sorted(set(files) if unique else files)


def load_files_to_dataframe(
    folders: Iterable[str],
    exts: Union[str, Sequence[str]],
    extractor: Callable,
    n_procs: int = 1,
    extractor_kwargs: dict | None = None,
    ignore_index: bool = True,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Load matched files with ``extractor`` and concatenate DataFrame results.

    This centralizes the common Digisonde parser pattern:
    find files, run a per-file extraction function sequentially or with a
    process pool, drop empty results, and return one concatenated DataFrame.

    Args:
        folders: Directories to search for matching files.
        exts: One glob pattern or a sequence of glob patterns.
        extractor: Callable that accepts one file path and returns a DataFrame.
        n_procs: Number of worker processes. Values <= 1 run sequentially.
        extractor_kwargs: Optional keyword arguments passed to ``extractor``.
        ignore_index: Passed to ``pandas.concat`` when combining frames.
        drop_empty: If True, omit ``None`` and empty DataFrame results.

    Returns:
        Concatenated DataFrame, or an empty DataFrame when nothing is loaded.
    """
    files = collect_files(folders=folders, exts=exts)
    if not files:
        logger.warning("No files found for requested folders/patterns.")
        return pd.DataFrame()

    worker = partial(extractor, **(extractor_kwargs or {}))
    if n_procs <= 1:
        frames = [worker(file) for file in tqdm(files)]
    else:
        with Pool(n_procs) as pool:
            frames = list(tqdm(pool.imap(worker, files), total=len(files)))

    if drop_empty:
        frames = [
            frame
            for frame in frames
            if frame is not None and not getattr(frame, "empty", False)
        ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=ignore_index)


def extract_station_code_from_filename(filename: str) -> str:
    """Return the leading station token from a Digisonde-style filename."""
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    return stem.split("_")[0] if stem else ""


def extract_datetime_token_from_filename(filename: str) -> str:
    """Return the trailing filename token likely containing date/time."""
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    parts = stem.split("_")
    return parts[-1] if parts else stem


def parse_digisonde_datetime_token(token: str) -> dt.datetime | None:
    """Parse common Digisonde filename timestamp tokens.

    Supported forms:
    - ``YYYYmmddHHMMSS`` for one scan
    - ``YYYYDDDHHMMSS`` for one scan
    - ``YYYYmmdd(DDD)`` for a day file
    - ``YYYYmmdd`` for a day file
    - ``YYYYDDD`` for a day file
    """
    token = str(token).strip()

    if re.fullmatch(r"\d{14}", token):
        return dt.datetime.strptime(token, "%Y%m%d%H%M%S")

    if re.fullmatch(r"\d{13}", token):
        return dt.datetime.strptime(token, "%Y%j%H%M%S")

    match = re.fullmatch(r"(?P<ymd>\d{8})\((?P<doy>\d{3})\)", token)
    if match:
        parsed = dt.datetime.strptime(match.group("ymd"), "%Y%m%d")
        doy = int(match.group("doy"))
        if parsed.timetuple().tm_yday != doy:
            logger.warning(
                f"Filename day-of-year mismatch: {match.group('ymd')} has "
                f"DOY {parsed.timetuple().tm_yday}, token says {doy}."
            )
        return parsed

    if re.fullmatch(r"\d{8}", token):
        return dt.datetime.strptime(token, "%Y%m%d")

    if re.fullmatch(r"\d{7}", token):
        return dt.datetime.strptime(token, "%Y%j")

    return None


def apply_filename_metadata(
    target: object,
    filename: str,
    extract_time_from_name: bool = False,
    extract_stn_from_name: bool = False,
    station_code_always: bool = False,
    load_station_info: bool = True,
    compute_local_time: bool = True,
) -> dict:
    """Attach common filename-derived metadata to an extractor instance.

    The helper preserves the attributes already used by Digisonde parsers:
    ``stn_code``, ``date``, ``stn_info``, ``local_timezone_converter``, and
    ``local_time``.

    Args:
        target: Extractor instance that receives metadata attributes.
        filename: Source file path used to derive station and time tokens.
        extract_time_from_name: If True, parse and attach ``target.date``.
        extract_stn_from_name: If True, parse and attach station metadata.
        station_code_always: If True, always attach ``target.stn_code``.
        load_station_info: If True, look up station latitude/longitude.
        compute_local_time: If True, compute local time when station and date
            metadata are available.

    Returns:
        Dictionary containing parsed station, datetime, and local-time metadata.
    """
    metadata = {
        "station": extract_station_code_from_filename(filename),
        "datetime_token": extract_datetime_token_from_filename(filename),
        "datetime": None,
        "stn_info": None,
        "local_datetime": None,
    }

    if station_code_always or extract_stn_from_name:
        target.stn_code = metadata["station"]

    if extract_time_from_name:
        parsed = parse_digisonde_datetime_token(metadata["datetime_token"])
        metadata["datetime"] = parsed
        target.date = parsed
        if parsed is None:
            logger.warning(
                f"Could not parse datetime from filename token "
                f"'{metadata['datetime_token']}' in '{filename}'"
            )
        else:
            logger.info(f"Date: {parsed}")

    if extract_stn_from_name and load_station_info:
        if not hasattr(target, "stn_code"):
            target.stn_code = metadata["station"]
        target.stn_info = get_digisonde_info(target.stn_code)
        metadata["stn_info"] = target.stn_info
        date = getattr(target, "date", None)
        if (
            compute_local_time
            and date is not None
            and isinstance(target.stn_info, dict)
            and "LAT" in target.stn_info
            and "LONG" in target.stn_info
        ):
            from pynasonde.vipir.ngi.utils import TimeZoneConversion

            target.local_timezone_converter = TimeZoneConversion(
                lat=target.stn_info["LAT"], long=target.stn_info["LONG"]
            )
            target.local_time = target.local_timezone_converter.utc_to_local_time(
                [date]
            )[0]
            metadata["local_datetime"] = target.local_time
        logger.info(f"Station code: {target.stn_code}; {target.stn_info}")

    return metadata


RSF_SBF_IONOGRAM_SETTINGS = {
    "128": dict(number_freq_blocks=15, number_range_bins=128, byte_length=262),
    "256": dict(number_freq_blocks=8, number_range_bins=249, byte_length=504),
    "512": dict(number_freq_blocks=4, number_range_bins=501, byte_length=1008),
}


def as_uint8(value: int) -> int:
    """Validate and return an integer byte value."""
    value = int(value)
    if value < 0 or value > 0xFF:
        raise ValueError(f"Expected byte value in [0, 255], got {value}.")
    return value


def split_packed_byte(value: int, high_bits: int, low_bits: int) -> tuple[int, int]:
    """Split one byte into high and low bit fields.

    Examples:
        ``split_packed_byte(0b00101011, 5, 3)`` returns ``(5, 3)``.
        ``split_packed_byte(0xAB, 4, 4)`` returns ``(10, 11)``.
    """
    if high_bits + low_bits != 8:
        raise ValueError("high_bits + low_bits must equal 8.")
    value = as_uint8(value)
    high = (value >> low_bits) & ((1 << high_bits) - 1)
    low = value & ((1 << low_bits) - 1)
    return high, low


def unpack_5_3_byte(value: int) -> list[int]:
    """Unpack a byte into a 5-bit high field and 3-bit low field."""
    return list(split_packed_byte(value, 5, 3))


def unpack_4_4_byte(value: int) -> tuple[int, int]:
    """Unpack a byte into high and low nibbles."""
    return split_packed_byte(value, 4, 4)


def low_nibble(value: int) -> int:
    """Return the low 4-bit nibble from a byte."""
    return unpack_4_4_byte(value)[1]


def unpack_bcd_byte(value: int, format: str = "int") -> int | tuple[int, int]:
    """Decode a packed BCD byte.

    Args:
        value: Byte encoded as two decimal nibbles.
        format: ``"int"`` returns ``10 * high + low``; ``"tuple"`` returns
            ``(high, low)``.
    """
    high, low = unpack_4_4_byte(value)
    if format == "int":
        return 10 * high + low
    if format == "tuple":
        return high, low
    raise ValueError("Invalid format specified. Use 'int' or 'tuple'.")


def read_exact(stream: BinaryIO, n_bytes: int) -> bytes:
    """Read exactly ``n_bytes`` or raise EOFError."""
    raw = stream.read(n_bytes)
    if len(raw) != n_bytes:
        raise EOFError(f"Expected {n_bytes} bytes, got {len(raw)}.")
    return raw


def read_u8(stream: BinaryIO) -> int:
    """Read one unsigned byte."""
    return read_exact(stream, 1)[0]


def read_i8(stream: BinaryIO) -> int:
    """Read one signed byte."""
    return struct.unpack("b", read_exact(stream, 1))[0]


def read_u16(stream: BinaryIO, endian: str = "<") -> int:
    """Read one unsigned 16-bit integer."""
    return struct.unpack(f"{endian}H", read_exact(stream, 2))[0]


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


def load_station_csv(fpath: str = None) -> pd.DataFrame:
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
        ref = importlib.resources.files("pynasonde").joinpath(
            "digisonde_station_codes.csv"
        )
        logger.info(f"Loading from {ref}")
        stations = pd.read_csv(ref)
    return stations


def load_dtd_file(fpath: str = None) -> etree.XMLParser:
    """Create an lxml XMLParser configured with DTD validation.

    If `fpath` is None the packaged `saoxml.dtd` resource is used.

    Args:
        fpath: Optional path to a DTD file.

    Returns:
        An instance of ``lxml.etree.XMLParser`` configured for DTD
        validation.
    """
    if fpath is None:
        ref = importlib.resources.files("pynasonde").joinpath("saoxml.dtd")
        fpath = str(ref)
    logger.info(f"Loading from {fpath}")
    parser = etree.XMLParser(dtd_validation=True)
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
