"""Digisonde instrument data parsers, datatypes, and plotting utilities.

This package provides all the tools needed to load, parse, and visualise
data from Digisonde DPS4D sounders.  It is organised into three
sub-packages:

:mod:`pynasonde.digisonde.parsers`
    One extractor class per file format: SAO, RSF, DFT, SBF, DVL, MMM,
    SKY, EDP, and ionogram image.

:mod:`pynasonde.digisonde.cadi`
    CADI MD2/MD4 binary reader/extractor utilities for raw detections and
    derived signal products.

:mod:`pynasonde.digisonde.datatypes`
    ``dataclass`` containers that mirror the on-disk binary/XML structures
    consumed and produced by the parsers.

:mod:`pynasonde.digisonde.raw`
    Raw IQ processing pipeline (complementary-code correlation, Doppler
    FFT) for DPS4D ``.bin`` recordings.

Top-level utility modules:
    :mod:`pynasonde.digisonde.digi_utils` — shared helpers (station CSV
    lookup, gridding, Matplotlib sizing).

:mod:`pynasonde.digisonde.digi_plots` — high-level plotting classes
    (:class:`DigiPlots`, :class:`SaoSummaryPlots`, :class:`SkySummaryPlots`,
    :class:`RsfIonogram`, :class:`CadiIonogram`).

Exported names (convenience re-exports)
----------------------------------------
Parsers:
    ``SaoExtractor``, ``RsfExtractor``, ``DftExtractor``, ``SbfExtractor``,
    ``DvlExtractor``, ``ModMaxExtractor``, ``SkyExtractor``,
    ``EdpExtractor``, ``IonogramImageExtractor``, ``CadiExtractor``

Plotting:
    ``DigiPlots``, ``SaoSummaryPlots``, ``SkySummaryPlots``,
    ``RsfIonogram``, ``CadiIonogram``

Utilities:
    ``to_namespace``, ``get_digisonde_info``, ``load_station_csv``,
    ``get_gridded_parameters``, ``parse_digisonde_datetime_token``,
    ``extract_station_code_from_filename``, ``merge_dicts_selected_keys``,
    ``flatten_dict``
"""

from pynasonde.digisonde.digi_plots import (
    CadiIonogram,
    DigiPlots,
    RsfIonogram,
    SaoSummaryPlots,
    SkySummaryPlots,
)
from pynasonde.digisonde.cadi import CadiExtractor, CadiReader
from pynasonde.digisonde.digi_utils import (
    apply_filename_metadata,
    extract_datetime_token_from_filename,
    extract_station_code_from_filename,
    flatten_dict,
    get_digisonde_info,
    get_gridded_parameters,
    load_station_csv,
    merge_dicts_selected_keys,
    parse_digisonde_datetime_token,
    to_namespace,
)
from pynasonde.digisonde.parsers import (
    DftExtractor,
    DvlExtractor,
    EdpExtractor,
    IonogramImageExtractor,
    ModMaxExtractor,
    RsfExtractor,
    SaoExtractor,
    SbfExtractor,
    SkyExtractor,
)

__all__ = [
    # Parsers
    "SaoExtractor",
    "RsfExtractor",
    "DftExtractor",
    "SbfExtractor",
    "DvlExtractor",
    "ModMaxExtractor",
    "SkyExtractor",
    "EdpExtractor",
    "IonogramImageExtractor",
    "CadiExtractor",
    "CadiReader",
    # Plotting
    "DigiPlots",
    "SaoSummaryPlots",
    "SkySummaryPlots",
    "RsfIonogram",
    "CadiIonogram",
    # Utilities
    "to_namespace",
    "apply_filename_metadata",
    "extract_datetime_token_from_filename",
    "extract_station_code_from_filename",
    "flatten_dict",
    "get_digisonde_info",
    "load_station_csv",
    "get_gridded_parameters",
    "merge_dicts_selected_keys",
    "parse_digisonde_datetime_token",
]
