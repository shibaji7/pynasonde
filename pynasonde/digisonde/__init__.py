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

# Names resolve lazily (PEP 562) so importing one parser does not load the
# plotting stack (and vice versa).  ``from pynasonde.digisonde import X`` is unchanged.
_LAZY = {
    "CadiIonogram": "pynasonde.digisonde.digi_plots",
    "DigiPlots": "pynasonde.digisonde.digi_plots",
    "RsfIonogram": "pynasonde.digisonde.digi_plots",
    "SaoSummaryPlots": "pynasonde.digisonde.digi_plots",
    "SkySummaryPlots": "pynasonde.digisonde.digi_plots",
    "CadiExtractor": "pynasonde.digisonde.cadi",
    "CadiReader": "pynasonde.digisonde.cadi",
    "apply_filename_metadata": "pynasonde.digisonde.digi_utils",
    "extract_datetime_token_from_filename": "pynasonde.digisonde.digi_utils",
    "extract_station_code_from_filename": "pynasonde.digisonde.digi_utils",
    "flatten_dict": "pynasonde.digisonde.digi_utils",
    "get_digisonde_info": "pynasonde.digisonde.digi_utils",
    "get_gridded_parameters": "pynasonde.digisonde.digi_utils",
    "load_station_csv": "pynasonde.digisonde.digi_utils",
    "merge_dicts_selected_keys": "pynasonde.digisonde.digi_utils",
    "parse_digisonde_datetime_token": "pynasonde.digisonde.digi_utils",
    "to_namespace": "pynasonde.digisonde.digi_utils",
    "DftExtractor": "pynasonde.digisonde.parsers",
    "DvlExtractor": "pynasonde.digisonde.parsers",
    "EdpExtractor": "pynasonde.digisonde.parsers",
    "IonogramImageExtractor": "pynasonde.digisonde.parsers",
    "ModMaxExtractor": "pynasonde.digisonde.parsers",
    "RsfExtractor": "pynasonde.digisonde.parsers",
    "SaoExtractor": "pynasonde.digisonde.parsers",
    "SbfExtractor": "pynasonde.digisonde.parsers",
    "SkyExtractor": "pynasonde.digisonde.parsers",
}

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


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
