"""pynasonde — Python tools for ionosonde data processing.

This package provides parsers, datatypes, echo extractors, and plotting
utilities for two families of ionosonde instruments:

:mod:`pynasonde.digisonde`
    Parsers for Digisonde DPS4D file formats (SAO, RSF, DFT, SBF, DVL,
    MMM, SKY, EDP) and the raw IQ pipeline for ``.bin`` recordings.
    Includes high-level plotting classes (:class:`SaoSummaryPlots`,
    :class:`SkySummaryPlots`, :class:`RsfIonogram`).

:mod:`pynasonde.vipir`
    VIPIR ionosonde data tools.  Covers scaled NGI ionograms
    (:class:`DataSource`, :class:`Trace`) and raw RIQ IQ data
    (:class:`RiqDataset`, :class:`Echo`, :class:`EchoExtractor`).

:mod:`pynasonde.webhook`
    HTTP download helper (:class:`Webhook`) for NGI and RIQ files from
    the Wallops VIPIR archive.

Convenience re-exports
----------------------
>>> from pynasonde import SaoExtractor, RiqDataset, Echo, EchoExtractor
>>> from pynasonde import DataSource, Trace, Webhook
"""

__version__ = "1.2.0"

# Digisonde parsers and plotting
from pynasonde.digisonde import (
    DftExtractor,
    DigiPlots,
    DvlExtractor,
    EdpExtractor,
    IonogramImageExtractor,
    ModMaxExtractor,
    RsfExtractor,
    RsfIonogram,
    SaoExtractor,
    SaoSummaryPlots,
    SbfExtractor,
    SkyExtractor,
    SkySummaryPlots,
)

# VIPIR NGI and RIQ
from pynasonde.vipir import (
    AutoScaler,
    Dataset,
    DataSource,
    Echo,
    EchoExtractor,
    Ionogram,
    IonogramFilter,
    NoiseProfile,
    RiqDataset,
    TimeZoneConversion,
    Trace,
)

# Download utility
from pynasonde.webhook import Webhook

__all__ = [
    "__version__",
    # Digisonde parsers
    "SaoExtractor",
    "RsfExtractor",
    "DftExtractor",
    "SbfExtractor",
    "DvlExtractor",
    "ModMaxExtractor",
    "SkyExtractor",
    "EdpExtractor",
    "IonogramImageExtractor",
    # Digisonde plotting
    "DigiPlots",
    "SaoSummaryPlots",
    "SkySummaryPlots",
    "RsfIonogram",
    # VIPIR NGI
    "Trace",
    "Dataset",
    "DataSource",
    "NoiseProfile",
    "AutoScaler",
    "Ionogram",
    "TimeZoneConversion",
    # VIPIR RIQ
    "Echo",
    "EchoExtractor",
    "IonogramFilter",
    "RiqDataset",
    # Download
    "Webhook",
]
