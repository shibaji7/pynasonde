"""VIPIR ionosonde data processing.

This package provides everything needed to work with VIPIR (Vertical
Incidence Pulsed Ionospheric Radar) data.  It is split into two
sub-packages:

:mod:`pynasonde.vipir.ngi`
    Scaled NGI ionogram data — data structures, autoscaling, plotting, and
    time-zone utilities.

:mod:`pynasonde.vipir.riq`
    Raw RIQ IQ data — binary parser, datatypes, and the Dynasonde-style
    seven-parameter echo extractor.

Exported names (convenience re-exports)
----------------------------------------
From NGI:
    ``Trace``, ``Dataset``, ``DataSource``,
    ``NoiseProfile``, ``AutoScaler``,
    ``Ionogram``, ``TimeZoneConversion``

From RIQ:
    ``Echo``, ``EchoExtractor``, ``IonogramFilter``, ``RiqDataset``
"""

from pynasonde.vipir.ngi import (
    AutoScaler,
    Dataset,
    DataSource,
    Ionogram,
    NoiseProfile,
    TimeZoneConversion,
    Trace,
)
from pynasonde.vipir.riq import Echo, EchoExtractor, IonogramFilter, RiqDataset

__all__ = [
    # NGI
    "Trace",
    "Dataset",
    "DataSource",
    "NoiseProfile",
    "AutoScaler",
    "Ionogram",
    "TimeZoneConversion",
    # RIQ
    "Echo",
    "EchoExtractor",
    "IonogramFilter",
    "RiqDataset",
]
