"""VIPIR NGI ionogram processing and visualisation.

This sub-package handles scaled ionogram data stored in VIPIR NGI format.
It provides data structures for individual echoes and complete datasets,
an autoscaling pipeline, an ionogram plotting class, and time-zone
conversion utilities.

:mod:`pynasonde.vipir.ngi.source`
    Core data structures and the ``DataSource`` orchestrator that loads
    NGI files, applies scaling, and optionally writes diagnostic plots.

:mod:`pynasonde.vipir.ngi.scale`
    Noise profiling and automatic frequency-trace scaling logic.

:mod:`pynasonde.vipir.ngi.plotlib`
    Ionogram rendering class (:class:`Ionogram`).

:mod:`pynasonde.vipir.ngi.utils`
    Shared utilities: :class:`TimeZoneConversion` (UTC ↔ local time),
    configuration helpers, and smoothing functions.

Exported names
--------------
:class:`Trace`
    Scaled echo trace from a single NGI file.

:class:`Dataset`
    Full NGI dataset holding all traces and metadata for one sounding.

:class:`DataSource`
    Orchestrator: loads NGI files, runs scaling, manages output.

:class:`NoiseProfile`
    Estimates background noise level as a function of virtual height.

:class:`AutoScaler`
    Fits parabolic models to NGI amplitude profiles for automatic scaling.

:class:`Ionogram`
    Matplotlib-based ionogram renderer for NGI data.

:class:`TimeZoneConversion`
    Converts ``datetime`` objects between UTC and a station's local
    timezone (derived from latitude/longitude via ``timezonefinder``).
"""

from pynasonde.vipir.ngi.plotlib import Ionogram
from pynasonde.vipir.ngi.scale import AutoScaler, NoiseProfile
from pynasonde.vipir.ngi.source import Dataset, DataSource, Trace
from pynasonde.vipir.ngi.utils import TimeZoneConversion

__all__ = [
    "Trace",
    "Dataset",
    "DataSource",
    "NoiseProfile",
    "AutoScaler",
    "Ionogram",
    "TimeZoneConversion",
]
