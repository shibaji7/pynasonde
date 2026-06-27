"""Parsers and signal-processing helpers for VIPIR RIQ data files.

This sub-package converts raw VIPIR RIQ binary files into structured Python
objects and pandas/xarray datasets.

:mod:`pynasonde.vipir.riq.parsers.read_riq`
    Core RIQ reader.  Provides :class:`RiqDataset` (file-level container),
    :class:`Pulset` (group of pulses at a single frequency), and the
    amplitude-based echo detection helpers
    :func:`find_thresholds`, :func:`remove_morphological_noise`, and
    :func:`adaptive_gain_filter`.

Exported names
--------------
:class:`Pulset`
    Container for ``pulse_count`` PCTs (pulse coherency tables) sounded
    at the same transmit frequency.

:class:`RiqDataset`
    Top-level dataset object produced by parsing a single ``.RIQ`` file.
    Holds one :class:`~pynasonde.vipir.riq.datatypes.sct.SctType` and a
    list of :class:`Pulset` objects.

:func:`find_thresholds`
    Estimate per-gate SNR thresholds from the amplitude distribution.

:func:`remove_morphological_noise`
    Morphological opening on the amplitude grid to remove isolated speckle.

:func:`adaptive_gain_filter`
    Suppress gain artefacts by normalising against local range statistics.

:class:`IonogramFilter`
    Multi-stage coherent filter for echo clouds from one or more soundings.
    Applies RFI blanking, EP filter, multi-hop removal, DBSCAN clustering,
    and temporal coherence (multi-sounding).
"""

from pynasonde.vipir.riq.parsers.filter import IonogramFilter
from pynasonde.vipir.riq.parsers.read_riq import (
    Pulset,
    RiqDataset,
    adaptive_gain_filter,
    find_thresholds,
    remove_morphological_noise,
)

__all__ = [
    "Pulset",
    "RiqDataset",
    "find_thresholds",
    "remove_morphological_noise",
    "adaptive_gain_filter",
    "IonogramFilter",
]
