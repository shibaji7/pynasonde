"""Raw IQ processing pipeline for DPS4D Digisonde recordings.

This sub-package provides Python translations of the Julia ``Digisonde``
and ``IQReader`` modules.  It reads one-second ``.bin`` files produced by
the DPS4D receiver, applies complementary-code correlation and Doppler
FFT, and assembles the result into a range–frequency power grid that can
be written to NetCDF.

Exported names
--------------
:class:`IQStream`
    Reads and represents a single one-second binary IQ file.  Handles
    filename parsing, sample deinterleaving, and complex-sample access.

:class:`IonogramResult`
    Container for the processed ionogram power grid (frequency × range)
    computed by :func:`process`.

:func:`process`
    End-to-end sounding pipeline: reads IQ files, performs
    complementary-code correlation, Doppler-integrates pulses, and writes
    results.

:class:`RawPlots`
    Base figure/axes manager for plotting raw IQ-derived ionograms.

:class:`AFRLPlots`
    :class:`RawPlots` subclass with AFRL/VIPIR-style ionogram layouts.
"""

from pynasonde.digisonde.raw.iq_reader import IQStream
from pynasonde.digisonde.raw.raw_parse import IonogramResult, process
from pynasonde.digisonde.raw.raw_plots import AFRLPlots, RawPlots

__all__ = [
    "IQStream",
    "IonogramResult",
    "process",
    "RawPlots",
    "AFRLPlots",
]
