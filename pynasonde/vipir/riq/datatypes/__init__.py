"""Dataclasses for VIPIR RIQ binary record structures.

This sub-package mirrors the on-disk layout of VIPIR RIQ files as Python
dataclasses, making the parsed data accessible with attribute syntax.

:mod:`pynasonde.vipir.riq.datatypes.sct`
    Sounding Configuration Table (SCT) datatypes.  The SCT appears once
    per RIQ file and describes the sounder configuration (station, timing,
    frequency schedule, receiver positions, and monitor data).

:mod:`pynasonde.vipir.riq.datatypes.pct`
    Pulse Coherency Table (PCT) and ionogram datatypes.  One PCT is
    stored per transmitted pulse and holds the raw I/Q samples.

:mod:`pynasonde.vipir.riq.datatypes.default_factory`
    Field descriptor list (``SCT_default_factory``) consumed by
    :func:`~pynasonde.vipir.riq.datatypes.sct.read_dtype` to
    deserialize SCT records with NumPy structured arrays.

Exported names
--------------
From SCT:
    ``SctType``, ``StationType``, ``TimingType``, ``FrequencyType``,
    ``RecieverType``, ``ExciterType``, ``MonitorType``

From PCT:
    ``Ionogram``, ``PctType``
"""

from pynasonde.vipir.riq.datatypes.pct import Ionogram, PctType
from pynasonde.vipir.riq.datatypes.sct import (
    ExciterType,
    FrequencyType,
    MonitorType,
    RecieverType,
    SctType,
    StationType,
    TimingType,
)

__all__ = [
    # PCT
    "Ionogram",
    "PctType",
    # SCT
    "SctType",
    "StationType",
    "TimingType",
    "FrequencyType",
    "RecieverType",
    "ExciterType",
    "MonitorType",
]
