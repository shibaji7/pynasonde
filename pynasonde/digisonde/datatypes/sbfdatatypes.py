"""SBF datatypes for Digisonde SBF-format files.

SBF and RSF share nearly identical header, frequency-group, and data-unit
structures. The common implementation lives in ``_rsf_sbf_base``; this module
keeps the public SBF class names and SBF-specific container field names.
"""

from dataclasses import dataclass
from typing import ClassVar, List

from pynasonde.digisonde.datatypes._rsf_sbf_base import (
    RsfSbfDataUnit,
    RsfSbfFreuencyGroup,
    RsfSbfHeader,
)


@dataclass
class SbfHeader(RsfSbfHeader):
    """Header describing an SBF data block."""


@dataclass
class SbfFreuencyGroup(RsfSbfFreuencyGroup):
    """SBF frequency group."""

    zero_amplitude_below_mpa: ClassVar[bool] = False
    compute_azm_directions: ClassVar[bool] = False


@dataclass
class SbfDataUnit(RsfSbfDataUnit):
    """Single SBF block containing a header and frequency groups."""

    header: SbfHeader = None
    frequency_groups: List[SbfFreuencyGroup] = None


@dataclass
class SbfDataFile:
    """Container representing the contents of an SBF file."""

    sbf_data_units: List[SbfDataUnit] = None
