"""RSF datatypes for Digisonde RSF-format files.

RSF and SBF share nearly identical header, frequency-group, and data-unit
structures. The common implementation lives in ``_rsf_sbf_base``; this module
keeps the public RSF class names and applies RSF-specific behavior such as
azimuth direction labels and MPA-based amplitude suppression.
"""

from dataclasses import dataclass
from typing import ClassVar, List

from pynasonde.digisonde.datatypes._rsf_sbf_base import (
    RsfSbfDataUnit,
    RsfSbfFreuencyGroup,
    RsfSbfHeader,
)


@dataclass
class RsfHeader(RsfSbfHeader):
    """Header describing an RSF data block."""


@dataclass
class RsfFreuencyGroup(RsfSbfFreuencyGroup):
    """RSF frequency group with azimuth direction labels."""

    zero_amplitude_below_mpa: ClassVar[bool] = True
    compute_azm_directions: ClassVar[bool] = True


@dataclass
class RsfDataUnit(RsfSbfDataUnit):
    """Single RSF block containing a header and frequency groups."""

    header: RsfHeader = None
    frequency_groups: List[RsfFreuencyGroup] = None


@dataclass
class RsfDataFile:
    """Container representing the contents of an RSF file."""

    rsf_data_units: List[RsfDataUnit] = None
