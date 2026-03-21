"""Dataclasses for Digisonde binary and XML record structures.

This sub-package defines lightweight ``dataclass`` containers that mirror
the on-disk layouts used by the various Digisonde file formats (SBF, RSF,
MMM, DFT, SAO XML).  They are consumed by the parsers in
:mod:`pynasonde.digisonde.parsers` and can be used directly by downstream
analysis code that needs structured, typed access to parsed fields.

Exported names
--------------
SBF format:
    ``SbfHeader``, ``SbfFreuencyGroup``, ``SbfDataUnit``, ``SbfDataFile``

RSF format:
    ``RsfHeader``, ``RsfFreuencyGroup``, ``RsfDataUnit``, ``RsfDataFile``

MMM / ModMax format:
    ``ModMaxHeader``, ``ModMaxFreuencyGroup``, ``ModMaxDataUnit``

DFT format:
    ``SubCaseHeader``, ``DftHeader``, ``DopplerSpectra``,
    ``DopplerSpectralBlock``

SAO XML format:
    ``URSI``, ``SAORecord``, ``SAORecordList``
"""

from pynasonde.digisonde.datatypes.dftdatatypes import (
    DftHeader,
    DopplerSpectra,
    DopplerSpectralBlock,
    SubCaseHeader,
)
from pynasonde.digisonde.datatypes.mmmdatatypes import (
    ModMaxDataUnit,
    ModMaxFreuencyGroup,
    ModMaxHeader,
)
from pynasonde.digisonde.datatypes.rsfdatatypes import (
    RsfDataFile,
    RsfDataUnit,
    RsfFreuencyGroup,
    RsfHeader,
)
from pynasonde.digisonde.datatypes.saoxmldatatypes import URSI, SAORecord, SAORecordList
from pynasonde.digisonde.datatypes.sbfdatatypes import (
    SbfDataFile,
    SbfDataUnit,
    SbfFreuencyGroup,
    SbfHeader,
)

__all__ = [
    # SBF
    "SbfHeader",
    "SbfFreuencyGroup",
    "SbfDataUnit",
    "SbfDataFile",
    # RSF
    "RsfHeader",
    "RsfFreuencyGroup",
    "RsfDataUnit",
    "RsfDataFile",
    # MMM
    "ModMaxHeader",
    "ModMaxFreuencyGroup",
    "ModMaxDataUnit",
    # DFT
    "SubCaseHeader",
    "DftHeader",
    "DopplerSpectra",
    "DopplerSpectralBlock",
    # SAO XML
    "URSI",
    "SAORecord",
    "SAORecordList",
]
