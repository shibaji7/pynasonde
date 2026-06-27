"""RSF (range-spectral-format) binary parser utilities for Digisonde.

This module exposes :class:`RsfExtractor`, a low-level reader that unpacks
RSF-format binary blocks into dataclasses defined in
``pynasonde.digisonde.datatypes.rsfdatatypes``. RSF shares its fixed-block
wire format with SBF; the byte-reading implementation lives in the internal
``_rsf_sbf_base`` module while this class supplies RSF-specific containers and
row metadata.
"""

from pynasonde.digisonde.datatypes.rsfdatatypes import (
    RsfDataFile,
    RsfDataUnit,
    RsfFreuencyGroup,
    RsfHeader,
)
from pynasonde.digisonde.digi_utils import RSF_SBF_IONOGRAM_SETTINGS
from pynasonde.digisonde.parsers._rsf_sbf_base import RsfSbfBinaryBlockExtractor

RSF_IONOGRAM_SETTINGS = RSF_SBF_IONOGRAM_SETTINGS  # backwards-compatible alias


class RsfExtractor(RsfSbfBinaryBlockExtractor):
    """Low-level reader for RSF-format files."""

    data_file_class = RsfDataFile
    data_unit_class = RsfDataUnit
    header_class = RsfHeader
    frequency_group_class = RsfFreuencyGroup
    ionogram_settings = RSF_SBF_IONOGRAM_SETTINGS
    data_attr = "rsf_data"
    units_attr = "rsf_data_units"
    format_label = "RSF"
    include_azm_directions = True
    empty_returns_dataframe = True
