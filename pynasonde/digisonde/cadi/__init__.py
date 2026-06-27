"""CADI support for Digisonde workflows (pynasonde 2.0 track)."""

from pynasonde.digisonde.cadi.echo import (
    CadiArray,
    CadiEcho,
    CadiEchoExtractor,
    CadiInterferometryExtractor,
    CadiInterferometryProduct,
    CadiReceiverLayout,
    compute_aoa,
    compute_height_correction_km,
    compute_velocity_from_skymap,
    debug_echo_geometry,
    doppler_frequency,
    los_velocity,
    map_file_iq_to_physical_rx,
)
from pynasonde.digisonde.cadi.extractor import CadiExtractor
from pynasonde.digisonde.cadi.reader import (
    CadiDataset,
    CadiDetection,
    CadiHeader,
    CadiReader,
)

__all__ = [
    "CadiExtractor",
    "CadiReader",
    "CadiHeader",
    "CadiDetection",
    "CadiDataset",
    "CadiArray",
    "CadiEcho",
    "CadiEchoExtractor",
    "CadiInterferometryExtractor",
    "CadiInterferometryProduct",
    "CadiReceiverLayout",
    "compute_aoa",
    "compute_height_correction_km",
    "compute_velocity_from_skymap",
    "debug_echo_geometry",
    "doppler_frequency",
    "los_velocity",
    "map_file_iq_to_physical_rx",
]
