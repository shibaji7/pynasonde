"""Shared dataclass implementations for RSF/SBF binary formats."""

import datetime as dt
from dataclasses import dataclass
from typing import ClassVar, List

import numpy as np


@dataclass
class RsfSbfHeader:
    """Common RSF/SBF header fields and unit conversions."""

    record_type: int
    header_length: int
    version_maker: int

    year: int = 0
    doy: int = 0
    month: int = 0
    dom: int = 0
    hour: int = 0
    minute: int = 0
    second: int = 0
    stn_code_rx: str = ""
    stn_code_tx: str = ""
    schedule: int = 0
    program: int = 0
    start_frequency: float = 0.0
    coarse_frequency_step: float = 0.0
    stop_frequency: float = 0.0
    fine_frequency_step: float = 0.0
    num_small_steps_in_scan: int = 0
    phase_code: int = 0
    option_code: int = 0
    number_of_samples: int = 0
    pulse_repetition_rate: int = 0
    range_start: int = 0
    range_increment: int = 0
    number_of_heights: int = 0
    delay: int = 0
    base_gain: int = 0
    frequency_search: int = 0
    operating_mode: int = 0
    data_format: int = 0
    printer_output: int = 0
    threshold: int = 0
    constant_gain: int = 0
    cit_length: int = 0
    journal: str = ""
    bottom_height_window: int = 0
    top_height_window: int = 0
    number_of_heights_stored: int = 0
    spare: bytes = b""
    number_of_frequency_groups: int = 0

    def __post_init__(self):
        self.start_frequency *= 1e2
        self.coarse_frequency_step *= 1e3
        self.stop_frequency *= 1e2
        self.fine_frequency_step *= 1e3
        if self.range_increment == 2:
            self.range_increment = 2.5
        elif self.range_increment == 5:
            self.range_increment = 5
        elif self.range_increment == 10:
            self.range_increment = 10
        self.threshold = 3 * (self.threshold - 10) if self.threshold else np.nan
        self.date = dt.datetime(
            self.year, self.month, self.dom, self.hour, self.minute, self.second
        )


@dataclass
class RsfSbfFreuencyGroup:
    """Common RSF/SBF frequency-group fields and conversions."""

    zero_amplitude_below_mpa: ClassVar[bool] = False
    compute_azm_directions: ClassVar[bool] = False

    pol: str = ""
    group_size: int = 0
    frequency_reading: float = 0.0
    offset: int = 0
    additional_gain: float = 0.0
    seconds: int = 0
    mpa: float = 0.0
    amplitude: np.array = None
    dop_num: np.array = None
    phase: np.array = None
    azimuth: np.array = None
    height: np.array = None
    azm_directions: list | None = None

    def setup(self, threshold: float = 0.0):
        """Convert raw group arrays and metadata to parser-facing units."""
        self.azimuth *= 60
        self.phase = self.phase.astype(np.float64) * 11.25
        self.amplitude *= 3
        self.offset = self._decode_offset(self.offset)
        self.additional_gain *= 3
        self.frequency_reading *= 10e3
        self.height = np.zeros_like(self.amplitude, dtype=np.float64)

        if self.zero_amplitude_below_mpa:
            self.amplitude[self.amplitude < self.mpa] = 0

        if self.compute_azm_directions:
            self.azm_directions = [
                self._direction_from_azimuth(az) for az in self.azimuth.astype(np.int64)
            ]

    @staticmethod
    def _decode_offset(offset):
        offset = str(offset)
        offset_map = {
            "0": -20e3,
            "1": -10e3,
            "2": 0,
            "3": 10e3,
            "4": 20e3,
            "5": "Search Failed",
            "E": "Forced",
            "F": "No Tx",
        }
        return offset_map.get(offset, offset)

    @staticmethod
    def _direction_from_azimuth(azimuth: int) -> str:
        direction_map = {
            0: "N",
            60: "NE",
            120: "SE",
            180: "S",
            240: "SW",
            300: "NW",
        }
        return direction_map[np.mod(azimuth, 360)]


@dataclass
class RsfSbfDataUnit:
    """Common data-unit container for RSF/SBF block-level products."""

    header: object = None
    frequency_groups: List[object] = None

    def setup(self):
        """Populate per-group height arrays from the block header."""
        for group in self.frequency_groups:
            group.height = self.header.range_start + (
                np.arange(len(group.height)) * self.header.range_increment
            )
