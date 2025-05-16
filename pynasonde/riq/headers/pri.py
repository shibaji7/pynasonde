from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from loguru import logger


@dataclass
class PriType:
    frequency: np.float64 = 0.0
    ut_time: np.float64 = 0.0
    gate_start: np.float64 = 0.0
    gate_end: np.float64 = 0.0
    gate_step: np.float64 = 0.0
    relative_time: np.float64 = 0.0
    tk: np.float64 = 0.0  # Magnitude of K vector |k|
    pulset_length: np.int32 = 0
    pulset_index: np.int32 = 0
    receiver_count: np.int32 = 0
    gate_count: np.int32 = 0
    max_rx: np.int32 = 0
    max_rg: np.int32 = 0

    # Control flags
    raw: bool = False
    stackraw: bool = False
    siraw: bool = False
    gate_select: bool = False
    r_limit: bool = False

    mask_rx: np.int32 = 0
    phase_ref: np.int32 = 0
    ir_gate: np.int32 = 0
    rgt1: np.int32 = 0
    rgt2: np.int32 = 0

    def __post_init__(self, max_rx: np.int32 = 16, max_rg: np.int32 = 64):
        # The raw I/Q values for each receiver/range
        self.a_scan: np.ndarray = field(
            default_factory=lambda: np.zeros((max_rx, max_rg), dtype=np.complex128)
        )
        # Amplitude and phase values of I/Q
        self.amplitude: np.ndarray = field(
            default_factory=lambda: np.zeros((max_rx, max_rg))
        )
        self.phase: np.ndarray = field(
            default_factory=lambda: np.zeros((max_rx, max_rg))
        )
        self.ampdB: np.ndarray = field(
            default_factory=lambda: np.zeros((max_rx, max_rg))
        )

        # Range Gate Time
        self.rg_time: List[float] = field(default_factory=lambda: [0.0] * max_rg)
        # Azimuth and Zenith in radian spherical coordinates, Doppler in Hz
        self.zenith: List[float] = field(default_factory=lambda: [0.0] * max_rg)
        self.azimuth: List[float] = field(default_factory=lambda: [0.0] * max_rg)
        self.doppler: List[float] = field(default_factory=lambda: [0.0] * max_rg)

        # Error Bars on Azimuth, Zenith, Doppler
        self.zn_err: List[float] = field(default_factory=lambda: [0.0] * max_rg)
        self.az_err: List[float] = field(default_factory=lambda: [0.0] * max_rg)
        self.dop_err: List[float] = field(default_factory=lambda: [0.0] * max_rg)

        # The K vector and it's Error
        self.vk: np.ndarray = field(default_factory=lambda: np.zeros((max_rg, 3)))
        self.vk_err: np.ndarray = field(default_factory=lambda: np.zeros((max_rg, 3)))

        # A bad data flag.  0 is good, -1 is bad, others TBD
        self.flag: List[int] = field(
            default_factory=lambda: [0] * max_rg
        )  # Bad data flag

        # Phase0 and Correlation Coefficents in the X and Y directions
        self.phase0: np.ndarray = field(default_factory=lambda: np.zeros((max_rg, 2)))
        self.corrC: np.ndarray = field(default_factory=lambda: np.zeros((max_rg, 2)))

        self.noise: List[float] = field(default_factory=lambda: [0.0] * max_rx)
        self.peak: List[float] = field(default_factory=lambda: [0.0] * max_rx)

        self.peak_range_gate: List[int] = field(default_factory=lambda: [0] * max_rx)
        return
