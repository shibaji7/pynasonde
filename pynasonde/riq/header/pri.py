from dataclasses import dataclass, field
from typing import List

import numpy as np

from pynasonde.riq.header.pct import PctType


@dataclass
class PriType:
    frequency: float = 0.0
    ut_time: float = 0.0
    gate_start: float = 0.0
    gate_step: float = 0.0
    relative_time: float = 0.0
    tk: float = 0.0  # Magnitude of K vector |k|
    pulset_length: int = 0
    pulset_index: int = 0
    receiver_count: int = 0
    gate_count: int = 0

    # Control flags
    raw: bool = False
    stackraw: bool = False
    siraw: bool = False
    gate_select: bool = False
    r_limit: bool = False

    mask_rx: int = 0
    phase_ref: int = 0
    ir_gate: int = 0
    rgt1: int = 0
    rgt2: int = 0

    pct: PctType = field(default_factory=PctType)

    def __init__(self, MAXRX: int = 16, MAXRG: int = 64):
        self.a_scan: np.ndarray = field(
            default_factory=lambda: np.zeros((MAXRX, MAXRG), dtype=np.complex128)
        )
        self.amplitude: np.ndarray = field(
            default_factory=lambda: np.zeros((MAXRX, MAXRG))
        )
        self.phase: np.ndarray = field(default_factory=lambda: np.zeros((MAXRX, MAXRG)))
        self.ampdB: np.ndarray = field(default_factory=lambda: np.zeros((MAXRX, MAXRG)))

        self.rg_time: List[float] = field(default_factory=lambda: [0.0] * MAXRG)
        self.zenith: List[float] = field(default_factory=lambda: [0.0] * MAXRG)
        self.azimuth: List[float] = field(default_factory=lambda: [0.0] * MAXRG)
        self.doppler: List[float] = field(default_factory=lambda: [0.0] * MAXRG)

        self.zn_err: List[float] = field(default_factory=lambda: [0.0] * MAXRG)
        self.az_err: List[float] = field(default_factory=lambda: [0.0] * MAXRG)
        self.dop_err: List[float] = field(default_factory=lambda: [0.0] * MAXRG)

        self.vk: np.ndarray = field(default_factory=lambda: np.zeros((MAXRG, 3)))
        self.vk_err: np.ndarray = field(default_factory=lambda: np.zeros((MAXRG, 3)))

        self.flag: List[int] = field(
            default_factory=lambda: [0] * MAXRG
        )  # Bad data flag

        self.phase0: np.ndarray = field(default_factory=lambda: np.zeros((MAXRG, 2)))
        self.corrC: np.ndarray = field(default_factory=lambda: np.zeros((MAXRG, 2)))

        self.noise: List[float] = field(default_factory=lambda: [0.0] * MAXRX)
        self.peak: List[float] = field(default_factory=lambda: [0.0] * MAXRX)

        self.peak_range_gate: List[int] = field(default_factory=lambda: [0] * MAXRX)
