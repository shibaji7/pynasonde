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

    # I/Q data for receiver and range gate (2, max_rg, max_rx)
    pulse_i: np.ndarray = None
    pulse_q: np.ndarray = None
    # Amplitude of I/Q data (max_rg, max_rx)
    amplitude: np.ndarray = None
    # Phase of I/Q data (max_rg, max_rx) in radians 0 to 2*pi
    phase: np.ndarray = None
    # Amplitude in dB (max_rg, max_rx)
    ampdB: np.ndarray = None
    # Range gate / time (max_rg)
    rg_time: List[float] = None
    # Zenith angle in radians (max_rg) 0 to 2*pi
    zenith: List[float] = None
    # Azimuth angle in radians (max_rg) 0 to 2*pi
    azimuth: List[float] = None
    # Doppler frequency in Hz (max_rg)
    doppler: List[float] = None
    # Zenith error in radians (max_rg)
    zn_err: List[float] = None
    # Azimuth error in radians (max_rg) 0 to 2*pi
    az_err: List[float] = None
    # Doppler error (max_rg)
    dop_err: List[float] = None
    # K vector (max_rg, 3)
    vk: np.ndarray = None
    # Error on K vector (max_rg, 3)
    vk_err: np.ndarray = None
    # Bad data flag (max_rg)
    # 0 is good, -1 is bad, others TBD
    flag: List[int] = None
    # Phase0 (max_rg, 2)
    phase0: np.ndarray = None
    # Correlation coefficients in X and Y directions (max_rg, 2)
    corrC: np.ndarray = None
    # Noise level (max_rx)
    noise: List[float] = None
    # Peak amplitude (max_rx)
    peak: List[float] = None
    # Peak range gate (max_rx)
    peak_range_gate: List[int] = None

    def __post_init__(self):
        logger.debug(f"Initializing PriType with frequency: {self.frequency} kHz")
        # Amplitude and phase values of I/Q
        # self.a_scan = 0.000167 * self.a_scan
        self.amplitude = np.sqrt(self.a_scan[:, :, 0] ** 2 + self.a_scan[:, :, 1] ** 2)
        self.phase = np.arctan2(self.a_scan[:, :, 1], self.a_scan[:, :, 0])
        self.ampdB = 20 * np.log10(self.amplitude)
        # Range gate time in useconds
        self.rg_time = np.arange(self.gate_start, self.gate_end, self.gate_step)

        # Azimuth and Zenith in radian spherical coordinates, Doppler in Hz
        self.zenith: List[float] = field(default_factory=lambda: [0.0] * self.max_rg)
        self.azimuth: List[float] = field(default_factory=lambda: [0.0] * self.max_rg)
        self.doppler: List[float] = field(default_factory=lambda: [0.0] * self.max_rg)

        # Error Bars on Azimuth, Zenith, Doppler
        self.zn_err: List[float] = field(default_factory=lambda: [0.0] * self.max_rg)
        self.az_err: List[float] = field(default_factory=lambda: [0.0] * self.max_rg)
        self.dop_err: List[float] = field(default_factory=lambda: [0.0] * self.max_rg)

        # The K vector and it's Error
        self.vk: np.ndarray = field(default_factory=lambda: np.zeros((self.max_rg, 3)))
        self.vk_err: np.ndarray = field(
            default_factory=lambda: np.zeros((self.max_rg, 3))
        )

        # A bad data flag.  0 is good, -1 is bad, others TBD
        self.flag: List[int] = field(
            default_factory=lambda: [0] * self.max_rg
        )  # Bad data flag

        # Phase0 and Correlation Coefficents in the X and Y directions
        self.phase0: np.ndarray = field(
            default_factory=lambda: np.zeros((self.max_rg, 2))
        )
        self.corrC: np.ndarray = field(
            default_factory=lambda: np.zeros((self.max_rg, 2))
        )

        self.noise: List[float] = field(default_factory=lambda: [0.0] * self.max_rx)
        self.peak: List[float] = field(default_factory=lambda: [0.0] * self.max_rx)

        self.peak_range_gate: List[int] = field(
            default_factory=lambda: [0] * self.max_rx
        )
        return

    def calculate_zenith(self) -> None:
        """
        Calculate the zenith angle in radians.
        """
        # Calculate the zenith angle in radians
        self.zenith = np.arccos(self.vk[:, 2] / np.linalg.norm(self.vk, axis=1))
        return

    def calculate_azimuth(self) -> None:
        """
        Calculate the azimuth angle in radians.
        """
        # Calculate the azimuth angle in radians
        self.azimuth = np.arctan2(self.vk[:, 1], self.vk[:, 0])
        return

    def calculate_doppler(self) -> None:
        """
        Calculate the doppler frequency in Hz.
        """
        # Calculate the doppler frequency in Hz
        self.doppler = np.sqrt(
            self.vk[:, 0] ** 2 + self.vk[:, 1] ** 2 + self.vk[:, 2] ** 2
        )
        return

    def calculate_errors(self) -> None:
        """
        Calculate the errors in the zenith, azimuth, and doppler frequencies.
        """
        # Calculate the errors in the zenith, azimuth, and doppler frequencies
        self.zn_err = np.abs(
            self.zenith - np.arccos(self.vk[:, 2] / np.linalg.norm(self.vk, axis=1))
        )
        self.az_err = np.abs(self.azimuth - np.arctan2(self.vk[:, 1], self.vk[:, 0]))
        self.dop_err = np.abs(
            self.doppler
            - np.sqrt(self.vk[:, 0] ** 2 + self.vk[:, 1] ** 2 + self.vk[:, 2] ** 2)
        )
        return

    def calculate_phase0(self) -> None:
        """
        Calculate the phase0 in radians.
        """
        # Calculate the phase0 in radians
        self.phase0 = np.arctan2(self.a_scan[:, :, 1], self.a_scan[:, :, 0])
        return

    def calculate_correlation(self) -> None:
        """
        Calculate the correlation coefficients in the X and Y directions.
        """
        # Calculate the correlation coefficients in the X and Y directions
        self.corrC = np.corrcoef(self.a_scan[:, :, 0], self.a_scan[:, :, 1])
        return

    def calculate_noise(self) -> None:
        """
        Calculate the noise level in dB.
        """
        return

    def calculate_peak(self) -> None:
        """
        Calculate the peak amplitude in dB.
        """
        return

    def calculate_peak_range_gate(self) -> None:
        """
        Calculate the peak range gate in dB.
        """
        return
