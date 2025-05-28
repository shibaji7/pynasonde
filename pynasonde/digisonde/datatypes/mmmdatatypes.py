import datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MultiplexHeader:
    """
    Class to represent the header of an RSF file.
    """

    # Header fields
    record_type: int
    header_length: int
    version_maker: hex

    # Preface fields
    # Units in Year
    year: int = 0
    # Units in Day of Year
    doy: int = 0
    # Units in Hour
    hour: int = 0
    # Units in Minute
    minute: int = 0
    # Units in Second
    second: int = 0
    # Station codes 000-999
    stn_code_rx: str = ""
    # Station codes 000-999
    stn_code_tx: str = ""
    # Schedule code 1-6
    schedule: int = 0
    # Program code 1-7 (A-G)
    program: int = 0
    # Frequency in 100 Hz (010000 - 450000)
    start_frequency: float = 0.0
    # Frequency step in 1 kHz (1-2000)
    coarse_frequency_step: float = 0.0
    # Frequency in 100 Hz (010000 - 450000)
    stop_frequency: float = 0.0
    # Frequency step in 1 kHz (1-9999)
    fine_frequency_step: float = 0.0
    # Number of frequency steps (-15 to 15)
    # negative value means no multiplexing
    num_small_steps_in_scan: int = 0
    # 1 (complim.) 2 (short) 3 (75% duty) 4 (100% duty) +8 (no phase switch)
    phase_code: int = 0
    # 0 (sum), 1-4 (individual antennas), 7 (antenna scan), +8 (only O polarization),
    # negative for alternative antennas
    option_code: int = 0
    # encoded, actual # is power of 2 (3-7)
    number_of_samples: int = 0
    # in pps; 50, 100, 200. nibble 4: 0 - Active Mode 1 - Radio Silent Mode
    pulse_repetition_rate: int = 0
    # in km; 0-9999
    range_start: int = 0
    # Encoded km; 2 (2.5 km) 5 (5 km) 10 (10 km)
    range_increment: int = 0
    # Units: 128, 256, 512
    number_of_heights: int = 0
    # in 15 km; 0 - 1500
    delay: int = 0
    # in dB; 0-7 (0-42 dB) +8 (+auto gain)
    base_gain: int = 0
    # 0 (no) 1 (yes)
    frequency_search: int = 0
    # 0 (VI) 1 (Drift Std) 2 (Drift Auto) 3 (Calibration) 4 (HRR) 5 (Beam) 6 (PGH) 7 (Test)
    operating_mode: int = 0
    # 0 (no data) 1 (MMM) 2 (Drift) 3 (PGH) 4 (RSF) 5 (SBF) 6 (BIT) high nibble: 0 (no Artist) 1 (with Artist)
    data_format: int = 0
    # 0 (none) 1 (b/w) 2 (color)
    printer_output: int = 0
    # 3 dB over the MPA
    threshold: int = 0
    # 0 (full gain: tracker high, switch high),
    # 1 ( -9 dB: tracker high, switch low),
    # 2 (-9 dB: tracker low, switch high),
    # 3 ( -18 dB: tracker low, switch low)
    constant_gain: int = 0
    # msec; 0-40000
    cit_length: int = 0
    # bit0: new gain bit1: new height bit2: new freq. bit3: new case
    journal: str = ""
    # in 1 km; 0-9999
    bottom_height_window: int = 0
    # in 1 km; 0-9999
    top_height_window: int = 0
    # Units; 1-512
    number_of_heights_stored: int = 0
    # spare: 2 bytes
    spare: bytes = b""
    # Number of frequency groups
    number_of_frequency_groups: int = 0

    def __post_init__(self):
        return
