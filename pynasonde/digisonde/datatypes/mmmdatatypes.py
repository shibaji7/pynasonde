"""MMM / ModMax datatypes used by Digisonde MMM files.

This module defines simple dataclasses that represent the header and data
blocks found in MMM/ModMax-format files. These are small containers that
help the parser provide structured access to parsed fields and perform
lightweight unit conversions in ``__post_init__`` where necessary.

Typical usage:
    header = ModMaxHeader(record_type=1, header_length=256, year=2023)
    block = ModMaxDataUnit(header=header, frequency_groups=[...])
"""

import datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ModMaxHeader:
    """Header for an MMM/ModMax data block.

    Attributes reflect the raw fields parsed from the MMM header. Units
    are noted in comments and converted where appropriate in
    ``__post_init__`` (for example frequency values are scaled to Hz).
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
    # Program Set
    program_set: hex = None
    # Program type
    program_type: hex = None
    # bit0: new gain bit1: new height bit2: new freq. bit3: new case
    journal: List[int] = None
    # Nominal Frequency in 100 Hz (010000 - 450000)
    nom_frequency: float = 0.0
    # Tape Write Control
    tape_ctrl: hex = None
    # Printer Control
    print_ctrl: hex = None
    # MMM Options
    mmm_opt: hex = None
    # Printer clean control
    print_clean_ctrl: hex = None
    # printer gain level
    print_gain_lev: hex = None
    # Control for intermittent Tx
    ctrl_intm_tx: hex = None
    # Use for drift
    drft_use: hex = None
    # Start Frequency (MHz)
    start_frequency: float = 0.0
    # Frequency increment
    freq_step: float = 0.0
    # Frequency in MHz
    stop_frequency: float = 0.0
    # Trigger
    trg: hex = None
    # Channel A
    ch_a: hex = None
    # Channel B
    ch_b: hex = None
    # Station id
    sta_id: str = ""
    # Phase code
    phase_code: int = 0
    # Antenna Azimuth
    ant_azm: int = 0
    # Antenna Scan
    ant_scan: int = 0
    # Antenna opt and Dop spacing
    ant_opt: int = 0
    # Num of samples
    num_samples: int = 0
    # Rep rate
    rep_rate: int = 0
    # pwd code
    pwd_code: int = 0
    # Time control
    time_ctrl: int = 0
    # frequency correction
    freq_cor: int = 0
    # Gain Correction
    gain_cor: int = 0
    # Range inc
    range_inc: int = 0
    # range start
    range_start: int = 0
    # Frequency search
    f_search: int = 0
    # Nominal gain
    nom_gain: int = 0

    def __post_init__(self):
        self.nom_frequency *= 1e2  # convert to Hz
        self.start_frequency *= 1e6  # convert to Hz
        self.freq_step *= 1e6  # convert to Hz
        self.stop_frequency *= 1e6  # convert to Hz
        return


@dataclass
class ModMaxFreuencyGroup:
    """Represents a single frequency group (sub-block) inside an MMM block.

    Contains frequency, gain and the most-probable-amplitude (mpa) values
    plus timing information. Parsers populate the numeric arrays and call
    ``setup`` or other helpers to convert units when needed.
    """

    # Block type (1,2)
    blk_type: int = 0
    # Frequency in Mhz
    frequency: int = 0
    # Frequency in KHz
    frequency_k: int = 0
    # frequency search param
    frequency_search: int = 0
    # Gain parameter
    gain_param: int = 0
    # Time of sec
    sec: int = 0
    # Most probable amplitude
    mpa: float = 0.0


@dataclass
class ModMaxDataUnit:
    """Container for a full MMM data block.

    Attributes:
        header: ModMaxHeader instance with block-level metadata.
        frequency_groups: List of ModMaxFreuencyGroup instances with parsed
            frequency-group data for the block.
    """

    header: ModMaxHeader = None
    frequency_groups: List[ModMaxFreuencyGroup] = None
