"""MMM / ModMax datatypes used by Digisonde MMM files.

This module defines simple dataclasses that represent the header and data
blocks found in MMM/ModMax-format files. These are small containers that
help the parser provide structured access to parsed fields and perform
lightweight unit conversions in ``__post_init__`` where necessary.
"""

import datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ModMaxHeader:
    """Header for an MMM/ModMax data block.

    Attributes:
        record_type: int
            Numeric record type identifier.
        header_length: int
            Length of the header in bytes.
        version_maker: hex
            Version/maker code (raw representation).
        year: int
            Year of the measurement.
        doy: int
            Day-of-year timestamp component.
        hour: int
            Hour component of timestamp.
        minute: int
            Minute component of timestamp.
        second: int
            Second component of timestamp.
        program_set: hex
            Program set identifier (raw).
        program_type: hex
            Program type identifier (raw).
        journal: List[int]
            Journal bits/flags (parser-specific meaning).
        nom_frequency: float
            Nominal frequency (converted to Hz in ``__post_init__``).
        tape_ctrl: hex
            Tape write control flags (raw).
        print_ctrl: hex
            Printer control flags (raw).
        mmm_opt: hex
            MMM options bitfield (raw).
        print_clean_ctrl: hex
            Printer clean control (raw).
        print_gain_lev: hex
            Printer gain level (raw).
        ctrl_intm_tx: hex
            Control for intermittent transmitter (raw).
        drft_use: hex
            Drift usage flag (raw).
        start_frequency: float
            Start frequency (converted to Hz in ``__post_init__``).
        freq_step: float
            Frequency step (converted to Hz in ``__post_init__``).
        stop_frequency: float
            Stop frequency (converted to Hz in ``__post_init__``).
        trg: hex
            Trigger flags (raw).
        ch_a: hex
            Channel A flags/identifier (raw).
        ch_b: hex
            Channel B flags/identifier (raw).
        sta_id: str
            Station identifier string.
        phase_code: int
            Phase code or modulation identifier.
        ant_azm: int
            Antenna azimuth.
        ant_scan: int
            Antenna scan setting.
        ant_opt: int
            Antenna options / Doppler spacing.
        num_samples: int
            Number of samples recorded.
        rep_rate: int
            Pulse repetition rate.
        pwd_code: int
            Password/code field (parser-specific meaning).
        time_ctrl: int
            Time control flags.
        freq_cor: int
            Frequency correction value.
        gain_cor: int
            Gain correction value.
        range_inc: int
            Range increment value.
        range_start: int
            Starting range value.
        f_search: int
            Frequency search parameter/flag.
        nom_gain: int
            Nominal gain setting.

    Note: "The ``__post_init__`` method performs unit conversions for a subset
    of fields (for example frequency fields are converted to Hz). The
    attributes here reflect the raw-parsed fields before or after those
    lightweight conversions.
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

    Attributes:
        blk_type: int
            Block type identifier (e.g. 1 or 2).
        frequency: int
            Frequency value (MHz in raw header; may be converted by parsers).
        frequency_k: int
            Frequency expressed in kHz.
        frequency_search: int
            Frequency-search parameter/flag.
        gain_param: int
            Gain parameter for this group.
        sec: int
            Time-of-second for this group (timing information).
        mpa: float
            Most probable amplitude value for the group.
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
        header: `ModMaxHeader` object containing block-level metadata.
        frequency_groups: List of parsed frequency-group `ModMaxFreuencyGroup`
            sub-blocks belonging to this unit.
    """

    header: ModMaxHeader = None
    frequency_groups: List[ModMaxFreuencyGroup] = None
