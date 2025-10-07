"""SBF datatypes for Digisonde SBF-format files.

Defines dataclasses for SBF headers and frequency groups. Methods such
as ``setup`` perform unit conversions and populate derived arrays (like
height vectors) used by plotting routines.
"""

import datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SbfHeader:
    """Header for an SBF data block.

    Many numeric fields are stored in compact encodings in the file format;
    the ``__post_init__`` method converts common fields to SI units (Hz,
    datetime, etc.) for convenience.

    Attributes:
        record_type: int
            Numeric record type identifier.
        header_length: int
            Length of the header in bytes.
        version_maker: int
            Version/maker code (raw integer).
        year: int
            Year of the measurement.
        doy: int
            Day-of-year component.
        month: int
            Month component.
        dom: int
            Day-of-month component.
        hour: int
            Hour component.
        minute: int
            Minute component.
        second: int
            Second component.
        stn_code_rx: str
            Receiver station code.
        stn_code_tx: str
            Transmitter station code.
        schedule: int
            Schedule code.
        program: int
            Program code.
        start_frequency: float
            Start frequency (converted to Hz in ``__post_init__``).
        coarse_frequency_step: float
            Coarse frequency step (converted to Hz in ``__post_init__``).
        stop_frequency: float
            Stop frequency (converted to Hz in ``__post_init__``).
        fine_frequency_step: float
            Fine frequency step (converted to Hz in ``__post_init__``).
        num_small_steps_in_scan: int
            Number of small frequency steps (negative => no multiplexing).
        phase_code: int
            Phase code field describing pulse/phase behaviour.
        option_code: int
            Option code (antenna selection, polarization flags, etc.).
        number_of_samples: int
            Encoded number of samples.
        pulse_repetition_rate: int
            Pulse repetition rate (pps).
        range_start: int
            Range start in km.
        range_increment: int or float
            Range increment (km); common encoded values are converted in
            ``__post_init__`` (e.g. 2 -> 2.5).
        number_of_heights: int
            Number of heights configured in the scan.
        delay: int
            Delay in 15 km units.
        base_gain: int
            Base gain code (dB-like representation with special bits).
        frequency_search: int
            Frequency search enabled flag.
        operating_mode: int
            Operating mode code.
        data_format: int
            Data format code (MMM, Drift, PGH, RSF, etc.).
        printer_output: int
            Printer output setting (none, b/w, color).
        threshold: int or float
            Threshold level (converted to dB-like value in ``__post_init__``).
        constant_gain: int
            Constant gain code (bitfield describing fixed gain settings).
        cit_length: int
            CIT length in ms.
        journal: str
            Journal / flags string (parser-specific meaning).
        bottom_height_window: int
            Bottom height window (km).
        top_height_window: int
            Top height window (km).
        number_of_heights_stored: int
            Number of height bins actually stored.
        spare: bytes
            Spare/reserved bytes from header.
        number_of_frequency_groups: int
            Number of frequency groups that follow the header.
        date: datetime.datetime
            Python datetime synthesized from the timestamp fields in
            ``__post_init__``.
    """

    # Header fields
    record_type: int
    header_length: int
    version_maker: int

    # Preface fields
    # Units in Year
    year: int = 0
    # Units in Day of Year
    doy: int = 0
    # Units in Month
    month: int = 0
    # Units in Day of Month
    dom: int = 0
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
        self.start_frequency *= 1e2  # to Hz
        self.coarse_frequency_step *= 1e3  # to Hz
        self.stop_frequency *= 1e2  # to Hz
        self.fine_frequency_step *= 1e3  # to Hz
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
        return


@dataclass
class SbfFreuencyGroup:
    """Represents a frequency group within an SBF block.

    Instances carry arrays for amplitude, phase and azimuth along with
    metadata fields like `mpa`, `offset`, and `frequency_reading`. The
    ``setup`` method converts units and interprets offset codes into
    human-readable values.

    Attributes:
        pol: str
            Polarization identifier.
        group_size: int
            Encoded group size (parser-specific meaning).
        frequency_reading: float
            Frequency reading (converted to Hz by ``setup``).
        offset: int or str
            Offset code (converted to Hz or descriptive string by ``setup``).
        additional_gain: float
            Additional gain (converted to dB by ``setup``).
        seconds: int
            Seconds timestamp for the group.
        mpa: float
            Most probable amplitude value for the group.
        amplitude: np.array
            Amplitude array for the group's samples.
        dop_num: np.array
            Doppler index array for each sample.
        phase: np.array
            Phase array (converted to degrees by ``setup``).
        azimuth: np.array
            Azimuth array (converted to degrees by ``setup``).
        height: np.array
            Height array corresponding to amplitude samples.
    """

    # Frequency group fields
    # 0 (no) 1 (yes)
    pol: str = ""
    # Units; encoded 2 (262), 3 (504), 4 (1008)
    group_size: int = 0
    # within the ionogram frequency range; 10 kHz
    frequency_reading: float = 0.0
    # Offset, 0 (-20 kHz) 1 (-10 kHz) 2 (no offset) 3 (+10 kHz) 4 (+20 kHz) 5 (search failure) E (forced) F (no transmission)
    offset: int = 0
    # 3 dB; Range 0-15
    additional_gain: float = 0.0
    # 00-59
    seconds: int = 0
    # Most Probable Amplitude; 0-31
    mpa: float = 0.0
    # amplitude, 3 dB * 0-31
    amplitude: np.array = None
    # Dopler num, 0-7
    dop_num: np.array = None
    # Phase, 0-31; Units 11.25 deg or 1 km
    phase: np.array = None
    # Units 60 deg; Range 0-7
    azimuth: np.array = None
    # Height, Dynamic range; Units km
    height: np.array = None

    def setup(self, threshold: float = 0.0):
        """
        Configures and converts the instance attributes to their appropriate units.

        This method performs the following conversions:
        * Converts `azimuth` from minutes to degrees by multiplying by 60.
        * Converts `phase` from units to degrees by multiplying by 11.25.
        * Converts `amplitude` from units to decibels (dB) by multiplying by 3.
        * Converts `offset` to a string representation.
        * Converts `mpa` from units to decibels (dB) by multiplying by 3.
        * Converts `additional_gain` from units to decibels (dB) by multiplying by 3.
        * Converts `frequency_reading` from kHz to Hz by multiplying by 10,000.
        Args:
            threshold (float, optional): Threshold value for filtering. Defaults to 0.0.
        """
        self.azimuth *= 60  # Convert to degrees
        self.phase = self.phase.astype(np.float64) * 11.25  # Convert to degrees
        self.amplitude *= 3  # Convert to dB
        self.offset = str(self.offset)
        if self.offset == "0":
            self.offset = -20e3  # in Hz
        if self.offset == "1":
            self.offset = -10e3  # in Hz
        if self.offset == "2":
            self.offset = 0  # in Hz
        if self.offset == "3":
            self.offset = 10e3  # in Hz
        if self.offset == "4":
            self.offset = 20e3  # in Hz
        if self.offset == "5":
            self.offset = "Search Failed"
        if self.offset == "E":
            self.offset = "Forced"
        if self.offset == "F":
            self.offset = "No Tx"
        # self.mpa *= 3  # convert to dB
        self.additional_gain *= 3  # convert to dB
        self.frequency_reading *= 10e3  # Convert to Hz
        self.height = np.zeros_like(self.amplitude, dtype=np.float64)
        # Need to filter out values below threshold
        return


@dataclass
class SbfDataUnit:
    """Container for a parsed SBF data block.

    Attributes:
        header: SbfHeader
            Header instance with block-level metadata in `SbfHeader`.
        frequency_groups: List[SbfFreuencyGroup]
            List of frequency-group `SbfFreuencyGroup` objects parsed for this block.

    Notes: After parsing, call ``setup`` to populate group-derived attributes like
    the `height` arrays for each frequency group.
    """

    header: SbfHeader = None
    frequency_groups: List[SbfFreuencyGroup] = None

    def setup(self):
        """
        Configures the SBF data unit by setting up each frequency group.

        This method iterates through each frequency group in the `frequency_groups` list
        and calls the `setup` method on each group to perform necessary conversions and configurations.
        """
        for group in self.frequency_groups:
            group.height = self.header.range_start + (
                np.arange(len(group.height)) * self.header.range_increment
            )
        return


@dataclass
class SbfDataFile:
    """Container representing the contents of an SBF file.

    Attributes:
        sbf_data_units: List[SbfDataUnit]
            List of parsed SBF data units (`SbfDataUnit` blocks) contained in the file.
    """

    sbf_data_units: List[SbfDataUnit] = None
