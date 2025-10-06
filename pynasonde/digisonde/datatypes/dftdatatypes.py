"""DFT datatypes used by the Digisonde DFT parser.

This module defines small `dataclass` containers that mirror the on-disk
DFT record structures used by Digisonde DFT-format files. The classes are
lightweight holders for parsed fields and provide small normalization steps
in ``__post_init__`` where appropriate.

These types are intended to be used by the parser code under
``pynasonde.digisonde.parsers.dft`` and by higher-level utilities that need
structured access to the parsed header and spectral data.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SubCaseHeader:
    """Header for a single frequency/height subcase inside a DFT block.

    Attributes:
        frequency: Raw frequency value for this subcase (parser-specific
            units).
        height_mpa: Height value (raw units as read from the record).
        height_bin: Height bin index used for indexing spectral data.
        agc_offset: AGC offset recorded for this subcase.
        polarization: Polarization identifier for the subcase.
    """

    frequency: int = None
    height_mpa: int = None
    height_bin: int = None
    agc_offset: int = None
    polarization: int = None

    def __post_init__(self):
        """Post-init hook for small normalization steps.

        Parsers can populate these fields using raw binary values; if unit
        normalization or type conversions are required they should be
        applied here. The code currently contains a commented-out example of
        scaling frequency if needed.
        """
        # Example normalization (left commented because parser decides):
        # if self.frequency is not None:
        #     self.frequency = int(self.frequency * 10)
        return


@dataclass
class DftHeader:
    """Top-level header for a DFT record block.

    This dataclass collects the primary header fields found at the
    beginning of a DFT data block. Field names follow the original parser
    naming where possible; values are stored as integers or small numeric
    types and may require interpretation by the consumer (units are
    parser-dependent).

    Attributes:
        record_type: int
            Numeric record type identifier.
        year: int
            Year of the measurement.
        doy: int
            Day-of-year timestamp component.
        hour: int
            Hour of day.
        minute: int
            Minute component of timestamp.
        second: int
            Second component of timestamp.
        schdule: int
            (Parser-specific) schedule or run identifier.
        program: int
            Program id or code recorded in the header.
        drift_data_flag: hex
            Drift-data flag field (raw representation).
        journal: hex
            Journal / log field (raw representation).
        first_height_sampling_winodw: int
            First height sampling window index (raw).
        height_resolution: int
            Height resolution or bin size.
        number_of_heights: int
            Number of height bins contained in the block.
        start_frequency: int
            Start frequency of the scan (format-dependent units).
        disk_io: hex
            Disk I/O flag or raw indicator.
        freq_search_enabled: bin
            Frequency-search enabled flag (raw/bin representation).
        fine_frequency_step: int
            Fine frequency step value.
        number_small_steps_scan_abs: int
            Absolute number of small frequency steps.
        number_small_steps_scan: int
            Number of small steps in the scan.
        start_frequency_case: int
            Case-specific start frequency index.
        coarse_frequency_step: int
            Coarse frequency step value.
        end_frequency: int
            End frequency of the scan (format-dependent units).
        bottom_height: int
            Bottom height index or value for the block.
        top_height: int
            Top height index or value for the block.
        unused: int
            Reserved / unused field.
        stn_id: int
            Station identifier.
        phase_code: int
            Phase code or modulation identifier.
        multi_antenna_sequence: int
            Multi-antenna sequencing flag/identifier.
        cit_length: int
            CIT (control info) length field.
        num_doppler_lines: int
            Number of Doppler spectral lines per sample.
        pulse_repeat_rate: int
            Pulse repeat rate of the measurement.
        waveform_type: int
            Waveform type identifier.
        delay: int
            Inter-pulse or processing delay.
        frequency_search_offset: int
            Frequency search offset used during scanning.
        auto_gain: int
            Automatic gain control setting.
        heights_to_output: int
            Number of heights to output or include.
        num_of_polarizations: int
            Number of polarization channels recorded.
        start_gain: int
            Start gain setting for the receiver.
        subcases: List[SubCaseHeader]
            Optional list of `SubCaseHeader` entries describing
            per-frequency subcases within this DFT block.

    Note: Many additional fields present in the original format are kept as
    attributes to preserve full fidelity. Consult the parser implementation
    for exact byte-to-field mappings.
    """

    record_type: int = None
    year: int = None
    doy: int = None
    hour: int = None
    minute: int = None
    second: int = None
    schdule: int = None
    program: int = None
    drift_data_flag: hex = None
    journal: hex = None
    first_height_sampling_winodw: int = None
    height_resolution: int = None
    number_of_heights: int = None
    start_frequency: int = None
    disk_io: hex = None
    freq_search_enabled: bin = None
    fine_frequency_step: int = None
    number_small_steps_scan_abs: int = None
    number_small_steps_scan: int = None
    start_frequency_case: int = None
    coarse_frequency_step: int = None
    end_frequency: int = None
    bottom_height: int = None
    top_height: int = None
    unused: int = None
    stn_id: int = None
    phase_code: int = None
    multi_antenna_sequence: int = None
    cit_length: int = None
    num_doppler_lines: int = None
    pulse_repeat_rate: int = None
    waveform_type: int = None
    delay: int = None
    frequency_search_offset: int = None
    auto_gain: int = None
    heights_to_output: int = None
    num_of_polarizations: int = None
    start_gain: int = None

    subcases: List[SubCaseHeader] = None

    def __post_init__(self):
        """Optional normalization after construction.

        Parsers may choose to perform small unit conversions or sanity checks
        here. The method is intentionally lightweight to avoid surprising
        side-effects during object construction.
        """
        # Example: if timestamp fields were zero-padded strings they could
        # be converted to ints here. Keep processing minimal.
        return


@dataclass
class DopplerSpectra:
    """Container for a single Doppler spectrum.

    Attributes:
        amplitude: Array-like amplitude values for the Doppler bins.
        phase: Array-like phase values for the Doppler bins.
    """

    amplitude: np.array = None
    phase: np.array = None

    def __post_init__(self):
        """Apply small amplitude scaling used by the original format.

        The parser historically scales amplitude values by 3/8 to convert
        to the units expected by downstream code (historical dB-like scale).
        This operation is performed in-place on the amplitude array when
        present.
        """
        if self.amplitude is not None:
            try:
                self.amplitude *= 3 / 8  # to dB-like units (parser convention)
            except Exception:
                # If amplitude is not numeric-array-like, skip scaling.
                pass
        return


@dataclass
class DopplerSpectralBlock:
    """Top-level block containing a DFT header and associated spectra.

    Attributes:
        header: `DftHeader` instance describing block-level metadata.
        spectra_line: List of `DopplerSpectra` instances (one per
            height/frequency sample) contained in the block.
    """

    header: DftHeader = None
    spectra_line: List[DopplerSpectra] = None
