from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SubCaseHeader:
    frequency: int = None
    height_mpa: int = None
    height_bin: int = None
    agc_offset: int = None
    polarization: int = None

    def __post_init__(self):
        # Convert units and types for specific fields
        # self.frequency = self.frequency * 10
        return


@dataclass
class DftHeader:
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
        # Convert units and types for specific fields
        return


@dataclass
class DopplerSpectra:
    amplitude: np.array = None
    phase: np.array = None

    def __post_init__(self):
        self.amplitude *= 3 / 8  # to dB
        return


@dataclass
class DopplerSpectralBlock:
    header: DftHeader = None
    spectra_line: List[DopplerSpectra] = None
