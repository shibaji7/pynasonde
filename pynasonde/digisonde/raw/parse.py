"""Functions to help read in raw binary data created by SDL/AFRL HF receivers.
Based on code written by Eugene Dao (AFRL). 
@author Riley Troyer, Space Dynamics Laboratory, riley.troyer@sdl.usu.edu

Copyright Space Dynamics Laboratory, 2025. The U.S. Federal Government retains a royalty-free, non-exclusive, non-
transferable license to read_binary_iq_sdl_afrl_receivers.py pursuant to
DFARS 252.227-7014. All other rights reserved.
"""

import datetime as dt
import os
from dataclasses import dataclass

import numpy as np
from loguru import logger

from pynasonde.digisonde.raw.iqstream_afrl import IQStream

C = 2.99792458e5
CHIP_BW = 30e3
P_CODE_A = np.array([+1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, -1], dtype=np.complex128)
P_CODE_B = np.array([-1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, -1, -1], dtype=np.complex128)

@dataclass
class Program:
    """Class to hold the program information."""

    epoch: dt.datetime
    fft_mode: bool
    rx_tag: str
    save_phase: bool
    signal_type: str
    id: str
    freq_sampling_law: str
    lower_freq_limit: float
    upper_freq_limit: float
    coarse_freq_step: float
    number_of_fine_steps: int
    fine_frequency_step: float
    fine_muliplexing: bool
    inter_pulse_period: float
    number_of_integrated_pulses: int
    inter_pulse_phase_switch: bool
    waveform_type: str
    polarization: str
    data_dir: str
    out_dir: str


class IQDigisonde(object):
    """Class to process IQ data from a digisonde to create the FFT outputs.
    INPUT"""

    def __init__(
        self,
        program: Program,
        min_range: float = -np.inf,
        max_range: float = np.inf,
        nc_flag: bool = True,
    ):
        """
        This class holds the raw datasets.
        """
        self.program = program
        self.min_range = min_range
        self.max_range = max_range
        self.nc_flag = nc_flag
        return

    def process(
        self,
        iq: IQStream,
    ):
        """
        Process the IQ data to create the FFT outputs.
        """
        # Get the program datasets
        epoch, id, rx_rag, save_phase = (
            self.program.epoch,
            self.program.id,
            self.program.rx_tag,
            self.program.save_phase,
        )
        out_dir = os.path.join(
            self.program.out_dir,
            os.uname().nodename,
            id,
            epoch.strftime("%Y-%m-%d"),
        )
        os.makedirs(out_dir, exist_ok=True)
        time_stamp = epoch.strftime("%Y-%m-%d_%H%M%S")
        file_nc = os.path.join(out_dir, f"{id}_{time_stamp}.nc")
        if os.path.isfile(file_nc):
            return self

        logger.info(f"Processing {id} {time_stamp}")

        # Program parameters
        if self.program.freq_sampling_law != "linear":
            raise ValueError("Only linear Freq Stepping Law implemented")

        if self.program.waveform_type != "16-chip complementary":
            raise ValueError("Only 16-chip complementary implemented")
        n_pol = 2 if self.program.polarization == "O and X" else 1
        if self.program.fine_muliplexing:
            phase_switching = False

        coarse_frequencies = np.arange(
            self.program.lower_freq_limit,
            self.program.upper_freq_limit + self.program.coarse_freq_step,
            self.program.coarse_freq_step,
        )
        n_coarse_freq = len(coarse_frequencies)
        n_pulses_per_coarse = (
            self.program.number_of_fine_steps * self.program.number_of_integrated_pulses
        )
        n_pulses_per_coarse *= (1 + (self.program.waveform_type == "16-chip complementary"))
        n_pulses_per_coarse *= (1 + (self.program.polarization == "O and X"))

        # Create IQ Buffer
        CIT = self.program.number_of_integrated_pulses * self.program.inter_pulse_period * n_pol * 2
        if self.program.fine_muliplexing:
            CIT *= self.program.number_of_fine_steps
        
        if self.program.fft_mode:
            ValueError(
                "FFT mode not implemented. Please use the IQ mode for now."
            )
        f_low = iq.center_freq - iq.sample_freq / 2
        n_gates = int(np.floor(
            self.program.inter_pulse_period * CHIP_BW
        ))
        r_axis = np.arange(n_gates) / CHIP_BW * C - 220e-6 * C
        n_freqs = 0
        f_axis = []

        for i_coarse_freq in range(n_coarse_freq):
            for i_fine_freq in range(self.program.number_of_fine_steps):
                f_tune = coarse_frequencies[i_coarse_freq] + self.program.fine_frequency_step * i_fine_freq
                if f_tune < f_low:
                    continue
                n_freqs += 1
                f_axis.append(f_tune)
        f_axis = np.array(f_axis, dtype=np.float32)
        t_axis = []
        cit_voltage = np.zeros((n_gates, self.program.number_of_integrated_pulses), dtype=np.complex64)
        range_dopp = np.zeros(nRep, dtype=np.complex64)
        ionoPOWO = np.zeros((n_freqs, nRanges))
        ionoPOWX = np.zeros((n_freqs, nRanges))
        if savePhase:
            ionoPhaseO = np.zeros((nFreqs, nRanges))
            ionoPhaseX = np.zeros((nFreqs, nRanges))

        return
