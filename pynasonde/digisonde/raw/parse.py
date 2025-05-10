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

from pynasonde.digisonde.raw.iqstream_afrl import IQStream


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
        iqstream: IQStream,
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
        self.iqstream = iqstream
        return

    def process(self):
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
            epoch.strftime("%Y-%m-%d_%H%M%S"),
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return self
