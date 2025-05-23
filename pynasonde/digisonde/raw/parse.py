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
import xarray as xr
from loguru import logger

from pynasonde.digisonde.raw.iqstream_afrl import IQStream

C = 2.99792458e5
CHIP_BW = 30e3
P_CODE_A = np.array(
    [+1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, -1],
    dtype=np.complex128,
)
P_CODE_B = np.array(
    [-1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, -1, -1],
    dtype=np.complex128,
)


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

    def process(self, iq: IQStream) -> xr.Dataset:
        """
        Process the IQ data to create the FFT outputs.
        """
        # Unpack program parameters
        p = self.program
        epoch = p.epoch
        id = p.id
        rx_tag = p.rx_tag
        save_phase = p.save_phase

        out_dir = os.path.join(
            p.out_dir,
            os.uname().nodename,
            id,
            epoch.strftime("%Y-%m-%d"),
        )
        os.makedirs(out_dir, exist_ok=True)
        time_stamp = epoch.strftime("%Y-%m-%d_%H%M%S")
        file_nc = os.path.join(out_dir, f"{id}_{time_stamp}.nc")
        if os.path.isfile(file_nc):
            logger.info(f"File {file_nc} already exists. Skipping.")
            return self

        logger.info(f"Processing {id} {time_stamp}")

        # Program checks
        if p.freq_sampling_law != "linear":
            raise ValueError("Only linear Freq Stepping Law implemented")
        if p.waveform_type != "16-chip complementary":
            raise ValueError("Only 16-chip complementary implemented")
        n_pol = 2 if p.polarization == "O and X" else 1
        phase_switching = p.inter_pulse_phase_switch
        if p.fine_muliplexing:
            phase_switching = False

        # Frequency stepping
        coarse_frequencies = np.arange(
            p.lower_freq_limit,
            p.upper_freq_limit + p.coarse_freq_step,
            p.coarse_freq_step,
        )
        n_coarse_freq = len(coarse_frequencies)
        n_pulses_per_coarse = p.number_of_fine_steps * p.number_of_integrated_pulses
        n_pulses_per_coarse *= 1 + (p.waveform_type == "16-chip complementary")
        n_pulses_per_coarse *= 1 + (p.polarization == "O and X")

        # Range and frequency axes
        n_gates = int(np.floor(p.inter_pulse_period * CHIP_BW))
        r_axis = np.arange(n_gates) / CHIP_BW * C - 220e-6 * C
        n_freqs = 0
        f_axis = []
        for i_coarse_freq in range(n_coarse_freq):
            for i_fine_freq in range(p.number_of_fine_steps):
                f_tune = (
                    coarse_frequencies[i_coarse_freq]
                    + p.fine_frequency_step * i_fine_freq
                )
                if f_tune < (iq.center_freq - iq.sample_freq / 2):
                    continue
                n_freqs += 1
                f_axis.append(f_tune)
        f_axis = np.array(f_axis, dtype=np.float32)
        t_axis = []

        # Allocate arrays for results
        cit_voltage = np.zeros(
            (n_gates, p.number_of_integrated_pulses), dtype=np.complex64
        )
        iono_pow_O = np.zeros((n_freqs, n_gates))
        iono_pow_X = np.zeros((n_freqs, n_gates))
        if save_phase:
            iono_phase_O = np.zeros((n_freqs, n_gates))
            iono_phase_X = np.zeros((n_freqs, n_gates))

        # Chipping parameters
        t_chips = np.arange(0, p.inter_pulse_period, 1 / CHIP_BW)
        n_chips = len(t_chips)
        n_pcode = len(P_CODE_A)
        n_conv = n_chips + n_pcode - 1
        np2 = int(2 ** np.ceil(np.log2(n_conv)))
        pcodeA_pad = np.concatenate(
            (P_CODE_A[::-1], np.zeros(np2 - n_pcode, dtype=np.complex128))
        )
        pcodeB_pad = np.concatenate(
            (P_CODE_B[::-1], np.zeros(np2 - n_pcode, dtype=np.complex128))
        )
        pcodeA_ft = np.fft.fft(pcodeA_pad)
        pcodeB_ft = np.fft.fft(pcodeB_pad)
        sCG_pad = np.zeros(np2, dtype=np.complex128)

        # Main frequency and pulse loop
        freq_idx = 0
        for i_coarse_freq in range(n_coarse_freq):
            for i_fine_freq in range(p.number_of_fine_steps):
                f_tune = (
                    coarse_frequencies[i_coarse_freq]
                    + p.fine_frequency_step * i_fine_freq
                )
                if f_tune < (iq.center_freq - iq.sample_freq / 2):
                    continue
                # For each polarization
                for i_pol in range(n_pol):
                    cit_voltage[:, :] = 0.0
                    # For each integrated pulse
                    for i_rep in range(p.number_of_integrated_pulses):
                        # For each complementary code (A/B)
                        for i_comp in range(2):
                            # Calculate pulse index (simplified, may need to adapt for multiplexing)
                            i_pulse = (
                                i_coarse_freq * n_pulses_per_coarse
                                + i_fine_freq
                                * p.number_of_integrated_pulses
                                * 2
                                * n_pol
                                + i_rep * 2 * n_pol
                                + i_pol * 2
                                + i_comp
                            )
                            # Calculate time offset for this pulse
                            pulse_time = epoch + dt.timedelta(
                                seconds=i_pulse * p.inter_pulse_period
                            )
                            t_axis.append(pulse_time.timestamp())

                            # Extract the correct samples for this pulse
                            # (Assume iq.samples is a 1D array of complex64 for the whole second)
                            # Calculate sample indices for this pulse
                            start_idx = int(
                                i_pulse * p.inter_pulse_period * iq.sample_freq
                            )
                            end_idx = start_idx + int(
                                p.inter_pulse_period * iq.sample_freq
                            )
                            if end_idx > len(iq.samples):
                                continue  # Not enough data for this pulse
                            pulse_samples = iq.samples[start_idx:end_idx]

                            # Interpolate to chip grid
                            t_samples = np.arange(len(pulse_samples)) / iq.sample_freq
                            samples_chip_grid = np.interp(
                                t_chips, t_samples, pulse_samples.real
                            ) + 1j * np.interp(t_chips, t_samples, pulse_samples.imag)

                            # Pick phase code
                            pcode_ft = pcodeA_ft if i_comp == 0 else pcodeB_ft
                            if phase_switching and (i_rep % 2 == 1):
                                pcode_ft = -pcode_ft

                            # Correlate by convolution (FFT)
                            sCG_pad[:n_chips] = samples_chip_grid
                            sCG_pad[n_chips:] = 0.0
                            sCG_pad_ft = np.fft.fft(sCG_pad)
                            corr = np.fft.ifft(sCG_pad_ft * pcode_ft)
                            # Only real part is used for voltage
                            cit_voltage[:, i_rep] += corr[n_pcode : n_conv + 1].real

                    # Doppler FFT across pulses for each range gate
                    for i_gate in range(n_gates):
                        range_dopp = np.fft.fft(cit_voltage[i_gate, :])
                        range_power = np.max(np.abs(range_dopp)) ** 2
                        # Phase (optional)
                        if save_phase:
                            max_idx = np.argmax(np.abs(range_dopp))
                            range_phase = np.angle(range_dopp[max_idx])
                        # Store results
                        if i_pol == 0:
                            iono_pow_O[freq_idx, i_gate] = range_power
                            if save_phase:
                                iono_phase_O[freq_idx, i_gate] = range_phase
                        else:
                            iono_pow_X[freq_idx, i_gate] = range_power
                            if save_phase:
                                iono_phase_X[freq_idx, i_gate] = range_phase
                freq_idx += 1

        # (Optional: Save to NetCDF or other output here)
        # Save results to NetCDF file using Xarray
        data_vars = {
            "power_O": (("frequency", "range"), iono_pow_O),
            "power_X": (("frequency", "range"), iono_pow_X),
        }
        if save_phase:
            data_vars["phase_O"] = (("frequency", "range"), iono_phase_O)
            data_vars["phase_X"] = (("frequency", "range"), iono_phase_X)

        coords = {
            "frequency": f_axis,
            "range": r_axis,
        }
        attrs = {
            "id": id,
            "rx_tag": rx_tag,
            "epoch": epoch.isoformat(),
            "created_by": "IQDigisonde.process",
            "history": f"Created {dt.datetime.utcnow().isoformat()}Z",
        }
        dso = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        if self.nc_flag:
            dso.to_netcdf(file_nc)
        logger.info(f"Processing complete for {id} {time_stamp}")
        return dso
