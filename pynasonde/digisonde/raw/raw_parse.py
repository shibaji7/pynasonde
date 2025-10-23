"""
Python translation of the Julia ``Digisonde`` module.

The ``process`` function reproduces the DPS4D Digisonde ionogram pipeline:
reading complex baseband IQ recordings, performing complementary-code
correlation, assembling range–frequency power grids, and writing results to a
NetCDF product.  The logic mirrors the original Julia implementation as closely
as possible while adopting Pythonic structure and naming.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import netCDF4
import numpy as np
from loguru import logger

from pynasonde.digisonde.raw.iq_reader import IQStream

# ---------------------------------------------------------------------------
# Constants and phase codes

p_code_a = np.array(
    [+1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, -1],
    dtype=np.complex128,
)
p_code_b = np.array(
    [-1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, -1, -1],
    dtype=np.complex128,
)
CHIP_BW = 30_000.0  # Hz
SPEED_OF_LIGHT = 2.99792458e5  # km / s
HOSTNAME = socket.gethostname()
_UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Utility helpers


def _ensure_datetime(value: dt.datetime | float | int) -> dt.datetime:
    """Normalize program epochs to timezone-aware UTC datetimes."""
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_UTC)
        return value.astimezone(_UTC)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=_UTC)
    raise TypeError(f"Unsupported epoch type: {type(value)!r}")


def _next_power_of_two(value: float) -> int:
    """Return the next power of two greater than or equal to ``value``."""
    if value <= 1:
        return 1
    return 1 << (math.ceil(math.log2(value)))


def _prev_power_of_two(value: float) -> int:
    """Return the previous power of two less than or equal to ``value``."""
    if value <= 1:
        return 1
    return 1 << (math.floor(math.log2(value)))


def _next_smooth_235(target: int) -> int:
    """Return the smallest 5-smooth (2^a 3^b 5^c) number >= target."""
    best = None
    limit = target * 4  # conservative upper bound
    for a in range(0, 20):
        for b in range(0, 20):
            value = (2**a) * (3**b)
            if value > limit:
                break
            c = 0
            while True:
                candidate = value * (5**c)
                if candidate >= target:
                    if best is None or candidate < best:
                        best = candidate
                    break
                if candidate > limit:
                    break
                c += 1
    return best if best is not None else target


def _interp_complex(
    x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """Linear interpolation for complex arrays via real/imag components."""
    real = np.interp(x_new, x_old, y_old.real)
    imag = np.interp(x_new, x_old, y_old.imag)
    return real + 1j * imag


# ---------------------------------------------------------------------------
# Processing pipeline


def process(
    program: Dict[str, object],
    dir_iq: Path | str = "/mnt/Data/",
    out_dir: Path | str = "out/",
    min_range: float = -math.inf,
    max_range: float = math.inf,
    nc_flag: bool = True,
    verbose: bool = True,
) -> None:
    """
    Run a single Digisonde sounding program and write the resulting ionogram.

    Parameters mirror the Julia implementation.  Only IQ-mode processing is
    currently supported (FFTMode must be false).
    """

    # ------------------------------------------------------------------
    # Program metadata and output bookkeeping
    program_id = str(program["ID"])
    epoch = _ensure_datetime(program["Epoch"])
    rx_tag = program.get("rxTag", "ch0")
    fft_mode = bool(program.get("FFTMode", False))

    if fft_mode:
        raise NotImplementedError("FFTMode is not supported in the Python port.")

    save_phase = bool(program.get("Save Phase", False))
    out_dir_path = Path(out_dir) / HOSTNAME / program_id / epoch.strftime("%Y-%m-%d")
    out_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = epoch.strftime("%Y-%m-%d_%H%M%S")
    nc_path = out_dir_path / f"{program_id}_{timestamp}.nc"
    if nc_path.exists():
        if verbose:
            logger.info(f"Skipping {nc_path} (already exists)")
        return
    if verbose:
        logger.info(f"Processing {program_id} at {epoch.isoformat()}")

    # ------------------------------------------------------------------
    # Validate program assumptions
    if program.get("Freq Stepping Law") != "linear":
        raise ValueError("Only linear frequency stepping is implemented.")
    if program.get("Wave Form") != "16-chip complementary":
        raise ValueError("Only the 16-chip complementary waveform is supported.")

    lower_freq = float(program["Lower Freq Limit"])
    upper_freq = float(program["Upper Freq Limit"])
    coarse_step = float(program["Coarse Freq Step"])
    fine_steps = int(program["Number of Fine Steps"])
    fine_step = float(program["Fine Freq step"])
    multiplexing = bool(program["Fine Multiplexing"])
    ipp_seconds = float(program["Inter-Pulse Period"])
    repeats = int(program["Number of Integrated Repeats"])
    phase_switching = bool(program["Interpulse Phase Switching"])
    waveform = program["Wave Form"]
    polarization = program.get("Polarization", "O and X")
    n_pol = 2 if polarization == "O and X" else 1
    if multiplexing:
        phase_switching = False  # Matches Julia's "undocumented quirk"

    coarse_frequencies = np.arange(
        lower_freq, upper_freq + coarse_step / 2, coarse_step
    )
    n_coarse = len(coarse_frequencies)
    pulses_per_coarse = fine_steps * repeats
    if waveform == "16-chip complementary":
        pulses_per_coarse *= 2
    if polarization == "O and X":
        pulses_per_coarse *= 2

    # ------------------------------------------------------------------
    # Open IQ stream and derive tuning parameters
    iq_stream = IQStream(dir_iq, epoch, rx_tag=rx_tag)
    f_low = iq_stream.f_center - iq_stream.f_sample / 2

    # Range bins and arrays
    n_ranges = int(math.floor(ipp_seconds * CHIP_BW))
    range_axis = (
        np.arange(n_ranges) / CHIP_BW
    ) * SPEED_OF_LIGHT - 220e-6 * SPEED_OF_LIGHT

    # Frequency axis assembly
    freq_axis: List[float] = []
    for coarse in coarse_frequencies:
        for step_index in range(fine_steps):
            tune_freq = coarse + fine_step * step_index
            if tune_freq < f_low:
                continue
            freq_axis.append(tune_freq)
    n_freqs = len(freq_axis)

    time_axis: List[float] = []
    cit_voltage = np.zeros((n_ranges, repeats), dtype=np.complex64)
    iono_power_o = np.zeros((n_freqs, n_ranges), dtype=np.float64)
    iono_power_x = np.zeros_like(iono_power_o)
    iono_phase_o = np.zeros_like(iono_power_o) if save_phase else None
    iono_phase_x = np.zeros_like(iono_power_o) if save_phase and n_pol == 2 else None

    # Sample buffers and FFT helpers
    n_samples = _next_power_of_two(ipp_seconds * iq_stream.f_sample)
    logger.debug(f"Using {n_samples} samples per pulse")
    samples = np.zeros(n_samples, dtype=np.complex64)
    hz_per_bin = iq_stream.f_sample / n_samples

    if not fft_mode:
        decimation_factor = max(1, _prev_power_of_two(iq_stream.f_sample / CHIP_BW))
        samples_ds = np.zeros(n_samples // decimation_factor, dtype=np.complex64)
        n_ds = samples_ds.size
        idx_front = np.arange(0, n_ds // 2 + (n_ds % 2))
        idx_back = np.arange(0, n_ds // 2)
        f_sample_ds = iq_stream.f_sample * n_ds / n_samples
        window_ds = np.hamming(n_ds).astype(np.float32)
        mix_signal = np.zeros(n_ds, dtype=np.complex64)
        t_ds = np.arange(n_ds, dtype=np.float64) / f_sample_ds
    else:
        raise NotImplementedError("FFTMode true path is not implemented.")

    t_chips = np.arange(0, ipp_seconds, 1.0 / CHIP_BW)
    n_chips = t_chips.size
    n_pcode = p_code_a.size
    n_conv = n_chips + n_pcode - 1
    fft_length = (
        _next_smooth_235(n_conv) if n_conv > 1024 else _next_power_of_two(n_conv)
    )

    p_code_a_pad = np.concatenate(
        (p_code_a[::-1], np.zeros(fft_length - n_pcode, dtype=np.complex128))
    )
    p_code_b_pad = np.concatenate(
        (p_code_b[::-1], np.zeros(fft_length - n_pcode, dtype=np.complex128))
    )
    p_code_a_fft = np.fft.fft(p_code_a_pad)
    p_code_b_fft = np.fft.fft(p_code_b_pad)
    s_cg_pad = np.zeros(fft_length, dtype=np.complex128)

    # ------------------------------------------------------------------
    # Main processing loops
    freq_counter = 0
    for coarse_index, coarse_freq in enumerate(coarse_frequencies):
        for fine_index in range(fine_steps):
            tune_freq = coarse_freq + fine_step * fine_index
            if tune_freq < f_low:
                continue
            freq_counter += 1
            new_mix = True
            mix_offset = 0.0
            new_freq = True

            for pol_index in range(n_pol):
                cit_voltage.fill(0.0)

                for rep_index in range(repeats):
                    for comp_index in range(2):  # complementary pair
                        if multiplexing:
                            pulse_in_coarse = (
                                2 * n_pol * fine_steps * rep_index
                                + 2 * n_pol * fine_index
                                + 2 * pol_index
                                + comp_index
                            )
                        else:
                            pulse_in_coarse = (
                                2 * n_pol * repeats * fine_index
                                + 2 * n_pol * rep_index
                                + 2 * pol_index
                                + comp_index
                            )
                        pulse_index = coarse_index * pulses_per_coarse + pulse_in_coarse
                        sub_time = epoch + dt.timedelta(
                            seconds=pulse_index * ipp_seconds
                        )

                        if new_freq:
                            time_axis.append(sub_time.timestamp())
                            new_freq = False

                        try:
                            samples[:] = iq_stream.read_samples(sub_time, n_samples)
                        except Exception as exc:  # File I/O errors
                            if verbose:
                                logger.warning(f"IQ sample read failed: {exc}")
                            continue

                        # Frequency translation / decimation via FFT domain slicing
                        samples_fft = np.fft.fft(samples)
                        i_tune_precise = (iq_stream.f_center - tune_freq) / hz_per_bin
                        i_tune = int(round(i_tune_precise))
                        samples_ds[idx_front] = samples_fft[
                            (-i_tune + idx_front) % n_samples
                        ]
                        samples_ds[-idx_back.size :] = samples_fft[
                            (-1 - i_tune - idx_back) % n_samples
                        ]

                        f_intermediate = iq_stream.f_center - i_tune * hz_per_bin
                        samples_ds *= window_ds
                        samples_time = np.fft.ifft(samples_ds)

                        if mix_offset != 0.0 or new_mix:
                            mix_offset = tune_freq - f_intermediate
                            if mix_offset != 0.0:
                                mix_signal[:] = np.exp(-2j * np.pi * mix_offset * t_ds)
                                samples_time *= mix_signal
                            new_mix = False
                        else:
                            samples_time = samples_time

                        # Interpolate onto chip grid
                        samples_chip = _interp_complex(t_ds, samples_time, t_chips)

                        # Select phase code
                        if comp_index == 0:
                            code_fft = p_code_a_fft.copy()
                        else:
                            code_fft = p_code_b_fft.copy()
                        if phase_switching and (rep_index % 2 == 1):
                            code_fft *= -1

                        s_cg_pad[:n_chips] = samples_chip
                        s_cg_pad[n_chips:] = 0.0
                        s_fft = np.fft.fft(s_cg_pad)
                        s_fft *= code_fft
                        s_ifft = np.fft.ifft(s_fft)
                        correlation = s_ifft[n_pcode - 1 : n_conv]

                        cit_voltage[:, rep_index] += correlation[:n_ranges].astype(
                            np.complex64
                        )

                # Doppler FFT per range gate
                for range_index in range(n_ranges):
                    doppler_spectrum = np.fft.fft(cit_voltage[range_index, :])
                    range_power = np.max(np.abs(doppler_spectrum)) ** 2
                    if save_phase:
                        max_bin = np.argmax(np.abs(doppler_spectrum))
                        range_phase = np.angle(doppler_spectrum[max_bin])
                    if pol_index == 0:
                        iono_power_o[freq_counter - 1, range_index] += range_power
                        if save_phase and iono_phase_o is not None:
                            iono_phase_o[freq_counter - 1, range_index] += range_phase
                    else:
                        iono_power_x[freq_counter - 1, range_index] += range_power
                        if save_phase and iono_phase_x is not None:
                            iono_phase_x[freq_counter - 1, range_index] += range_phase

    iq_stream.close()
    if verbose:
        logger.info(f"Completed {program_id}")

    if not nc_flag:
        return

    # ------------------------------------------------------------------
    # NetCDF output
    pow_total = iono_power_o + iono_power_x
    with np.errstate(divide="ignore", invalid="ignore"):
        pow_db = 10.0 * np.log10(pow_total)
    pow_db[~np.isfinite(pow_db)] = np.nan
    median_val = np.nanmedian(pow_db)
    if not np.isfinite(median_val):
        median_val = 0.0
    pow_db -= median_val
    pow_db = np.clip(pow_db, 0, 255)
    pow_db = np.nan_to_num(pow_db, nan=0.0)
    pow_uint8 = pow_db.astype(np.uint8)

    freq_axis_mhz = np.asarray(freq_axis, dtype=np.float32) * 1e-6
    range_axis_km = range_axis.astype(np.float32)
    time_axis_arr = np.asarray(time_axis, dtype=np.float64)

    dataset = netCDF4.Dataset(nc_path, "w", format="NETCDF4")
    try:
        dataset.createDimension("frequency", n_freqs)
        dataset.createDimension("range", n_ranges)

        system_str = "SORcer"
        dataset.createDimension("system_strlen", len(system_str))
        system_var = dataset.createVariable("system", "S1", ("system_strlen",))
        system_var.setncatts({"long_name": "system identifier"})
        system_var[:] = np.array(list(system_str), dtype="S1")
        dataset.createDimension("id_strlen", len(program_id))
        id_var = dataset.createVariable("ID", "S1", ("id_strlen",))
        id_var.setncatts({"long_name": "transmitter/receiver identifier"})
        id_var[:] = np.array(list(program_id), dtype="S1")
        dataset.createDimension("channel_dim", 1)
        channel_var = dataset.createVariable("channel", "u1", ("channel_dim",))
        channel_var.setncatts({"long_name": "receiver channel"})
        try:
            channel_index = int("".join(filter(str.isdigit, rx_tag)))
        except ValueError:
            channel_index = 0
        channel_var[:] = np.array([channel_index], dtype="u1")
        power_var = dataset.createVariable(
            "power", "u1", ("frequency", "range"), zlib=True, complevel=9
        )
        power_var.setncatts(
            {
                "units": "dB",
                "long_name": "receive power",
                "notes": "SNR, noise estimate via median power",
            }
        )
        power_var[:, :] = pow_uint8

        if save_phase and iono_phase_o is not None:
            phase_o_var = dataset.createVariable(
                "phaseOMode", "f4", ("frequency", "range"), zlib=True, complevel=9
            )
            phase_o_var.setncatts(
                {
                    "units": "radians",
                    "long_name": "ordinary mode phase",
                }
            )
            phase_o_var[:, :] = iono_phase_o.astype(np.float32)

            if n_pol == 2 and iono_phase_x is not None:
                phase_x_var = dataset.createVariable(
                    "phaseXMode", "f4", ("frequency", "range"), zlib=True, complevel=9
                )
                phase_x_var.setncatts(
                    {
                        "units": "radians",
                        "long_name": "extraordinary mode phase",
                    }
                )
                phase_x_var[:, :] = iono_phase_x.astype(np.float32)
            print(4)
        print(5)
        freq_var = dataset.createVariable(
            "frequency", "f4", ("frequency",), zlib=True, complevel=9
        )
        freq_var.setncatts({"units": "MHz", "long_name": "radio frequency"})
        freq_var[:] = freq_axis_mhz
        print(6)
        range_var = dataset.createVariable(
            "range", "f4", ("range",), zlib=True, complevel=9
        )
        range_var.setncatts(
            {
                "units": "km",
                "long_name": "group path",
                "notes": "group delay * speed of light",
            }
        )
        range_var[:] = range_axis_km
        print(7)
        time_var = dataset.createVariable(
            "time", "f8", ("frequency",), zlib=True, complevel=9
        )
        time_var.setncatts(
            {
                "units": "seconds",
                "long_name": "Unix time",
                "notes": "since 1970-01-01T00:00:00Z, ignoring leap seconds",
            }
        )
        time_var[:] = time_axis_arr
        print(8)
    finally:
        dataset.close()


# ---------------------------------------------------------------------------
# Convenience harness for manual testing (mirrors Julia main())


def main() -> None:
    epochs = [
        dt.datetime(2023, 10, 14, 16, 0, tzinfo=_UTC) + dt.timedelta(minutes=12 * idx)
        for idx in range(0, 120 // 12)
    ]
    for epoch in epochs:
        program = {
            "Epoch": epoch,
            "FFTMode": False,
            "rxTag": "ch0",
            "Save Phase": True,
            "Signal Type": "DPS4D",
            "ID": "DPS4D_Kirtland0",
            "Freq Stepping Law": "linear",
            "Lower Freq Limit": 2e6,
            "Upper Freq Limit": 15e6,
            "Coarse Freq Step": 30e3,
            "Number of Fine Steps": 1,
            "Fine Freq step": 5e3,
            "Fine Multiplexing": False,
            "Inter-Pulse Period": 2 * 5e-3,
            "Number of Integrated Repeats": 8,
            "Interpulse Phase Switching": False,
            "Wave Form": "16-chip complementary",
            "Polarization": "O and X",
        }
        try:
            process(program, dir_iq="/media/chakras4/69F9D939661D263B", verbose=True)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            logger.error(f"Processing failed for {epoch.isoformat()}: {exc}")
            break


if __name__ == "__main__":
    main()
