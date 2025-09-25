from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from pynasonde.vipir.riq.datatypes.sct import SctType


def compute_phase(i, q):
    return np.arctan2(q, i) % (2 * np.pi)


def get_clean_iq_by_heights(
    pulse_i: np.ndarray,
    pulse_q: np.ndarray,
    f1_range_low: Optional[int] = 70,
    f1_range_high: Optional[int] = 1000,
    # range_dev 1.5 because 3km/2 for speed of light back and forth across 10us rangegate
    range_dev: Optional[float] = 1.5,
):
    f1_rlow = int(np.floor(f1_range_low / range_dev))
    f1_rhigh = int(np.ceil(f1_range_high / range_dev))
    pulse_i_range, pulse_q_range = (
        pulse_i[:, f1_rlow:f1_rhigh, :],
        pulse_q[:, f1_rlow:f1_rhigh, :],
    )
    power = np.sqrt(pulse_i_range**2 + pulse_q_range**2)
    n_pulses = power.shape[0]
    return pulse_i_range, pulse_q_range, power, n_pulses, f1_rlow, f1_rhigh


def extract_echo_traces(
    sct: SctType,
    pulse_i: np.ndarray,
    pulse_q: np.ndarray,
    f1_range_low: Optional[int] = 70,
    f1_range_high: Optional[int] = 1000,
    snr_variance_threshold: Optional[float] = 1.1,
    f2max: Optional[float] = 13000,
    # range_dev 1.5 because 3km/2 for speed of light back and forth across 10us rangegate
    range_dev: Optional[float] = 1.5,
):
    (pulse_i_range, pulse_q_range, power, n_pulses, _, _) = get_clean_iq_by_heights(
        pulse_i,
        pulse_q,
        f1_range_low=f1_range_low,
        f1_range_high=f1_range_high,
        range_dev=range_dev,
    )

    if sct.frequency.tune_type == 1:
        peak_gates = np.nanargmax(20 * np.log10(power[..., 0]), axis=1)
        num_sets = int(n_pulses / sct.frequency.pulse_count)
        block_freq = sct.frequency.base_table[:num_sets]
    elif sct.frequency.tune_type == 2:
        raise NotImplementedError("TUNE_TYPE 2 not implemented yet")
    elif sct.frequency.tune_type == 3:
        raise NotImplementedError("TUNE_TYPE 3 not implemented yet")
    elif sct.frequency.tune_type >= 4:
        peak_gates = np.nanargmax(20 * np.log10(power[..., 0]), axis=1)
        num_sets = int(n_pulses / sct.frequency.pulse_count)
        block_freq = sct.frequency.base_table[::2]
    else:
        raise ValueError(f"Unknown frequency tune type: {sct.frequency.tune_type}")
    clean_range = np.zeros(num_sets, dtype=int)
    # Reshape peak_gates to (num_sets, pulse_count)
    range_blocks = peak_gates[: num_sets * sct.frequency.pulse_count].reshape(
        num_sets, sct.frequency.pulse_count
    )
    # Compute variance and mean along pulse axis
    variances = np.var(range_blocks, axis=1)
    means = np.mean(range_blocks, axis=1)
    # Assign means to clean_range where variance is below threshold
    mask = variances < snr_variance_threshold
    clean_range[mask] = means[mask]

    good_index = np.where(
        (block_freq < f2max)
        & (block_freq > 0)
        & (clean_range > 0)
        & (clean_range >= f1_range_low)
        & (clean_range <= f1_range_high)
    )[0]

    compute_phase_velocity(
        sct,
        pulse_i,
        pulse_q,
        f1_range_low=f1_range_low,
        f1_range_high=f1_range_high,
        snr_variance_threshold=snr_variance_threshold,
        f2max=f2max,
        range_dev=range_dev,
    )
    return good_index


def compute_phase_velocity(
    sct: SctType,
    pulse_i: np.ndarray,
    pulse_q: np.ndarray,
    f1_range_low: Optional[int] = 70,
    f1_range_high: Optional[int] = 1000,
    snr_variance_threshold: Optional[float] = 1.1,
    f2max: Optional[float] = 13000,
    # range_dev 1.5 because 3km/2 for speed of light back and forth across 10us rangegate
    range_dev: Optional[float] = 1.5,
):
    (pulse_i_range, pulse_q_range, power, n_pulses, _, _) = get_clean_iq_by_heights(
        pulse_i,
        pulse_q,
        f1_range_low=f1_range_low,
        f1_range_high=f1_range_high,
        range_dev=range_dev,
    )

    if sct.frequency.tune_type == 1:
        peak_gates = np.nanargmax(20 * np.log10(power[..., 0]), axis=1)
    elif sct.frequency.tune_type == 2:
        raise NotImplementedError("TUNE_TYPE 2 not implemented yet")
    elif sct.frequency.tune_type == 3:
        raise NotImplementedError("TUNE_TYPE 3 not implemented yet")
    elif sct.frequency.tune_type >= 4:
        peak_gates = np.nanargmax(20 * np.log10(power[..., 0]), axis=1)
    else:
        raise ValueError(f"Unknown frequency tune type: {sct.frequency.tune_type}")

    pulse_q_peakrange, pulse_i_peakrange = (
        pulse_q_range[:, peak_gates, :],
        pulse_i_range[:, peak_gates, :],
    )
    phase = np.unwrap(np.arctan2(pulse_q_peakrange, pulse_i_peakrange))
    # xphases, yphases = (
    #     phase[..., 0],  # E-W Phases
    #     phase[..., 1],  # N-S Phases
    # )
    n_pulses, n_gates, n_ch = pulse_q_range.shape
    peak_gates = np.clip(peak_gates, 0, n_gates - 1)
    idx = np.arange(n_pulses)

    pulse_q_peakrange = pulse_q_range[idx, peak_gates, :]  # shape (n_pulses, 2)
    print(pulse_q_peakrange.shape)
    pulse_i_peakrange = pulse_i_range[idx, peak_gates, :]  # shape (n_pulses, 2)
    return
