"""Signal-processing helpers for extracting echo traces from VIPIR IQ pulse data.

Functions here are reused by the RIQ parsers to normalize raw I/Q samples,
select candidate echo ranges, and derive phase-related diagnostics.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from pynasonde.vipir.riq.datatypes.sct import SctType


def compute_phase(i, q):
    """Return wrapped phase angles for the provided I/Q samples.

    Args:
        i: In-phase component array.
        q: Quadrature component array.

    Returns:
        Array of phase angles wrapped to the ``[0, 2π)`` interval.
    """
    return np.arctan2(q, i) % (2 * np.pi)


def get_clean_iq_by_heights(
    pulse_i: np.ndarray,
    pulse_q: np.ndarray,
    f1_range_low: Optional[int] = 70,
    f1_range_high: Optional[int] = 1000,
    # range_dev 1.5 because 3km/2 for speed of light back and forth across 10us rangegate
    range_dev: Optional[float] = 1.5,
):
    """Slice the pulse data to focus on a selectable height band.

    Args:
        pulse_i: Full I-channel pulse data (frequency × range × receiver).
        pulse_q: Full Q-channel pulse data.
        f1_range_low: Lower bound of the height window (km).
        f1_range_high: Upper bound of the height window (km).
        range_dev: Conversion factor from km to gate index (default 1.5).

    Returns:
        Tuple `(pulse_i_range, pulse_q_range, power, n_pulses, f1_rlow, f1_rhigh)`
        containing sliced arrays, computed power, number of pulses, and the
        index bounds applied.
    """
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
    tag_rx_num: Optional[int] = 0,
):
    """Identify candidate echo traces that satisfy the configured thresholds.

    Args:
        sct: SCT metadata describing the capture configuration.
        pulse_i: I-channel samples (flattened across pulse sets).
        pulse_q: Q-channel samples (flattened across pulse sets).
        f1_range_low: Lower height bound used for gating.
        f1_range_high: Upper height bound used for gating.
        snr_variance_threshold: Maximum allowed variance across pulses.
        f2max: Upper frequency limit for valid traces.
        range_dev: Conversion factor from km to gate index.
        tag_rx_num: Receiver number to tag in the output (default 0).

    Returns:
        Indices into the `block_freq` array that passed the selection criteria.
    """
    (pulse_i_range, pulse_q_range, power, n_pulses, _, _) = get_clean_iq_by_heights(
        pulse_i,
        pulse_q,
        f1_range_low=f1_range_low,
        f1_range_high=f1_range_high,
        range_dev=range_dev,
    )

    if sct.frequency.tune_type == 1:
        peak_gates = np.nanargmax(20 * np.log10(power[..., tag_rx_num]), axis=1)
        num_sets = int(n_pulses / sct.frequency.pulse_count)
        block_freq = sct.frequency.base_table[:num_sets]
    elif sct.frequency.tune_type == 2:
        raise NotImplementedError("TUNE_TYPE 2 not implemented yet")
    elif sct.frequency.tune_type == 3:
        raise NotImplementedError("TUNE_TYPE 3 not implemented yet")
    elif sct.frequency.tune_type >= 4:
        peak_gates = np.nanargmax(20 * np.log10(power[..., tag_rx_num]), axis=1)
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
        tag_rx_num=tag_rx_num,
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
    tag_rx_num: Optional[int] = 0,
) -> Optional[Dict[str, np.ndarray]]:
    """Compute per-channel phase velocities at the detected peak gates.

    Args:
        sct: SCT metadata describing the capture configuration.
        pulse_i: I-channel samples (flattened across pulse sets).
        pulse_q: Q-channel samples (flattened across pulse sets).
        f1_range_low: Lower height bound used for gating.
        f1_range_high: Upper height bound used for gating.
        snr_variance_threshold: Maximum allowed variance across pulses.
        f2max: Upper frequency limit for valid traces.
        range_dev: Conversion factor from km to gate index.
        tag_rx_num: Receiver number to tag in the output (default 0).

    Returns:
        Optional[Dict[str, np.ndarray]]: Collection of intermediate arrays including the
        unwrapped phase, regression fit parameters, derived Doppler frequency,
        and estimated line-of-sight velocity per pulse set and channel. Returns
        ``None`` when insufficient pulse data is available.
    """
    logger.info("Computing phase velocity....")
    (pulse_i_range, pulse_q_range, power, n_pulses, _, _) = get_clean_iq_by_heights(
        pulse_i,
        pulse_q,
        f1_range_low=f1_range_low,
        f1_range_high=f1_range_high,
        range_dev=range_dev,
    )

    if sct.frequency.tune_type == 1:
        peak_gates = np.nanargmax(20 * np.log10(power[..., tag_rx_num]), axis=1)
    elif sct.frequency.tune_type == 2:
        raise NotImplementedError("TUNE_TYPE 2 not implemented yet")
    elif sct.frequency.tune_type == 3:
        raise NotImplementedError("TUNE_TYPE 3 not implemented yet")
    elif sct.frequency.tune_type >= 4:
        peak_gates = np.nanargmax(20 * np.log10(power[..., tag_rx_num]), axis=1)
    else:
        raise ValueError(f"Unknown frequency tune type: {sct.frequency.tune_type}")

    n_pulses, n_gates, n_ch = pulse_q_range.shape
    if n_pulses == 0 or n_ch == 0:
        return None

    pulse_count = int(getattr(sct.frequency, "pulse_count", 0))
    if pulse_count <= 1:
        return None

    peak_gates = np.clip(peak_gates, 0, n_gates - 1)
    idx = np.arange(n_pulses)
    pulse_i_peakrange = pulse_i_range[idx, peak_gates, :]
    pulse_q_peakrange = pulse_q_range[idx, peak_gates, :]

    # Complex representation simplifies phase computation
    complex_peak = pulse_i_peakrange + 1j * pulse_q_peakrange
    phase = np.angle(complex_peak)

    freq_table = np.asarray(getattr(sct.frequency, "base_table", []), dtype=float)
    if sct.frequency.tune_type == 1:
        freq_view = freq_table
    elif sct.frequency.tune_type >= 4:
        freq_view = freq_table[::2]
    else:
        raise NotImplementedError(f"TUNE_TYPE {sct.frequency.tune_type} not supported")

    max_sets = freq_view.size
    num_sets = min(n_pulses // pulse_count, max_sets)
    usable_samples = num_sets * pulse_count
    if usable_samples == 0:
        return None

    phase = phase[:usable_samples].reshape(num_sets, pulse_count, n_ch)
    # Unwrap phase along the pulse/time axis to remove 2π discontinuities
    phase_unwrapped = np.unwrap(phase, axis=1)

    # Build the time vector using PRI (microseconds) -> seconds.
    pri_us = float(getattr(sct.timing, "pri", 0.0) or 0.0)
    dt = pri_us * 1e-6 if pri_us > 0 else 1.0
    pulse_times = (np.arange(pulse_count, dtype=float) * dt) + dt

    # Prepare linear regression components for slope/intercept.
    x = pulse_times
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2)
    if denom == 0:
        return None

    y_mean = np.mean(phase_unwrapped, axis=1)
    y_centered = phase_unwrapped - y_mean[:, None, :]
    slopes = np.sum(y_centered * x_centered[None, :, None], axis=1) / denom
    intercepts = y_mean - slopes * x.mean()

    # Convert phase rate (rad/s) to Doppler Hz and nominal line-of-sight velocity.
    doppler_hz = slopes / (2 * np.pi)
    block_freq = freq_view[:num_sets]

    block_freq_hz = block_freq * 1e3  # base_table is stored in kHz
    c = 299_792_458.0  # speed of light (m/s)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity_mps = (doppler_hz * c)[:, :] / (2 * block_freq_hz[:, None])

    return {
        "phase_unwrapped": phase_unwrapped[..., tag_rx_num],
        "phase_slopes": slopes[..., tag_rx_num],
        "phase_intercepts": intercepts[..., tag_rx_num],
        "doppler_hz": doppler_hz[..., tag_rx_num],
        "velocity_mps": velocity_mps[..., tag_rx_num],
        "time_axis_s": pulse_times,
    }
