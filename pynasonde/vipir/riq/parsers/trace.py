from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from pynasonde.vipir.riq.datatypes.sct import SctType


def compute_phase(i, q):
    return np.arctan2(q, i) % (2 * np.pi)


def extract_echo_traces(
    sct: SctType,
    pulse_i: np.ndarray,
    pulse_q: np.ndarray,
    f1_freq_low: Optional[int] = 0,
    f1_freq_high: Optional[int] = 20000,
    f1_range_low: Optional[int] = 70,
    f1_range_high: Optional[int] = 1000,
    f2_freq_max: Optional[int] = 13000,
    pulse_block_min: Optional[float] = 0.0,
    snr_variance_threshold: Optional[int] = 1.1,
    doppler_time_offsets: Optional[np.array] = np.array(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    ),
):
    frequencies = sct.frequency.base_table
    freq_id_range = np.argwhere(
        (frequencies < f1_freq_high) & (frequencies > f1_freq_low)
    )
    pulse_i_range, pulse_q_range = (
        pulse_i[:, f1_range_low:f1_range_high, :],
        pulse_q[:, f1_range_low:f1_range_high, :],
    )
    power = np.sqrt(pulse_i_range**2 + pulse_q_range**2)
    n_pulses = power.shape[0]

    p_block_size = int(n_pulses - pulse_block_min)
    v_coef = 300000 / (4 * np.pi)

    peak_gates, phases, xphases, yphases = (
        np.zeros(p_block_size),
        np.zeros(p_block_size),
        np.zeros(p_block_size),
        np.zeros(p_block_size),
    )

    peak_gates = np.nanargmax(20 * np.log10(power[..., 0]), axis=1)
    print(power.shape)
    idx = np.arange(power.shape[0])
    ew_i = pulse_i_range[idx, peak_gates, 0]
    ew_q = pulse_q_range[idx, peak_gates, 0]
    xphases = compute_phase(ew_i, ew_q)

    return
