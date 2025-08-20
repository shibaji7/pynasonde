from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from pynasonde.vipir.riq.datatypes.sct import SctType


def extract_echo_traces(
    sct: SctType,
    pulse_set: List,
    f1_freq_low: Optional[int] = 0,
    f1_freq_high: Optional[int] = 20000,
    f1_range_low: Optional[int] = 70,
    f1_range_high: Optional[int] = 1000,
    f2_freq_max: Optional[int] = 13000,
    snr_variance_threshold: Optional[int] = 1.1,
    doppler_time_offsets: Optional[np.array] = np.array(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    ),
):

    # print([pulse_set[i]["pri"].frequency for i in range(len(pulse_set))])
    # IQ data have shape (n_pulses, n_gates, n_receivers)
    i_data, q_data = (
        np.array([p["pri"].a_scan[:, :, 0] for p in pulse_set]),
        np.array([p["pri"].a_scan[:, :, 1] for p in pulse_set]),
    )
    # compute power from I/Q data sqrt(I^2 + Q^2)
    power = np.nan_to_num(
        20 * np.log10(np.sqrt(i_data**2 + q_data**2)), nan=0.0, posinf=0.0, neginf=0.0
    )
    print(power.shape)
    power = np.nanmean(power, axis=0)
    print(power.shape)
    noise = np.nanmean(power, axis=0).reshape((1, 2))
    print(noise.shape)
    power = np.mean(power, axis=1)
    print(power.shape)
    # v_coef = 300000 / (4 * np.pi)

    # ranges = np.zeros(power.shape[0])
    # phases = np.zeros(power.shape[0])
    # xphases = np.zeros(power.shape[0])
    # yphases = np.zeros(power.shape[0])
    # peakgates = np.zeros(power.shape[0])
    # # Compute SNR for each pulse and 1st receiver
    # for i in range(power.shape[0]):
    #     print(i)
    #     line = 20 * np.log10(power[i, :, 0])
    #     line = np.nan_to_num(line, nan=0.0, posinf=0.0, neginf=0.0)
    #     maxval = max(line)
    #     peakrgt = np.argwhere(line == maxval)
    #     peakgates[i] = peakrgt[0][0]
    #     ranges[i] = peakrgt[0][0]

    # if np.var(ranges) >= snr_variance_threshold:
    #     logger.info(f"High Variance in Range: {np.var(ranges)}")
    return power
