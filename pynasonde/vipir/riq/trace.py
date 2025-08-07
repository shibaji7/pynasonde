import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

from pynasonde.vipir.riq.headers.sct import SctType


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
    
    return
