from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DftHeader:
    record_type: int = None
    number_of_bytes: int = 0


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
