"""pynasonde.vipir.analysis.es_imaging — High-resolution Es layer range imaging.

Implements the Capon cross-spectrum analysis of Liu et al. (2023) for
ionosonde RIQ data, and a multi-file aggregator that combines coherent
Rx beamforming (Option A), incoherent pulse averaging (Option B), and
incoherent file stacking (Option C) to reach WISS-equivalent SNR from
VIPIR's 4-pulse-per-file constraint.

Key design choices
------------------
* The only singularity constraint is on Z (number of subbands):
  Z ≤ (V+1)/2 ensures R_f = G·G^H/(V-Z+1) is full-rank and invertible.
  Exceeding this limit degrades imaging (rank-deficient); a warning is issued.
* K (resolution factor) is a free parameter — it only sets the output grid
  spacing Δr = r₀/K and does NOT affect the covariance matrix.

Classes
-------
EsCaponImager
    Single-cube Capon imager.  Operates on one ``(pulse_count, gate_count [, rx_count])``
    IQ cube and returns an :class:`EsImagingResult`.  Best used when many pulses
    per frequency are available (e.g. WISS, 256 pulses).

EsImagingResult
    Dataclass holding the normalised pseudospectrum (dB), height axes, and
    imaging metadata.  Provides ``summary()``, ``to_dataframe()``, ``plot()``.

RiqAggregator
    Multi-file imager implementing Options A+B+C for VIPIR's sparse-pulse
    constraint (4 pulses/file, 8 Rx channels).  Call ``combine(cubes)`` with
    a list of pre-loaded IQ cubes, or ``fit(file_list, freq_target_khz)`` to
    load from RIQ files and image in one step.
"""

from pynasonde.vipir.analysis.es_imaging.aggregator import RiqAggregator
from pynasonde.vipir.analysis.es_imaging.capon import EsCaponImager, EsImagingResult

__all__ = [
    "EsCaponImager",
    "EsImagingResult",
    "RiqAggregator",
]
