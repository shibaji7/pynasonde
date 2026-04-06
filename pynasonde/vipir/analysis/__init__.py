"""pynasonde.vipir.analysis — Physics analysis layer for ionosonde data.

This sub-package contains instrument-agnostic analysis tools that operate on
a filtered echo DataFrame (from :class:`~pynasonde.vipir.riq.parsers.filter.IonogramFilter`)
or on a scaled O-mode trace (from :class:`~pynasonde.vipir.ngi.source.Trace`).

Modules
-------
absorption
    HF radio absorption estimation: LOF index, differential O/X SNR, and
    calibrated absolute absorption profile.
    :class:`AbsorptionAnalyzer` → :class:`LOFResult`, :class:`DifferentialResult`,
    :class:`TotalAbsorptionResult`, :class:`AbsorptionProfileResult`

polarization
    O/X wave-mode separation via the PP (polarization chirality) parameter.
    :class:`PolarizationClassifier` → :class:`PolarizationResult`

spread_f
    Spread-F detection and characterisation (range, frequency, mixed).
    :class:`SpreadFAnalyzer` → :class:`SpreadFResult`

inversion
    Virtual height → true height inversion via the lamination / Abel method.
    :class:`TrueHeightInversion` → :class:`EDPResult`

scaler
    Automatic ionogram parameter scaling (foF2, foE, h'F, MUF, …).
    :class:`IonogramScaler` → :class:`ScaledParameters`

irregularities
    Small-scale irregularity spectral-index estimation from EP statistics.
    :class:`IrregularityAnalyzer` → :class:`IrregularityProfile`

nextyz
    NeXtYZ 3-D electron density inversion (Zabotin et al. 2006) using the
    Wedge-Stratified Ionosphere model and Hamiltonian ray tracing.
    :class:`NeXtYZInverter` → :class:`NeXtYZResult`

es_imaging
    High-resolution sporadic-E layer range imaging via Capon cross-spectrum
    analysis (Liu et al. 2023).  Achieves up to 10× finer range resolution
    from pulse-compressed RIQ gate data without sacrificing temporal resolution.
    :class:`EsCaponImager` → :class:`EsImagingResult` (single-cube imager)
    :class:`RiqAggregator` → :class:`EsImagingResult` (multi-file A+B+C combining:
    coherent Rx beamforming + incoherent pulse averaging + incoherent file stacking)
"""

from pynasonde.vipir.analysis.absorption import (
    AbsorptionAnalyzer,
    AbsorptionProfileResult,
    DifferentialResult,
    LOFResult,
    TotalAbsorptionResult,
)
from pynasonde.vipir.analysis.es_imaging import (
    EsCaponImager,
    EsImagingResult,
    RiqAggregator,
)
from pynasonde.vipir.analysis.inversion import EDPResult, TrueHeightInversion
from pynasonde.vipir.analysis.irregularities import (
    IrregularityAnalyzer,
    IrregularityProfile,
)
from pynasonde.vipir.analysis.nextyz import NeXtYZInverter, NeXtYZResult, WedgePlane
from pynasonde.vipir.analysis.polarization import (
    PolarizationClassifier,
    PolarizationResult,
)
from pynasonde.vipir.analysis.scaler import IonogramScaler, ScaledParameters
from pynasonde.vipir.analysis.spread_f import SpreadFAnalyzer, SpreadFResult

__all__ = [
    "EsCaponImager",
    "EsImagingResult",
    "RiqAggregator",
    "AbsorptionAnalyzer",
    "LOFResult",
    "DifferentialResult",
    "TotalAbsorptionResult",
    "AbsorptionProfileResult",
    "PolarizationClassifier",
    "PolarizationResult",
    "SpreadFAnalyzer",
    "SpreadFResult",
    "TrueHeightInversion",
    "EDPResult",
    "IonogramScaler",
    "ScaledParameters",
    "IrregularityAnalyzer",
    "IrregularityProfile",
    "NeXtYZInverter",
    "NeXtYZResult",
    "WedgePlane",
]
