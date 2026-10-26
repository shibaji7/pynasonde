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

pynasonde 2.0 scope
-------------------
Only :mod:`polarization` and :mod:`scaler` are part of the 2.0 stable API
(``__all__``).  The remaining modules are unchanged and their 1.x imports
(``from pynasonde.vipir.analysis import AbsorptionAnalyzer`` etc.) keep
working, but they are not yet covered by 2.0 docs, examples or extras.
"""

_LAZY = {
    # 2.0 stable
    "PolarizationClassifier": "pynasonde.vipir.analysis.polarization",
    "PolarizationResult": "pynasonde.vipir.analysis.polarization",
    "IonogramScaler": "pynasonde.vipir.analysis.scaler",
    "ScaledParameters": "pynasonde.vipir.analysis.scaler",
    # kept for 1.x compatibility, outside the 2.0 scope
    "EsCaponImager": "pynasonde.vipir.analysis.es_imaging",
    "EsImagingResult": "pynasonde.vipir.analysis.es_imaging",
    "RiqAggregator": "pynasonde.vipir.analysis.es_imaging",
    "AbsorptionAnalyzer": "pynasonde.vipir.analysis.absorption",
    "LOFResult": "pynasonde.vipir.analysis.absorption",
    "DifferentialResult": "pynasonde.vipir.analysis.absorption",
    "TotalAbsorptionResult": "pynasonde.vipir.analysis.absorption",
    "AbsorptionProfileResult": "pynasonde.vipir.analysis.absorption",
    "TrueHeightInversion": "pynasonde.vipir.analysis.inversion",
    "EDPResult": "pynasonde.vipir.analysis.inversion",
    "IrregularityAnalyzer": "pynasonde.vipir.analysis.irregularities",
    "IrregularityProfile": "pynasonde.vipir.analysis.irregularities",
    "NeXtYZInverter": "pynasonde.vipir.analysis.nextyz",
    "NeXtYZResult": "pynasonde.vipir.analysis.nextyz",
    "WedgePlane": "pynasonde.vipir.analysis.nextyz",
    "SpreadFAnalyzer": "pynasonde.vipir.analysis.spread_f",
    "SpreadFResult": "pynasonde.vipir.analysis.spread_f",
}

__all__ = [
    "PolarizationClassifier",
    "PolarizationResult",
    "IonogramScaler",
    "ScaledParameters",
]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
