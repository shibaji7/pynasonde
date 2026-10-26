"""VIPIR ionosonde data processing.

This package provides everything needed to work with VIPIR (Vertical
Incidence Pulsed Ionospheric Radar) data.  It is split into two
sub-packages:

:mod:`pynasonde.vipir.ngi`
    Scaled NGI ionogram data — data structures, autoscaling, plotting, and
    time-zone utilities.

:mod:`pynasonde.vipir.riq`
    Raw RIQ IQ data — binary parser, datatypes, and the Dynasonde-style
    seven-parameter echo extractor.

Exported names (convenience re-exports)
----------------------------------------
From NGI:
    ``Trace``, ``Dataset``, ``DataSource``,
    ``NoiseProfile``, ``AutoScaler``,
    ``Ionogram``, ``TimeZoneConversion``

From RIQ:
    ``Echo``, ``EchoExtractor``, ``IonogramFilter``, ``RiqDataset``
"""

# Names resolve lazily (PEP 562) so importing a submodule such as
# ``pynasonde.vipir.analysis.polarization`` does not load the NGI/RIQ stacks.
_LAZY = {
    "AutoScaler": "pynasonde.vipir.ngi",
    "Dataset": "pynasonde.vipir.ngi",
    "DataSource": "pynasonde.vipir.ngi",
    "Ionogram": "pynasonde.vipir.ngi",
    "NoiseProfile": "pynasonde.vipir.ngi",
    "Trace": "pynasonde.vipir.ngi",
    "TimeZoneConversion": "pynasonde.timeutils",
    "Echo": "pynasonde.vipir.riq",
    "EchoExtractor": "pynasonde.vipir.riq",
    "IonogramFilter": "pynasonde.vipir.riq",
    "RiqDataset": "pynasonde.vipir.riq",
}

__all__ = [
    # NGI
    "Trace",
    "Dataset",
    "DataSource",
    "NoiseProfile",
    "AutoScaler",
    "Ionogram",
    "TimeZoneConversion",
    # RIQ
    "Echo",
    "EchoExtractor",
    "IonogramFilter",
    "RiqDataset",
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
