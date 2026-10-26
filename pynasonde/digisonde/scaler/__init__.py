"""Generic Digisonde ionogram scaling utilities.

The scaler package is DataFrame-native: RSF, SBF, MMM, and future Digisonde
parser outputs should first be normalized to the canonical echo schema before
running filtering or autoscaling algorithms.
"""

# Names resolve lazily (PEP 562): ``core``, ``viterbi`` and ``rfi_filter`` need
# scikit-learn/scipy/joblib (``pip install "pynasonde[digisonde-scaler]"``), while
# ``schema`` and ``solar`` work with the core install.
_LAZY = {
    "CANONICAL_COLUMNS": "pynasonde.digisonde.scaler.schema",
    "REQUIRED_CANONICAL_COLUMNS": "pynasonde.digisonde.scaler.schema",
    "EchoSchema": "pynasonde.digisonde.scaler.schema",
    "detect_echo_schema": "pynasonde.digisonde.scaler.schema",
    "normalize_echo_dataframe": "pynasonde.digisonde.scaler.schema",
    "normalize_mode_values": "pynasonde.digisonde.scaler.schema",
    "AdaptiveAmpGate": "pynasonde.digisonde.scaler.rfi_filter",
    "ColumnDensityFilter": "pynasonde.digisonde.scaler.rfi_filter",
    "NeighborPredictedFilter": "pynasonde.digisonde.scaler.rfi_filter",
    "VerticalRfiFilter": "pynasonde.digisonde.scaler.rfi_filter",
    "IonoscaleBase": "pynasonde.digisonde.scaler.core",
    "IonoscaleResult": "pynasonde.digisonde.scaler.core",
    "IonogramAutoScaler": "pynasonde.digisonde.scaler.core",
    "ViterbiTracer": "pynasonde.digisonde.scaler.viterbi",
    "ScaleHmF2": "pynasonde.digisonde.scaler.hmF2",
    "scale_dataframe": "pynasonde.digisonde.scaler.core",
    "scale_dataframes": "pynasonde.digisonde.scaler.core",
    "scale_dataframe_with_preset": "pynasonde.digisonde.scaler.core",
    "scale_dataframes_with_preset": "pynasonde.digisonde.scaler.core",
    "ScalerPreset": "pynasonde.digisonde.scaler.presets",
    "get_viterbi_preset": "pynasonde.digisonde.scaler.presets",
    "solar_zenith_angle": "pynasonde.digisonde.scaler.solar",
    "classify_solar_regime": "pynasonde.digisonde.scaler.solar",
    "SZA_DAY_MAX_DEG": "pynasonde.digisonde.scaler.solar",
    "SZA_TWILIGHT_MAX_DEG": "pynasonde.digisonde.scaler.solar",
    "SOLAR_PARAMETER_SETS": "pynasonde.digisonde.scaler.presets",
    "BASE_LAYER_SCALER_KWARGS": "pynasonde.digisonde.scaler.presets",
    "TUNING_PRESETS": "pynasonde.digisonde.scaler.presets",
    "despike_scaled_parameters": "pynasonde.digisonde.scaler.postprocess",
}

__all__ = [
    "CANONICAL_COLUMNS",
    "REQUIRED_CANONICAL_COLUMNS",
    "EchoSchema",
    "detect_echo_schema",
    "normalize_echo_dataframe",
    "normalize_mode_values",
    "AdaptiveAmpGate",
    "ColumnDensityFilter",
    "NeighborPredictedFilter",
    "VerticalRfiFilter",
    "IonoscaleBase",
    "IonoscaleResult",
    "IonogramAutoScaler",
    "ViterbiTracer",
    "ScaleHmF2",
    "scale_dataframe",
    "scale_dataframes",
    "scale_dataframe_with_preset",
    "scale_dataframes_with_preset",
    "ScalerPreset",
    "get_viterbi_preset",
    "solar_zenith_angle",
    "classify_solar_regime",
    "SZA_DAY_MAX_DEG",
    "SZA_TWILIGHT_MAX_DEG",
    "SOLAR_PARAMETER_SETS",
    "BASE_LAYER_SCALER_KWARGS",
    "TUNING_PRESETS",
    "despike_scaled_parameters",
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
