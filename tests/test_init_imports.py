"""Smoke tests for package-level __init__.py re-exports.

Each test verifies that a public symbol is reachable via the *short*,
package-level import path introduced when the sub-package ``__init__.py``
files were populated.  Deep-path imports (e.g.
``pynasonde.digisonde.parsers.sao.SaoExtractor``) remain valid and are
covered by ``test_parsers_imports.py``; this file targets only the new
convenience re-export layer.

Tests skip automatically when an optional dependency (e.g. OpenCV for
``IonogramImageExtractor``) is absent from the test environment.
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# pynasonde.digisonde.parsers  __init__
# ---------------------------------------------------------------------------
DIGISONDE_PARSERS = [
    ("pynasonde.digisonde.parsers", "SaoExtractor"),
    ("pynasonde.digisonde.parsers", "RsfExtractor"),
    ("pynasonde.digisonde.parsers", "DftExtractor"),
    ("pynasonde.digisonde.parsers", "SbfExtractor"),
    ("pynasonde.digisonde.parsers", "DvlExtractor"),
    ("pynasonde.digisonde.parsers", "ModMaxExtractor"),
    ("pynasonde.digisonde.parsers", "SkyExtractor"),
    ("pynasonde.digisonde.parsers", "EdpExtractor"),
    ("pynasonde.digisonde.parsers", "IonogramImageExtractor"),
]

# ---------------------------------------------------------------------------
# pynasonde.digisonde.datatypes  __init__
# ---------------------------------------------------------------------------
DIGISONDE_DATATYPES = [
    ("pynasonde.digisonde.datatypes", "SbfHeader"),
    ("pynasonde.digisonde.datatypes", "SbfFreuencyGroup"),
    ("pynasonde.digisonde.datatypes", "SbfDataUnit"),
    ("pynasonde.digisonde.datatypes", "SbfDataFile"),
    ("pynasonde.digisonde.datatypes", "RsfHeader"),
    ("pynasonde.digisonde.datatypes", "RsfFreuencyGroup"),
    ("pynasonde.digisonde.datatypes", "RsfDataUnit"),
    ("pynasonde.digisonde.datatypes", "RsfDataFile"),
    ("pynasonde.digisonde.datatypes", "ModMaxHeader"),
    ("pynasonde.digisonde.datatypes", "ModMaxFreuencyGroup"),
    ("pynasonde.digisonde.datatypes", "ModMaxDataUnit"),
    ("pynasonde.digisonde.datatypes", "SubCaseHeader"),
    ("pynasonde.digisonde.datatypes", "DftHeader"),
    ("pynasonde.digisonde.datatypes", "DopplerSpectra"),
    ("pynasonde.digisonde.datatypes", "DopplerSpectralBlock"),
    ("pynasonde.digisonde.datatypes", "URSI"),
    ("pynasonde.digisonde.datatypes", "SAORecord"),
    ("pynasonde.digisonde.datatypes", "SAORecordList"),
]

# ---------------------------------------------------------------------------
# pynasonde.digisonde.raw  __init__
# ---------------------------------------------------------------------------
DIGISONDE_RAW = [
    ("pynasonde.digisonde.raw", "IQStream"),
    ("pynasonde.digisonde.raw", "IonogramResult"),
    ("pynasonde.digisonde.raw", "process"),
    ("pynasonde.digisonde.raw", "RawPlots"),
    ("pynasonde.digisonde.raw", "AFRLPlots"),
]

# ---------------------------------------------------------------------------
# pynasonde.digisonde  __init__  (flat convenience re-exports)
# ---------------------------------------------------------------------------
DIGISONDE_TOP = [
    ("pynasonde.digisonde", "SaoExtractor"),
    ("pynasonde.digisonde", "RsfExtractor"),
    ("pynasonde.digisonde", "DftExtractor"),
    ("pynasonde.digisonde", "SbfExtractor"),
    ("pynasonde.digisonde", "DvlExtractor"),
    ("pynasonde.digisonde", "ModMaxExtractor"),
    ("pynasonde.digisonde", "SkyExtractor"),
    ("pynasonde.digisonde", "EdpExtractor"),
    ("pynasonde.digisonde", "DigiPlots"),
    ("pynasonde.digisonde", "SaoSummaryPlots"),
    ("pynasonde.digisonde", "SkySummaryPlots"),
    ("pynasonde.digisonde", "RsfIonogram"),
    ("pynasonde.digisonde", "to_namespace"),
    ("pynasonde.digisonde", "get_digisonde_info"),
    ("pynasonde.digisonde", "load_station_csv"),
    ("pynasonde.digisonde", "get_gridded_parameters"),
]

# ---------------------------------------------------------------------------
# pynasonde.vipir.riq.parsers  __init__
# ---------------------------------------------------------------------------
VIPIR_RIQ_PARSERS = [
    ("pynasonde.vipir.riq.parsers", "Pulset"),
    ("pynasonde.vipir.riq.parsers", "RiqDataset"),
    ("pynasonde.vipir.riq.parsers", "find_thresholds"),
    ("pynasonde.vipir.riq.parsers", "remove_morphological_noise"),
    ("pynasonde.vipir.riq.parsers", "adaptive_gain_filter"),
]

# ---------------------------------------------------------------------------
# pynasonde.vipir.riq.datatypes  __init__
# ---------------------------------------------------------------------------
VIPIR_RIQ_DATATYPES = [
    ("pynasonde.vipir.riq.datatypes", "Ionogram"),
    ("pynasonde.vipir.riq.datatypes", "PctType"),
    ("pynasonde.vipir.riq.datatypes", "SctType"),
    ("pynasonde.vipir.riq.datatypes", "StationType"),
    ("pynasonde.vipir.riq.datatypes", "TimingType"),
    ("pynasonde.vipir.riq.datatypes", "FrequencyType"),
    ("pynasonde.vipir.riq.datatypes", "RecieverType"),
    ("pynasonde.vipir.riq.datatypes", "ExciterType"),
    ("pynasonde.vipir.riq.datatypes", "MonitorType"),
]

# ---------------------------------------------------------------------------
# pynasonde.vipir.riq  __init__  (echo extractor already existed)
# ---------------------------------------------------------------------------
VIPIR_RIQ_TOP = [
    ("pynasonde.vipir.riq", "Echo"),
    ("pynasonde.vipir.riq", "EchoExtractor"),
    ("pynasonde.vipir.riq", "RiqDataset"),
]

# ---------------------------------------------------------------------------
# pynasonde.vipir.ngi  __init__
# ---------------------------------------------------------------------------
VIPIR_NGI = [
    ("pynasonde.vipir.ngi", "Trace"),
    ("pynasonde.vipir.ngi", "Dataset"),
    ("pynasonde.vipir.ngi", "DataSource"),
    ("pynasonde.vipir.ngi", "NoiseProfile"),
    ("pynasonde.vipir.ngi", "AutoScaler"),
    ("pynasonde.vipir.ngi", "Ionogram"),
    ("pynasonde.vipir.ngi", "TimeZoneConversion"),
]

# ---------------------------------------------------------------------------
# pynasonde.vipir  __init__  (combined NGI + RIQ)
# ---------------------------------------------------------------------------
VIPIR_TOP = [
    ("pynasonde.vipir", "Trace"),
    ("pynasonde.vipir", "Dataset"),
    ("pynasonde.vipir", "DataSource"),
    ("pynasonde.vipir", "NoiseProfile"),
    ("pynasonde.vipir", "AutoScaler"),
    ("pynasonde.vipir", "Ionogram"),
    ("pynasonde.vipir", "TimeZoneConversion"),
    ("pynasonde.vipir", "Echo"),
    ("pynasonde.vipir", "EchoExtractor"),
    ("pynasonde.vipir", "RiqDataset"),
]

# ---------------------------------------------------------------------------
# pynasonde  (top-level package)
# ---------------------------------------------------------------------------
PYNASONDE_TOP = [
    ("pynasonde", "__version__"),
    ("pynasonde", "SaoExtractor"),
    ("pynasonde", "RsfExtractor"),
    ("pynasonde", "DftExtractor"),
    ("pynasonde", "SbfExtractor"),
    ("pynasonde", "DvlExtractor"),
    ("pynasonde", "ModMaxExtractor"),
    ("pynasonde", "SkyExtractor"),
    ("pynasonde", "EdpExtractor"),
    ("pynasonde", "DigiPlots"),
    ("pynasonde", "SaoSummaryPlots"),
    ("pynasonde", "SkySummaryPlots"),
    ("pynasonde", "RsfIonogram"),
    ("pynasonde", "Trace"),
    ("pynasonde", "Dataset"),
    ("pynasonde", "DataSource"),
    ("pynasonde", "Echo"),
    ("pynasonde", "EchoExtractor"),
    ("pynasonde", "RiqDataset"),
    ("pynasonde", "Webhook"),
]


def _all_cases():
    """Flatten all symbol lists into a single parametrize set."""
    return (
        DIGISONDE_PARSERS
        + DIGISONDE_DATATYPES
        + DIGISONDE_RAW
        + DIGISONDE_TOP
        + VIPIR_RIQ_PARSERS
        + VIPIR_RIQ_DATATYPES
        + VIPIR_RIQ_TOP
        + VIPIR_NGI
        + VIPIR_TOP
        + PYNASONDE_TOP
    )


@pytest.mark.parametrize("module_path, attr_name", _all_cases())
def test_init_re_export(module_path: str, attr_name: str) -> None:
    """Symbol must be reachable via the package-level __init__ re-export."""
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Skipping {module_path}: missing dependency -> {exc}")

    assert hasattr(mod, attr_name), (
        f"{module_path} does not expose '{attr_name}' — "
        "check that __init__.py imports it correctly"
    )
