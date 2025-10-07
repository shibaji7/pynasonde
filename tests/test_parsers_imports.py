"""Import-oriented smoke tests for Digisonde parser modules.

These checks ensure each parser can be imported and exposes the primary
extractor class. Tests skip automatically if optional third-party
dependencies (e.g., OpenCV) are missing in the test environment.
"""

import importlib
from typing import Tuple

import pytest

PARSER_MODULES: Tuple[Tuple[str, str], ...] = (
    ("pynasonde.digisonde.parsers.sao", "SaoExtractor"),
    ("pynasonde.digisonde.parsers.sky", "SkyExtractor"),
    ("pynasonde.digisonde.parsers.rsf", "RsfExtractor"),
    ("pynasonde.digisonde.parsers.dvl", "DvlExtractor"),
    ("pynasonde.digisonde.parsers.dft", "DftExtractor"),
    ("pynasonde.digisonde.parsers.edp", "EdpExtractor"),
    ("pynasonde.digisonde.parsers.mmm", "ModMaxExtractor"),
    ("pynasonde.digisonde.parsers.sbf", "SbfExtractor"),
)


@pytest.mark.parametrize("module_name, attr_name", PARSER_MODULES)
def test_parser_imports(module_name: str, attr_name: str) -> None:
    """Each parser module should import and expose its primary extractor."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when deps missing
        pytest.skip(f"Skipping {module_name}: missing dependency -> {exc}")

    assert hasattr(
        module, attr_name
    ), f"{module_name} does not expose expected attribute {attr_name}"
