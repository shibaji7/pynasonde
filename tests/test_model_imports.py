"""Ensure selected model modules import without crashing."""

import importlib
import types

MODULES = [
    "pynasonde.model.absorption.constants",
    "pynasonde.model.absorption.collisions",
    "pynasonde.model.absorption.dispersion_relations",
    "pynasonde.model.polan.polan_utils",
]


def test_model_modules_importable():
    for name in MODULES:
        module = importlib.import_module(name)
        assert isinstance(module, types.ModuleType)
