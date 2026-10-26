"""Digisonde file-format parsers.

This sub-package collects one extractor class per Digisonde binary/text
format.  Each extractor follows the same lifecycle:

1. Instantiate with a file path (and optional flags to parse timestamps
   and station metadata from the filename).
2. Call ``extract()`` to populate internal structures.
3. Call a conversion helper (``to_pandas()``, ``load_*_files()``, etc.)
   to obtain a tidy ``pandas.DataFrame``.

Exported names
--------------
.. autosummary::

    SaoExtractor
    RsfExtractor
    DftExtractor
    SbfExtractor
    DvlExtractor
    ModMaxExtractor
    SkyExtractor
    EdpExtractor
    IonogramImageExtractor
"""

# Names resolve lazily (PEP 562): ``IonogramImageExtractor`` needs the optional
# ``ocr`` extra (OpenCV, pytesseract) and must not be imported for other parsers.
_LAZY = {
    "DftExtractor": "pynasonde.digisonde.parsers.dft",
    "DvlExtractor": "pynasonde.digisonde.parsers.dvl",
    "EdpExtractor": "pynasonde.digisonde.parsers.edp",
    "IonogramImageExtractor": "pynasonde.digisonde.parsers.image_parser",
    "ModMaxExtractor": "pynasonde.digisonde.parsers.mmm",
    "RsfExtractor": "pynasonde.digisonde.parsers.rsf",
    "SaoExtractor": "pynasonde.digisonde.parsers.sao",
    "SbfExtractor": "pynasonde.digisonde.parsers.sbf",
    "SkyExtractor": "pynasonde.digisonde.parsers.sky",
}

__all__ = [
    "SaoExtractor",
    "RsfExtractor",
    "DftExtractor",
    "SbfExtractor",
    "DvlExtractor",
    "ModMaxExtractor",
    "SkyExtractor",
    "EdpExtractor",
    "IonogramImageExtractor",
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
