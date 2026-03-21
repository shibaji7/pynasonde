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

from pynasonde.digisonde.parsers.dft import DftExtractor
from pynasonde.digisonde.parsers.dvl import DvlExtractor
from pynasonde.digisonde.parsers.edp import EdpExtractor
from pynasonde.digisonde.parsers.image_parser import IonogramImageExtractor
from pynasonde.digisonde.parsers.mmm import ModMaxExtractor
from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.parsers.sbf import SbfExtractor
from pynasonde.digisonde.parsers.sky import SkyExtractor

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
