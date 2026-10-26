"""Import smoke check for one install extra (used by CI; not collected by pytest).

Usage: python tests/extras_smoke.py <core|digisonde|digisonde-scaler|digisonde-db|vipir|ocr|web>

Imports every module that the given extra is meant to make usable and exits
non-zero if any import fails, so a missing dependency in ``pyproject.toml``
shows up in the matching CI job.
"""

import importlib
import sys

MODULES = {
    "core": [
        "pynasonde",
        "pynasonde.timeutils",
        "pynasonde.vipir.analysis.polarization",
        "pynasonde.vipir.analysis.scaler",
        "pynasonde.digisonde.scaler.schema",
        "pynasonde.digisonde.scaler.solar",
    ],
    "digisonde": [
        "pynasonde.digisonde",
        "pynasonde.digisonde.parsers.sao",
        "pynasonde.digisonde.parsers.sky",
        "pynasonde.digisonde.parsers.rsf",
        "pynasonde.digisonde.cadi",
        "pynasonde.digisonde.raw",
    ],
    "digisonde-scaler": [
        "pynasonde.digisonde.scaler.core",
        "pynasonde.digisonde.scaler.viterbi",
        "pynasonde.digisonde.scaler.rfi_filter",
        "pynasonde.digisonde.scaler.hmF2",
    ],
    "digisonde-db": ["pynasonde.digisonde.didbase"],
    "vipir": [
        "pynasonde.vipir",
        "pynasonde.vipir.ngi",
        "pynasonde.vipir.riq",
        "pynasonde.vipir.analysis",
    ],
    "ocr": ["pynasonde.digisonde.parsers.image_parser"],
    "web": ["pynasonde.webhook"],
}


def main(extra: str) -> int:
    failed = []
    for name in MODULES[extra]:
        try:
            importlib.import_module(name)
            print(f"ok    {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            failed.append(name)
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODULES:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(MODULES)}>")
    sys.exit(main(sys.argv[1]))
