"""Tests for the DFT parser with synthetic data."""

import struct
from pathlib import Path

import numpy as np

from pynasonde.digisonde.parsers.dft import DftExtractor


def _packed_byte(bit: int) -> bytes:
    return struct.pack("B", bit & 0x01)


def _synthetic_block() -> bytes:
    block = bytearray()
    header_bits = []

    def emit_bits(bits):
        header_bits.extend(bits)
        return [_packed_byte(b) + _packed_byte(0) for b in bits]

    header_bits = [0] * 128

    block.extend(bytes([0] * 128))
    block.extend(bytes([0] * 128))

    amplitude_bytes = bytes(header_bits)
    phase_bytes = bytes([0] * 128)

    for _ in range(4):
        block.extend(amplitude_bytes)
        block.extend(phase_bytes)

    block.extend(bytes([0] * (4096 - len(block))))

    return bytes(block)


def test_dft_extract_header(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/data/KR835_2023287000915.DFT"
    extractor = DftExtractor(str(path))
    extractor.extract()
