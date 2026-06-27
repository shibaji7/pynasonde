"""Regression tests for shared packed-byte parsing paths."""

import pytest

from pynasonde.digisonde.parsers.mmm import ModMaxExtractor
from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.parsers.sbf import SbfExtractor

BLOCK_SIZE = 4096


def _bcd(value: int) -> int:
    return ((value // 10) << 4) | (value % 10)


def _write_sbf_header(block: bytearray) -> None:
    block[0] = 5
    block[1] = 60
    block[2] = 1
    block[3] = _bcd(24)
    block[4] = _bcd(2)
    block[5] = _bcd(87)
    block[6] = _bcd(10)
    block[7] = _bcd(14)
    block[8] = _bcd(0)
    block[9] = _bcd(9)
    block[10] = _bcd(15)
    block[11:14] = b"K83"
    block[14:17] = b"K83"
    block[17] = _bcd(1)
    block[18] = _bcd(1)
    block[19:22] = bytes([_bcd(1), _bcd(0), _bcd(0)])
    block[22:24] = bytes([_bcd(1), _bcd(0)])
    block[24:27] = bytes([_bcd(4), _bcd(5), _bcd(0)])
    block[27:29] = bytes([_bcd(1), _bcd(0)])
    block[29] = 1
    block[30] = _bcd(1)
    block[31] = 0
    block[32] = _bcd(5)
    block[33:35] = bytes([_bcd(1), _bcd(0)])
    block[35:37] = bytes([_bcd(0), _bcd(8)])
    block[37] = _bcd(5)
    block[38:40] = bytes([_bcd(1), _bcd(28)])
    block[40:42] = bytes([_bcd(0), _bcd(0)])
    block[42] = _bcd(3)
    block[43] = _bcd(0)
    block[44] = _bcd(0)
    block[45] = _bcd(5)
    block[46] = _bcd(0)
    block[47] = _bcd(10)
    block[48] = _bcd(0)
    block[49:51] = b"\x00\x00"
    block[51:53] = (500).to_bytes(2, "little")
    block[53] = 0
    block[54:56] = bytes([_bcd(0), _bcd(8)])
    block[56:58] = bytes([_bcd(5), _bcd(0)])
    block[58:60] = bytes([_bcd(1), _bcd(28)])


def _synthetic_sbf_block() -> bytes:
    block = bytearray(BLOCK_SIZE)
    _write_sbf_header(block)
    offset = 60
    group = bytearray()
    group.extend([0x32, _bcd(1), _bcd(23), 0x23, _bcd(45), _bcd(10)])
    for _ in range(128):
        group.extend([(2 << 3) | 3, (4 << 3) | 5])
    for _ in range(15):
        block[offset : offset + len(group)] = group
        offset += len(group)
    return bytes(block)


def _synthetic_rsf_block() -> bytes:
    block = bytearray(_synthetic_sbf_block())
    block[0] = 4
    block[45] = _bcd(4)
    group_len = 6 + 2 * 128
    for offset in range(60, 60 + 15 * group_len, group_len):
        block[offset + 5] = _bcd(1)
    return bytes(block)


def _synthetic_mmm_block() -> bytes:
    block = bytearray(BLOCK_SIZE)
    block[0] = 9
    block[1] = 60
    block[2] = 1
    block[3:14] = bytes([2, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    block[14:22] = bytes([1, 1, 0, 0, 0, 0, 0, 0])
    block[22:28] = bytes([0, 0, 0, 0, 1, 0])
    block[28] = 0
    block[29] = 0
    block[30] = 0x00
    block[31] = 0x00
    block[32] = 0x03
    block[33] = 0x00
    block[34] = 0x20
    block[35:40] = bytes([0, 1, 1, 1, 5])
    block[40:56] = bytes([0] * 16)
    block[56] = 1  # range increment table index
    block[57] = 2  # range start table index
    block[58] = 0
    block[59] = 0

    offset = 60
    group = bytearray()
    group.extend([0x31, _bcd(5), _bcd(10), 0x12, _bcd(34), 2])
    group.extend([(3 << 4) | 0] * 128)
    for _ in range(30):
        block[offset : offset + len(group)] = group
        offset += len(group)
    return bytes(block)


def test_sbf_extract_reads_two_packed_bytes_per_range_bin(tmp_path):
    path = tmp_path / "TEST_2024287000915.SBF"
    path.write_bytes(_synthetic_sbf_block())

    extractor = SbfExtractor(str(path))
    extractor.extract()
    df = extractor.to_pandas()

    assert len(df) == 15 * 128
    first = df.iloc[0]
    assert first["amplitude"] == pytest.approx(6.0)
    assert first["dop_num"] == 3
    assert first["phase"] == pytest.approx(45.0)
    assert first["azimuth"] == 300


def test_rsf_extract_uses_shared_block_parser_and_adds_azimuth_directions(tmp_path):
    path = tmp_path / "TEST_2024287000915.RSF"
    path.write_bytes(_synthetic_rsf_block())

    extractor = RsfExtractor(str(path))
    extractor.extract()
    df = extractor.to_pandas()

    assert len(df) == 15 * 128
    assert "azm_directions" in df.columns
    first = df.iloc[0]
    assert first["amplitude"] == pytest.approx(6.0)
    assert first["dop_num"] == 3
    assert first["phase"] == pytest.approx(45.0)
    assert first["azimuth"] == 300
    assert first["azm_directions"] == "NW"


def test_mmm_extract_decodes_four_four_amplitude_channel_bytes(tmp_path):
    path = tmp_path / "TEST_2024001000000.MMM"
    path.write_bytes(_synthetic_mmm_block())

    extractor = ModMaxExtractor(str(path))
    extractor.extract()
    df = extractor.to_pandas()

    assert len(df) == 30 * (128 - 18)
    first = df.iloc[0]
    assert first["frequency_mhz"] == pytest.approx(5.1)
    assert first["amplitude_dB"] == pytest.approx(18.0)
    assert first["channel"] == 0
    assert first["polarization"] == "X"
    assert first["doppler_channel"] == 1
    assert first["doppler_hz"] == pytest.approx(0.625)
