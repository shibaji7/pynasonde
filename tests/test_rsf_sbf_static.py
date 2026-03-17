"""Tests for static methods of RsfExtractor, SbfExtractor, and ModMaxExtractor."""

import pytest

from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.parsers.sbf import SbfExtractor
from pynasonde.digisonde.parsers.mmm import ModMaxExtractor


# ---------------------------------------------------------------------------
# RsfExtractor static methods
# ---------------------------------------------------------------------------

class TestRsfUnpackBcd:
    def test_int_format_zero(self):
        assert RsfExtractor.unpack_bcd(0x00, format="int") == 0

    def test_int_format_bcd_12(self):
        # 0x12 → high=1, low=2 → 12
        assert RsfExtractor.unpack_bcd(0x12, format="int") == 12

    def test_int_format_bcd_99(self):
        # 0x99 → high=9, low=9 → 99
        assert RsfExtractor.unpack_bcd(0x99, format="int") == 99

    def test_tuple_format(self):
        high, low = RsfExtractor.unpack_bcd(0x34, format="tuple")
        assert high == 3
        assert low == 4

    def test_tuple_format_zero(self):
        high, low = RsfExtractor.unpack_bcd(0x00, format="tuple")
        assert high == 0
        assert low == 0

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid format"):
            RsfExtractor.unpack_bcd(0x12, format="hex")


class TestRsfUnpack53:
    def test_zero_byte(self):
        result = RsfExtractor.unpack_5_3(0)
        assert result == [0, 0]

    def test_splits_correctly(self):
        # byte = 0b00101_011 = 0x2B = 43
        # 5-bit high: 43 >> 3 = 5; 3-bit low: 43 & 0b111 = 3
        result = RsfExtractor.unpack_5_3(0b00101011)
        assert result[0] == 5
        assert result[1] == 3

    def test_max_byte(self):
        # 0xFF: high = (255>>3) & 0b11111 = 31; low = 255 & 0b111 = 7
        result = RsfExtractor.unpack_5_3(0xFF)
        assert result[0] == 31
        assert result[1] == 7


class TestRsfAddDictsSelectedKeys:
    def test_merge_all_keys(self):
        d0 = {"a": 1}
        du = {"b": 2, "c": 3}
        result = RsfExtractor(None).add_dicts_selected_keys.__func__(
            None, d0, du, keys=None
        ) if False else {**d0, **du}
        # Test the static method properly:
        ex = object.__new__(RsfExtractor)  # skip __init__
        result = ex.add_dicts_selected_keys({"a": 1}, {"b": 2, "c": 3}, keys=None)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_merge_selected_keys(self):
        ex = object.__new__(RsfExtractor)
        result = ex.add_dicts_selected_keys(
            {"a": 1}, {"b": 2, "c": 3, "d": 4}, keys=["b", "d"]
        )
        assert result == {"a": 1, "b": 2, "d": 4}
        assert "c" not in result


# ---------------------------------------------------------------------------
# SbfExtractor instance methods (not @staticmethod in sbf.py)
# ---------------------------------------------------------------------------

class TestSbfUnpackBcd:
    def _ex(self):
        return object.__new__(SbfExtractor)

    def test_int_format(self):
        assert self._ex().unpack_bcd(0x56, format="int") == 56

    def test_tuple_format(self):
        high, low = self._ex().unpack_bcd(0x78, format="tuple")
        assert high == 7
        assert low == 8

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid format"):
            self._ex().unpack_bcd(0x12, format="bad")


class TestSbfUnpack53:
    def test_basic(self):
        ex = object.__new__(SbfExtractor)
        result = ex.unpack_5_3(0b00101011)
        assert result[0] == 5
        assert result[1] == 3


class TestSbfAddDicts:
    def test_merge(self):
        ex = object.__new__(SbfExtractor)
        result = ex.add_dicts_selected_keys({"x": 9}, {"y": 8, "z": 7}, keys=["y"])
        assert result == {"x": 9, "y": 8}


# ---------------------------------------------------------------------------
# ModMaxExtractor instance methods (not @staticmethod in mmm.py)
# ---------------------------------------------------------------------------

class TestMmmUnpackBcd:
    def _ex(self):
        return object.__new__(ModMaxExtractor)

    def test_int_format(self):
        assert self._ex().unpack_bcd(0x42, format="int") == 42

    def test_tuple_format(self):
        high, low = self._ex().unpack_bcd(0xAB, format="tuple")
        assert high == 10
        assert low == 11

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            self._ex().unpack_bcd(0x00, format="wrong")


class TestMmmUnpack53:
    def test_zero(self):
        ex = object.__new__(ModMaxExtractor)
        result = ex.unpack_5_3(0)
        assert result == [0, 0]

    def test_non_zero(self):
        ex = object.__new__(ModMaxExtractor)
        result = ex.unpack_5_3(0xFF)
        assert result[0] == 31
        assert result[1] == 7


class TestMmmAddDicts:
    def test_merge_selected(self):
        ex = object.__new__(ModMaxExtractor)
        result = ex.add_dicts_selected_keys({"a": 1}, {"b": 2, "c": 3}, keys=["b"])
        assert result == {"a": 1, "b": 2}
