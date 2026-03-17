"""Tests for pynasonde.vipir.riq.utils — odd/even/trim_null/unwrap helpers."""

import math

import pytest

from pynasonde.vipir.riq.utils import even, len_trim_null, odd, trim_null, unwrap


class TestOddEven:
    def test_odd_positive(self):
        assert odd(1) is True
        assert odd(3) is True
        assert odd(99) is True

    def test_odd_negative(self):
        assert odd(2) is False
        assert odd(100) is False

    def test_zero_is_even(self):
        assert odd(0) is False
        assert even(0) is True

    def test_even_positive(self):
        assert even(2) is True
        assert even(4) is True

    def test_even_negative(self):
        assert even(1) is False
        assert even(7) is False


class TestTrimNull:
    def test_no_nulls(self):
        assert trim_null("hello") == "hello"

    def test_embedded_nulls(self):
        assert trim_null("he\x00llo\x00") == "hello"

    def test_all_nulls(self):
        assert trim_null("\x00\x00\x00") == ""

    def test_leading_trailing_spaces_stripped(self):
        assert trim_null("  abc  ") == "abc"

    def test_len_trim_null(self):
        assert len_trim_null("abc\x00\x00") == 3

    def test_len_trim_null_empty(self):
        assert len_trim_null("\x00\x00") == 0


class TestUnwrap:
    def test_within_range(self):
        assert unwrap(0.0) == pytest.approx(0.0)
        assert unwrap(math.pi) == pytest.approx(math.pi)
        assert unwrap(-math.pi) == pytest.approx(-math.pi)

    def test_above_pi(self):
        result = unwrap(math.pi + 0.5)
        assert result == pytest.approx(math.pi + 0.5 - 2 * math.pi)

    def test_below_neg_pi(self):
        result = unwrap(-math.pi - 0.5)
        assert result == pytest.approx(-math.pi - 0.5 + 2 * math.pi)

    def test_small_values(self):
        assert unwrap(0.1) == pytest.approx(0.1)
        assert unwrap(-0.1) == pytest.approx(-0.1)
