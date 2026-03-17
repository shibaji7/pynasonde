"""Tests for pure utility functions in pynasonde.digisonde.raw.raw_parse."""

import datetime as dt

import numpy as np
import pytest

from pynasonde.digisonde.raw.raw_parse import (
    _ensure_datetime,
    _interp_complex,
    _next_power_of_two,
    _next_smooth_235,
    _prev_power_of_two,
)

_UTC = dt.timezone.utc


class TestEnsureDatetime:
    def test_naive_datetime_gets_utc(self):
        d = dt.datetime(2024, 4, 9, 12, 0, 0)
        result = _ensure_datetime(d)
        assert result.tzinfo is not None
        assert result.replace(tzinfo=None) == d

    def test_aware_datetime_returns_utc(self):
        est = dt.timezone(dt.timedelta(hours=-5))
        d = dt.datetime(2024, 4, 9, 12, 0, 0, tzinfo=est)
        result = _ensure_datetime(d)
        assert result.tzinfo == _UTC
        assert result.hour == 17  # 12 EST → 17 UTC

    def test_int_timestamp(self):
        epoch = 0
        result = _ensure_datetime(epoch)
        assert result == dt.datetime(1970, 1, 1, 0, 0, 0, tzinfo=_UTC)

    def test_float_timestamp(self):
        result = _ensure_datetime(86400.0)
        assert result.day == 2
        assert result.hour == 0

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _ensure_datetime("not-a-date")


class TestNextPowerOfTwo:
    def test_exact_power(self):
        assert _next_power_of_two(8) == 8

    def test_not_exact_power(self):
        assert _next_power_of_two(5) == 8

    def test_value_one(self):
        assert _next_power_of_two(1) == 1

    def test_value_less_than_one(self):
        assert _next_power_of_two(0.5) == 1

    def test_large_value(self):
        assert _next_power_of_two(1000) == 1024


class TestPrevPowerOfTwo:
    def test_exact_power(self):
        assert _prev_power_of_two(16) == 16

    def test_not_exact_power(self):
        assert _prev_power_of_two(9) == 8

    def test_value_one(self):
        assert _prev_power_of_two(1) == 1

    def test_value_less_than_one(self):
        assert _prev_power_of_two(0) == 1

    def test_large_value(self):
        assert _prev_power_of_two(1000) == 512


class TestNextSmooth235:
    def test_already_smooth(self):
        # 1=2^0, already >=1
        assert _next_smooth_235(1) == 1

    def test_two_is_smooth(self):
        assert _next_smooth_235(2) == 2

    def test_finds_smooth_above(self):
        # 7 is not 5-smooth; next is 8 (2^3)
        result = _next_smooth_235(7)
        assert result >= 7
        # verify result is 5-smooth: only prime factors 2, 3, 5
        n = result
        for p in (2, 3, 5):
            while n % p == 0:
                n //= p
        assert n == 1

    def test_target_100(self):
        result = _next_smooth_235(100)
        assert result >= 100
        n = result
        for p in (2, 3, 5):
            while n % p == 0:
                n //= p
        assert n == 1

    def test_target_large(self):
        result = _next_smooth_235(500)
        assert result >= 500


class TestInterpComplex:
    def test_real_only(self):
        x_old = np.array([0.0, 1.0, 2.0])
        y_old = np.array([0.0, 1.0, 2.0], dtype=complex)
        x_new = np.array([0.5, 1.5])
        result = _interp_complex(x_old, y_old, x_new)
        np.testing.assert_allclose(result.real, [0.5, 1.5])
        np.testing.assert_allclose(result.imag, [0.0, 0.0])

    def test_complex_values(self):
        x_old = np.array([0.0, 1.0])
        y_old = np.array([0 + 0j, 2 + 4j])
        x_new = np.array([0.5])
        result = _interp_complex(x_old, y_old, x_new)
        assert result[0] == pytest.approx(1.0 + 2.0j)
