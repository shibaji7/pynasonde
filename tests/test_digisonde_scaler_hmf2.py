"""Tests for the hmF2 / M(3000)F2 helpers of the Digisonde scaler.

The formulas are checked against hand calculations, and the Abel inversion against
profiles whose true height is known (a flat virtual-height trace has hmF2 equal to
that height).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pynasonde.digisonde.scaler.hmF2 import ScaleHmF2

# ── M(3000)F2 ─────────────────────────────────────────────────────────────────


def test_m3000f2_matches_the_secant_law_by_hand():
    # one point at f = foF2, h' = 300 km: sqrt(1 + (1500/300)^2) = sqrt(26)
    m = ScaleHmF2.calculate_m3000f2([5.0], [300.0], fof2=5.0)
    assert m == pytest.approx(np.sqrt(26.0))


def test_m3000f2_takes_the_maximum_over_the_trace_and_skips_bad_heights():
    freqs = [4.0, 5.0, 5.5]
    heights = [250.0, 0.0, 350.0]  # h' <= 0 is ignored
    expected = max(
        np.sqrt(1 + (1500 / 250) ** 2) * 4.0 / 5.5,
        np.sqrt(1 + (1500 / 350) ** 2) * 5.5 / 5.5,
    )
    assert ScaleHmF2.calculate_m3000f2(freqs, heights, fof2=5.5) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "freqs, heights, fof2, message",
    [
        ([], [], 5.0, "match in size"),
        ([1.0, 2.0], [300.0], 5.0, "match in size"),
        ([1.0], [300.0], 0.0, "positive"),
        ([1.0, 2.0], [0.0, -5.0], 5.0, "No valid height"),
    ],
)
def test_m3000f2_rejects_bad_input(freqs, heights, fof2, message):
    with pytest.raises(ValueError, match=message):
        ScaleHmF2.calculate_m3000f2(freqs, heights, fof2)


# ── empirical hmF2 formulas ───────────────────────────────────────────────────


def test_shimazaki_by_hand_and_validation():
    assert ScaleHmF2.shimazaki(3.0) == pytest.approx(1490.0 / 3.0 - 176.0)
    with pytest.raises(ValueError, match="greater than 0"):
        ScaleHmF2.shimazaki(0.0)


def test_bilitza_by_hand():
    m, fof2, foe = 3.0, 8.0, 2.0  # r = 4
    delta = 0.18 / (4.0 - 1.4) + 0.096 * (4.0 - 1.4)
    assert delta < 0.4  # inside the clamp, so the raw value is used
    assert ScaleHmF2.bilitza(m, fof2, foe) == pytest.approx(
        1490.0 / (m + delta) - 176.0
    )


def test_bilitza_clamps_the_singularity_and_the_correction():
    # r just above 1.4 would give a huge delta_m; it is clamped to 0.4
    near_singular = ScaleHmF2.bilitza(3.0, 1.45, 1.0)
    assert near_singular == pytest.approx(1490.0 / (3.0 + 0.4) - 176.0)
    # r below 1.41 is lifted to 1.41, so the result equals the r = 1.41 result
    assert ScaleHmF2.bilitza(3.0, 1.0, 1.0) == pytest.approx(
        ScaleHmF2.bilitza(3.0, 1.41, 1.0)
    )


@pytest.mark.parametrize("args", [(0, 5, 2), (3, 0, 2), (3, 5, 0), (3, 5, -1)])
def test_bilitza_rejects_non_positive_inputs(args):
    with pytest.raises(ValueError, match="positive"):
        ScaleHmF2.bilitza(*args)


def test_constant_ratio_fallback_uses_ratio_3_3():
    fof2 = 9.9
    assert ScaleHmF2.constant_ratio_fallback(3.0, fof2) == pytest.approx(
        ScaleHmF2.bilitza(3.0, fof2, fof2 / 3.3)
    )


def test_zenith_fallback_night_is_shimazaki_and_day_is_bilitza():
    assert ScaleHmF2.zenith_estimate_fallback(3.0, 8.0, 100.0, 50.0) == pytest.approx(
        ScaleHmF2.shimazaki(3.0)
    )
    sza, r12 = 30.0, 80.0
    foe = 3.31 * (1 + 0.008 * r12) ** 0.25 * np.cos(np.radians(sza)) ** 0.25
    assert ScaleHmF2.zenith_estimate_fallback(3.0, 8.0, sza, r12) == pytest.approx(
        ScaleHmF2.bilitza(3.0, 8.0, foe)
    )


# ── polynomial smoothing of the F trace ───────────────────────────────────────


def test_smooth_f_trace_rejects_an_outlier():
    f = np.linspace(2.0, 7.0, 16)
    hp = 250.0 + 5.0 * f
    hp[8] += 300.0  # one wild Viterbi point
    f_clean, hp_clean, coeffs = ScaleHmF2._smooth_f_trace(f, hp, foF2=8.0)
    assert len(f_clean) == 15 and 8 not in np.searchsorted(f, f_clean)
    # the refit follows the underlying line, not the outlier
    assert np.polyval(coeffs, 4.5) == pytest.approx(250.0 + 5.0 * 4.5, abs=1.0)


def test_smooth_f_trace_with_few_points():
    f, hp = np.array([3.0, 4.0]), np.array([300.0, 310.0])
    f_out, hp_out, c = ScaleHmF2._smooth_f_trace(f, hp, 8.0, poly_deg=3)
    assert len(f_out) == 2
    assert np.polyval(c, 3.5) == pytest.approx(305.0)  # straight line through both
    _, _, c1 = ScaleHmF2._smooth_f_trace(np.array([3.0]), np.array([300.0]), 8.0)
    assert np.polyval(c1, 1.0) == pytest.approx(300.0)
    _, _, c0 = ScaleHmF2._smooth_f_trace(np.array([]), np.array([]), 8.0)
    assert np.polyval(c0, 1.0) == pytest.approx(200.0)


def test_smooth_f_trace_keeps_all_points_if_too_many_are_rejected():
    f = np.array([2.0, 3.0, 4.0, 5.0])
    hp = np.array([300.0, 300.0, 300.0, 300.0])
    f_out, _, _ = ScaleHmF2._smooth_f_trace(f, hp, 8.0, poly_deg=3)
    assert len(f_out) == 4


# ── Abel inversion ────────────────────────────────────────────────────────────

needs_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None, reason="abel_inversion needs scipy"
)


def flat_trace(height=300.0, fof2=8.0, n=20):
    f = np.linspace(2.0, 0.97 * fof2, n)
    return f, np.full_like(f, height)


@needs_scipy
def test_abel_flat_night_trace_returns_that_height():
    f, hp = flat_trace(300.0)
    hm = ScaleHmF2.abel_inversion(f, hp, foF2=8.0, sza_deg=120.0)
    assert hm == pytest.approx(300.0, abs=1.0)


@needs_scipy
def test_abel_is_independent_of_quadrature_order():
    f, hp = flat_trace(300.0)
    coarse = ScaleHmF2.abel_inversion(f, hp, 8.0, sza_deg=120.0, n_quad=16)
    fine = ScaleHmF2.abel_inversion(f, hp, 8.0, sza_deg=120.0, n_quad=128)
    assert coarse == pytest.approx(fine, abs=0.5)


@needs_scipy
def test_abel_rising_trace_is_below_the_virtual_height_at_foF2():
    """True height of the peak is lower than the virtual height at foF2."""
    fof2 = 8.0
    f = np.linspace(2.0, 0.97 * fof2, 25)
    hp = 280.0 + 60.0 * (f / fof2) ** 4  # virtual height rises toward foF2
    hm = ScaleHmF2.abel_inversion(f, hp, fof2, sza_deg=120.0)
    assert 250.0 < hm < hp.max()


@needs_scipy
def test_abel_returns_nan_without_enough_data():
    f, hp = flat_trace()
    assert np.isnan(ScaleHmF2.abel_inversion(f, hp, foF2=0.0))
    assert np.isnan(ScaleHmF2.abel_inversion(f[:2], hp[:2], foF2=8.0))


@needs_scipy
def test_abel_with_a_good_e_trace():
    fof2, foe = 8.0, 3.0
    f_f, hp_f = flat_trace(300.0, fof2)
    f_e = np.linspace(1.0, 0.95 * foe, 8)
    hp_e = np.full_like(f_e, 110.0)
    hm = ScaleHmF2.abel_inversion(
        f_f, hp_f, fof2, f_E=f_e, hp_E=hp_e, foE=foe, hpE=110.0, sza_deg=30.0
    )
    # with an E layer below, the integral includes the low E-region heights
    assert 90.0 < hm < 300.0


@needs_scipy
def test_abel_with_sparse_e_uses_the_qp_model():
    f, hp = flat_trace(300.0)
    hm = ScaleHmF2.abel_inversion(
        f,
        hp,
        8.0,
        f_E=np.array([1.0, 2.0]),
        hp_E=np.array([110.0, 115.0]),
        foE=3.0,
        hpE=110.0,
        sza_deg=120.0,
    )
    assert 90.0 < hm < 300.0


@needs_scipy
def test_abel_daytime_without_e_uses_the_chapman_prediction():
    f, hp = flat_trace(300.0)
    day = ScaleHmF2.abel_inversion(f, hp, 8.0, sza_deg=20.0)
    night = ScaleHmF2.abel_inversion(f, hp, 8.0, sza_deg=120.0)
    assert 90.0 < day < night  # a predicted E layer pulls the integral down


@needs_scipy
def test_abel_good_e_trace_with_too_few_valid_points_falls_back():
    f, hp = flat_trace(300.0)
    # five E points claimed, but all above foE so none survive the mask
    f_e = np.linspace(4.0, 5.0, 5)
    hp_e = np.full(5, 110.0)
    hm = ScaleHmF2.abel_inversion(
        f, hp, 8.0, f_E=f_e, hp_E=hp_e, foE=3.0, hpE=110.0, sza_deg=120.0
    )
    # the unusable E trace is ignored: at night the flat F trace is extrapolated
    assert hm == pytest.approx(300.0, abs=1.0)


@needs_scipy
def test_abel_trace_with_one_valid_point_defaults_to_200_km_profile():
    # only one F point lies in (0, foF2): the profile is a constant 200 km
    f = np.array([3.0, 9.0, 10.0])
    hp = np.array([300.0, 300.0, 300.0])
    assert ScaleHmF2.abel_inversion(f, hp, 8.0) == pytest.approx(200.0, abs=1.0)


@needs_scipy
def test_abel_never_returns_below_90_km():
    f, hp = flat_trace(60.0)
    assert ScaleHmF2.abel_inversion(f, hp, 8.0, sza_deg=120.0) >= 90.0


@needs_scipy
def test_abel_survives_a_failure_in_profile_building(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("broken profile")

    monkeypatch.setattr(ScaleHmF2, "_build_hp_profile", staticmethod(boom))
    f, hp = flat_trace()
    assert np.isnan(ScaleHmF2.abel_inversion(f, hp, 8.0))


def test_build_hp_profile_is_monotone_and_on_a_512_point_grid():
    f, hp = flat_trace(300.0)
    f_grid, hp_grid = ScaleHmF2._build_hp_profile(
        f, hp, 8.0, None, None, None, None, sza_deg=120.0
    )
    assert f_grid.shape == hp_grid.shape == (512,)
    assert f_grid[0] == 0.0 and f_grid[-1] == pytest.approx(0.999 * 8.0)
    assert np.isfinite(hp_grid).all() and hp_grid.min() >= 90.0
