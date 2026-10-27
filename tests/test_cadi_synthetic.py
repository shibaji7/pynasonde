"""CADI MD2/MD4 reader, extractor and echo products on a synthetic binary file.

``tests/test_cadi_reader.py`` exercises the real sample file when it is present;
that file is not shipped, so these tests build a small MD4 byte stream with the
documented layout instead. Every field written here is read back and checked.
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynasonde.digisonde.cadi import CadiArray, CadiExtractor, CadiReader
from pynasonde.digisonde.cadi.echo import compute_aoa, map_file_iq_to_physical_rx

FREQS = (2.0e6, 3.0e6, 4.0e6)
HEADER_DT = "May 13 12:00:00 2026"


def build_md4(records, nrx=3, freqs=FREQS, ascii_dt=HEADER_DT, truncate=0) -> bytes:
    """Serialise ``records`` in the MD4 layout.

    ``records`` is a list of ``(minute, second, gain0, per_freq)`` where
    ``per_freq`` has one ``(noise_flag, noise_power, heights, end_flag)`` per
    frequency and ``heights`` is ``[(hflag, high_bit, [(doppler_flag, [(i, q)])])]``.
    """
    out = bytearray()
    out += b"SHA" + ascii_dt.ljust(22).encode()[:22] + b"I"
    out += struct.pack("<HBHHBBHHBH", len(freqs), 8, 60, 600, 2, 4, 150, 200, 1, 12)
    out += b"AS" + struct.pack("<B", nrx) + b"\x00" * 11
    out += struct.pack("<" + "f" * len(freqs), *freqs)
    for r, (minute, second, gain0, per_freq) in enumerate(records):
        out += bytes([minute, second, gain0])
        for f, (noise_flag, noise_pow, heights, end_flag) in enumerate(per_freq):
            out += bytes([noise_flag]) + struct.pack("<H", noise_pow)
            for hflag, high, dops in heights:
                out += bytes([hflag, len(dops) + (128 if high else 0)])
                for dop_flag, iq in dops:
                    out += bytes([dop_flag])
                    for i, q in iq:
                        out += bytes([i, q])
            last = r == len(records) - 1 and f == len(per_freq) - 1
            out += bytes([255 if last else end_flag])
    if not records:
        out += bytes([255])
    return bytes(out[: len(out) - truncate] if truncate else out)


# three-receiver echoes: (i, q) bytes; values above 127 are negative (two's complement)
ECHO_A = [(100, 10), (90, 30), (80, 200)]  # rx3 q = -56
ECHO_B = [(0, 0), (0, 0), (0, 0)]  # silent: maps to 0 dB
ECHO_C = [(50, 100), (255, 20), (30, 30)]  # rx2 i = -1
ECHO_D = [(20, 5), (10, 15), (200, 100)]

RECORDS = [
    (
        0,
        5,
        224,
        [
            (1, 300, [(10, 0, [(3, ECHO_A), (4, ECHO_B)])], 225),
            (0, 310, [], 226),  # a frequency with no detections
            (2, 320, [(20, 1, [(5, ECHO_C)]), (21, 0, [(6, ECHO_D)])], 230),
        ],
    ),
    (
        1,
        30,
        231,
        [
            (0, 100, [(11, 0, [(7, ECHO_A)])], 227),
            (0, 101, [], 228),
            (0, 102, [], 255),
        ],
    ),
]


@pytest.fixture
def md4(tmp_path) -> Path:
    path = tmp_path / "SYN.md4"
    path.write_bytes(build_md4(RECORDS))
    return path


# ── reader ────────────────────────────────────────────────────────────────────


def test_reader_decodes_header_and_frequencies(md4):
    ds = CadiReader(str(md4), dheight_km=3.0).parse()
    h = ds.header
    assert (h.site, h.filetype, h.nfreqs, h.ndops) == ("SHA", "I", 3, 8)
    assert (h.minheight, h.maxheight, h.pps, h.npulses_avgd) == (60, 600, 2, 4)
    assert (h.base_thr100, h.noise_thr100, h.min_dop_forsave, h.dtime) == (
        150,
        200,
        1,
        12,
    )
    assert (h.gain_control, h.sig_process, h.noofreceivers) == ("A", "S", 3)
    assert h.header_datetime == dt.datetime(2026, 5, 13, 12, 0, 0)
    assert ds.frequencies_hz == FREQS and ds.dheight_km == 3.0


def test_reader_decodes_every_detection(md4):
    d = CadiReader(str(md4)).parse().detections
    assert len(d) == 5
    first = d[0]
    assert (first.time_index, first.time_min, first.time_sec) == (0, 0, 5)
    assert (first.frequency_index, first.noise_flag, first.noise_power10) == (0, 1, 300)
    assert (first.gain_flag, first.height_flag, first.doppler_flag) == (224, 10, 3)
    assert first.iq_samples == tuple(ECHO_A)
    assert first.record_datetime == dt.datetime(2026, 5, 13, 12, 0, 5)
    # gain flag of a frequency is the terminator of the previous one
    assert d[2].frequency_index == 2 and d[2].gain_flag == 226
    # high bit in the Doppler count adds 200 to the height flag
    assert (d[2].height_flag, d[3].height_flag) == (220, 21)
    # second time record
    assert (d[4].time_index, d[4].record_datetime.minute) == (1, 1)
    assert d[4].record_datetime.second == 30 and d[4].gain_flag == 231


def test_reader_without_a_parseable_header_date(tmp_path):
    p = tmp_path / "x.md4"
    p.write_bytes(build_md4(RECORDS, ascii_dt="garbage"))
    ds = CadiReader(str(p)).parse()
    assert ds.header.header_datetime is None
    assert all(d.record_datetime is None for d in ds.detections)


def test_header_date_parser_collapses_whitespace():
    parse = CadiReader._parse_header_datetime
    assert parse("May  13 12:00:00 2026  ") == dt.datetime(2026, 5, 13, 12)
    assert parse("not a date") is None


def test_reader_with_no_detections(tmp_path):
    p = tmp_path / "empty.md4"
    p.write_bytes(build_md4([]))
    assert CadiReader(str(p)).parse().detections == []
    assert CadiExtractor(str(p)).to_dataframe_raw().empty
    assert CadiExtractor(str(p)).to_dataframe_products().empty


@pytest.mark.parametrize("cut", [1, 4, 40])
def test_reader_raises_on_truncated_data(tmp_path, cut):
    p = tmp_path / "cut.md4"
    p.write_bytes(build_md4(RECORDS, truncate=cut))
    with pytest.raises((EOFError, struct.error)):
        CadiReader(str(p)).parse()


def test_reader_raises_on_truncated_header(tmp_path):
    p = tmp_path / "short.md4"
    p.write_bytes(build_md4(RECORDS)[:30])
    with pytest.raises(EOFError):
        CadiReader(str(p)).parse()


# ── extractor ─────────────────────────────────────────────────────────────────


def test_raw_dataframe_values(md4):
    df = CadiExtractor(str(md4), dheight_km=2.0).to_dataframe_raw()
    assert len(df) == 5 and set(df["site"]) == {"SHA"}
    r = df.iloc[0]
    assert (r["frequency_mhz"], r["height_km"]) == (2.0, 20.0)
    assert (r["rx3_i_raw"], r["rx3_q_raw"], r["rx3_q"]) == (80, 200, -56)
    assert df.iloc[2]["rx2_i"] == -1  # 255 -> -1
    assert df["source_file"].iloc[0] == str(md4)


def test_signed_conversion():
    assert CadiExtractor._as_signed(127) == 127
    assert CadiExtractor._as_signed(128) == -128
    assert CadiExtractor._as_signed(255) == -1


def test_baseline_pairs():
    assert CadiExtractor._baseline_pairs([1, 2, 3]) == [(1, 2), (1, 3), (2, 3)]
    assert CadiExtractor._baseline_pairs([1]) == []


def test_products_amplitude_phase_and_baselines(md4):
    df = CadiExtractor(str(md4)).to_dataframe_products()
    r = df.iloc[0]  # ECHO_A
    assert r["rx1_amp"] == pytest.approx(np.hypot(100, 10))
    assert r["rx3_phase_rad"] == pytest.approx(np.arctan2(-56, 80))
    assert r["rx3_phase_deg"] == pytest.approx(np.degrees(np.arctan2(-56, 80)))
    assert r["mean_power_db"] == pytest.approx(
        20 * np.log10(np.mean([np.hypot(100, 10), np.hypot(90, 30), np.hypot(80, 56)]))
    )
    assert df.loc[1, "mean_power_db"] == 0.0  # the silent echo
    expected = np.angle(np.exp(1j * (r["rx1_phase_rad"] - r["rx2_phase_rad"])))
    assert r["dphi_12_rad"] == pytest.approx(expected)
    for a, b in ((1, 2), (1, 3), (2, 3)):
        assert f"dphi_{a}{b}_deg" in df and f"coh_{a}{b}" in df
    assert df["coh_12"].between(0, 1).all() or df["coh_12"].isna().any()
    assert (df["doppler_bin"] == df["doppler_flag"]).all()
    assert df["height_correction_km"].isna().all()


def test_products_height_correction_and_custom_grouping(md4):
    c = 2.99792458e5
    df = CadiExtractor(str(md4)).to_dataframe_products(
        coherence_groupby=["time_index"],
        transmitting_delay_us=10.0,
        sampling_delay_us=50.0,
    )
    corr = 0.5 * c * 40e-6
    assert df["height_correction_km"].iloc[0] == pytest.approx(corr)
    np.testing.assert_allclose(df["height_km"], df["height_uncorrected_km"] + corr)
    assert (df["transmitting_delay_us"] == 10.0).all()


def test_group_coherence_of_identical_and_silent_signals():
    z = pd.Series([1 + 1j, 2 - 1j, 0.5j])
    g = pd.Series(["a", "a", "b"])
    out = CadiExtractor._group_coherence(z, z, g)
    assert out.tolist() == pytest.approx([1.0, 1.0, 1.0])
    silent = CadiExtractor._group_coherence(z * 0, z, g)
    assert silent.isna().all()


ARRAY_KW = dict(lat=40.0, lon=-105.0, diagonal_m=100.0)


def test_echo_dataframe_matches_direct_computation(md4):
    df = CadiExtractor(str(md4)).to_dataframe_echoes(**ARRAY_KW)
    assert len(df) == 5 and (df["rx_count"] == 3).all()
    assert df["mode"].isin(["O", "X", "ambiguous", "unknown"]).all()
    assert np.isnan(df["doppler_hz"]).all() and np.isnan(df["v_los"]).all()
    # recompute the first echo's AOA by hand through the public helpers
    ex = CadiExtractor(str(md4))
    prod = ex.to_dataframe_products()
    array = CadiArray.from_receiver_count(n_receivers=3, **ARRAY_KW)
    iq = map_file_iq_to_physical_rx(prod.iloc[0], array)
    aoa = compute_aoa(iq, array, float(prod.iloc[0]["frequency_hz"]))
    assert df.iloc[0]["zenith"] == pytest.approx(aoa["zenith"])
    assert df.iloc[0]["azimuth"] == pytest.approx(aoa["azimuth"])
    assert 0 <= df["zenith"].min() and df["zenith"].max() <= 90


def test_echo_dataframe_with_doppler_calibration(md4):
    df = CadiExtractor(str(md4)).to_dataframe_echoes(
        fft_size=16, pulse_rate_hz=64.0, **ARRAY_KW
    )
    first = df.iloc[0]  # doppler bin 3, f = (3 - 8) * 64/16 = -20 Hz
    assert first["doppler_hz"] == pytest.approx(-20.0)
    wavelength = 2.99792458e8 / 2.0e6
    assert first["v_los"] == pytest.approx(-20.0 * wavelength / 2.0)


def test_interferometry_dataframe(md4):
    df = CadiExtractor(str(md4)).to_dataframe_interferometry(**ARRAY_KW)
    assert len(df) == 5 * 3  # three baselines per detection
    assert df["projected_direction_cosine"].between(-1, 1).all()
    assert df["baseline_azimuth"].between(0, 360).all()
    assert set(df["rx_a"]).issubset({1, 2, 3})


def test_two_receiver_file(tmp_path):
    two = [(0, 1, 224, [(0, 1, [(5, 0, [(1, [(10, 20), (30, 40)])])], 255)])]
    p = tmp_path / "two.md4"
    p.write_bytes(build_md4(two, nrx=2, freqs=(3.0e6,)))
    ex = CadiExtractor(str(p))
    assert len(ex.to_dataframe_interferometry(**ARRAY_KW)) == 1
    with pytest.raises(ValueError):
        ex.to_dataframe_echoes(**ARRAY_KW)


# ── extract_CADI / load_CADI_files ────────────────────────────────────────────


@pytest.mark.parametrize("product", ["raw", "products", "interferometry", "echoes"])
def test_extract_cadi_products(md4, product):
    df = CadiExtractor.extract_CADI(str(md4), product=product, **ARRAY_KW)
    assert not df.empty


def test_extract_cadi_validation(md4):
    for product in ("interferometry", "echoes"):
        with pytest.raises(ValueError, match="requires lat"):
            CadiExtractor.extract_CADI(str(md4), product=product)
    with pytest.raises(ValueError, match="product must be"):
        CadiExtractor.extract_CADI(str(md4), product="nope")


def test_load_cadi_files_reads_a_folder(tmp_path):
    folder = tmp_path / "day"
    folder.mkdir()
    (folder / "A.md4").write_bytes(build_md4(RECORDS))
    (folder / "B.MD4").write_bytes(build_md4(RECORDS))
    (folder / "ignored.txt").write_text("not cadi")
    df = CadiExtractor.load_CADI_files([str(folder)], product="raw")
    assert len(df) == 10
    prod = CadiExtractor.load_CADI_files([str(folder)], product="products")
    assert "mean_power_db" in prod


# ── echo helpers not reached through the file path ────────────────────────────


def test_velocity_fit_recovers_a_known_drift():
    from pynasonde.digisonde.cadi.echo import compute_velocity_from_skymap

    true = np.array([30.0, -40.0, 5.0])
    dirs = [(0.3, 0.1), (-0.2, 0.4), (0.1, -0.5), (-0.4, -0.2), (0.5, 0.3)]
    pts = []
    for xl, yl in dirs:
        zl = float(np.sqrt(1 - xl**2 - yl**2))
        pts.append(dict(XL=xl, YL=yl, ZL=zl, v_los=float(np.dot(true, [xl, yl, zl]))))
    fit = compute_velocity_from_skymap(pts)
    assert (fit["Vx"], fit["Vy"], fit["Vz"]) == pytest.approx(tuple(true))
    assert fit["Vh"] == pytest.approx(50.0) and fit["n_points"] == 5
    assert fit["azimuth"] == pytest.approx(np.degrees(np.arctan2(30, -40)))
    assert fit["residual"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="at least 3"):
        compute_velocity_from_skymap(pts[:2])


def test_doppler_and_velocity_helpers():
    from pynasonde.digisonde.cadi.echo import doppler_frequency, los_velocity

    assert np.isnan(doppler_frequency(3, None, 64.0))
    assert doppler_frequency(8, 16, 64.0) == 0.0
    assert np.isnan(los_velocity(np.nan, 3e6)) and np.isnan(los_velocity(1.0, 0.0))
    assert los_velocity(10.0, 3.0e6) == pytest.approx(10 * 99.93082 / 2, rel=1e-4)


def test_ox_mode_from_perpendicular_dipoles():
    from pynasonde.digisonde.cadi.echo import classify_ox_mode

    array = CadiArray.from_receiver_count(n_receivers=4, **ARRAY_KW)
    a, b = array.get_perpendicular_pairs()[0]
    ew, ns = (a, b) if abs(array.dipole_orientations[a] - 90) < 45 else (b, a)
    base = {r: 1 + 0j for r in array.active_rx}
    mode_lh, pp = classify_ox_mode({**base, ew: 1 + 0j, ns: 1j}, array)
    mode_rh, pm = classify_ox_mode({**base, ew: 1 + 0j, ns: -1j}, array)
    assert {mode_lh, mode_rh} == {"O", "X"} and pp == pytest.approx(-pm)
    assert classify_ox_mode({**base, ew: 1 + 0j, ns: 1 + 0j}, array)[0] == "ambiguous"
    mode, pp = classify_ox_mode({}, array)
    assert mode == "unknown" and np.isnan(pp)
