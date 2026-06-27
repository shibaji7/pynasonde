"""Tests for phase-1 CADI MD2/MD4 support."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pynasonde.digisonde import CadiIonogram, DigiPlots
from pynasonde.digisonde.cadi import (
    CadiArray,
    CadiExtractor,
    CadiReader,
    CadiReceiverLayout,
    compute_height_correction_km,
    debug_echo_geometry,
    map_file_iq_to_physical_rx,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MD4 = PROJECT_ROOT / "tmp" / "CADI" / "6E131200.md4"


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_cadi_reader_parses_header_and_records():
    reader = CadiReader(str(SAMPLE_MD4))
    dataset = reader.parse()

    assert dataset.header.site.strip() == "SHA"
    assert dataset.header.filetype == "I"
    assert dataset.header.nfreqs == 310
    assert dataset.header.noofreceivers == 3
    assert len(dataset.frequencies_hz) == dataset.header.nfreqs
    assert len(dataset.detections) > 0


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_cadi_extractor_dataframe_columns_and_counts():
    extractor = CadiExtractor(str(SAMPLE_MD4))
    df = extractor.to_dataframe_raw()

    assert not df.empty
    assert len(df) == len(extractor.dataset.detections)
    assert set(
        [
            "site",
            "source_file",
            "record_datetime",
            "frequency_mhz",
            "height_km",
            "doppler_flag",
            "rx1_i",
            "rx1_q",
            "rx2_i",
            "rx2_q",
            "rx3_i",
            "rx3_q",
        ]
    ).issubset(df.columns)


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_cadi_products_columns_present():
    extractor = CadiExtractor(str(SAMPLE_MD4))
    df = extractor.to_dataframe_products()

    assert not df.empty
    assert set(
        [
            "doppler_bin",
            "mean_power_db",
            "rx1_amp",
            "rx1_phase_rad",
            "rx1_phase_deg",
            "rx2_amp",
            "rx2_phase_rad",
            "rx2_phase_deg",
            "rx3_amp",
            "rx3_phase_rad",
            "rx3_phase_deg",
            "dphi_12_rad",
            "dphi_12_deg",
            "dphi_13_rad",
            "dphi_13_deg",
            "dphi_23_rad",
            "dphi_23_deg",
            "coh_12",
            "coh_13",
            "coh_23",
        ]
    ).issubset(df.columns)
    assert df["mean_power_db"].notna().any()
    coh12 = df["coh_12"].dropna()
    assert not coh12.empty
    assert (coh12 >= -1e-9).all()
    assert (coh12 <= 1.0 + 1e-9).all()


def test_cadi_height_correction_from_timing_delays():
    correction = compute_height_correction_km(
        transmitting_delay_us=2056,
        sampling_delay_us=2063,
    )

    assert correction == pytest.approx(1.049273603)


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_cadi_products_apply_optional_height_correction():
    extractor = CadiExtractor(str(SAMPLE_MD4))
    raw = extractor.to_dataframe_raw()
    products = extractor.to_dataframe_products(
        transmitting_delay_us=2056,
        sampling_delay_us=2063,
    )

    correction = compute_height_correction_km(2056, 2063)

    assert "height_uncorrected_km" in products
    assert "height_correction_km" in products
    assert products["height_correction_km"].iloc[0] == pytest.approx(correction)
    assert products["height_uncorrected_km"].iloc[0] == raw["height_km"].iloc[0]
    assert products["height_km"].iloc[0] == pytest.approx(
        raw["height_km"].iloc[0] + correction
    )


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_extract_cadi_products_mode():
    df = CadiExtractor.extract_CADI(str(SAMPLE_MD4), product="products")
    assert not df.empty
    assert "mean_power_db" in df.columns
    assert "doppler_bin" in df.columns
    assert "dphi_12_deg" in df.columns
    assert "coh_12" in df.columns


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_load_cadi_files_from_folder():
    folder = str(SAMPLE_MD4.parent)
    df = CadiExtractor.load_CADI_files(
        folders=[folder],
        exts="*.md4",
        n_procs=1,
    )
    assert not df.empty
    assert (df["site"] == "SHA").any()


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_load_cadi_files_products_mode():
    folder = str(SAMPLE_MD4.parent)
    df = CadiExtractor.load_CADI_files(
        folders=[folder],
        exts="*.md4",
        n_procs=1,
        product="products",
    )
    assert not df.empty
    assert "mean_power_db" in df.columns


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_cadi_ionogram_methods_return_artists():
    pytest.importorskip("matplotlib")

    df = CadiExtractor.extract_CADI(str(SAMPLE_MD4), product="products")
    plotter = CadiIonogram(figsize=(5, 3), nrows=1, ncols=2)

    ax1, im1 = plotter.add_power_ionogram(df, text="power")
    ax2, im2 = plotter.add_doppler_ionogram(df, text="doppler")

    assert ax1 is not None
    assert ax2 is not None
    assert im1 is not None
    assert im2 is not None
    plotter.close()


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_generic_frequency_height_scatter_supports_cadi_products():
    pytest.importorskip("matplotlib")

    products = CadiExtractor.extract_CADI(str(SAMPLE_MD4), product="products")
    echoes = CadiExtractor.extract_CADI(
        str(SAMPLE_MD4),
        product="echoes",
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=7,
    )
    plotter = DigiPlots(figsize=(5, 3), nrows=1, ncols=2)

    ax1, im1 = plotter.add_frequency_height_scatter(
        products,
        zparam="mean_power_db",
        prange=[0, 35],
        cbar_label="Mean Power, dB",
    )
    ax2, handles = plotter.add_categorical_frequency_height_scatter(
        echoes,
        category_param="mode",
        colors={"O": "#0C5DA5", "X": "#FF9500", "ambiguous": "#9E9E9E"},
        category_order=["O", "X", "ambiguous"],
    )

    assert ax1 is not None
    assert ax2 is not None
    assert im1 is not None
    assert handles
    plotter.close()


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_cadi_echoes_require_geometry_and_return_columns():
    df = CadiExtractor.extract_CADI(
        str(SAMPLE_MD4),
        product="echoes",
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=7,
    )

    assert not df.empty
    assert set(
        [
            "frequency_hz",
            "height_km",
            "mean_power_db",
            "doppler_bin",
            "doppler_hz",
            "v_los",
            "XL",
            "YL",
            "ZL",
            "zenith",
            "azimuth",
            "aoa_method",
            "polarization_deg",
            "mode",
            "rx_count",
        ]
    ).issubset(df.columns)
    assert set(df["mode"].dropna().unique()).issubset(
        {"O", "X", "ambiguous", "unknown"}
    )
    assert df["doppler_hz"].isna().all()


@pytest.mark.skipif(not SAMPLE_MD4.exists(), reason="CADI sample file not available")
def test_cadi_echoes_optional_doppler_calibration():
    df = CadiExtractor.extract_CADI(
        str(SAMPLE_MD4),
        product="echoes",
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=7,
        fft_size=8,
        pulse_rate_hz=20.0,
    )
    assert np.isfinite(df["doppler_hz"]).any()
    assert np.isfinite(df["v_los"]).any()


def test_cadi_echoes_reject_less_than_three_receivers():
    extractor = CadiExtractor("dummy.md4")
    extractor.dataset = SimpleNamespace(header=SimpleNamespace(noofreceivers=2))

    with pytest.raises(ValueError, match="at least 3"):
        extractor.to_dataframe_echoes(
            lat=17.47,
            lon=78.57,
            diagonal_m=30.0,
            rx_bitmask=3,
        )


def test_cadi_interferometry_accepts_two_receivers(monkeypatch):
    extractor = CadiExtractor("dummy.md4")
    extractor.dataset = SimpleNamespace(header=SimpleNamespace(noofreceivers=2))
    df_products = pd.DataFrame(
        {
            "frequency_hz": [3_000_000.0],
            "frequency_mhz": [3.0],
            "height_km": [120.0],
            "mean_power_db": [10.0],
            "doppler_bin": [4],
            "rx1_i": [10.0],
            "rx1_q": [0.0],
            "rx2_i": [0.0],
            "rx2_q": [10.0],
            "coh_12": [1.0],
            "source_file": ["dummy.md4"],
            "record_datetime": [None],
        }
    )
    monkeypatch.setattr(extractor, "to_dataframe_products", lambda **kwargs: df_products)

    df = extractor.to_dataframe_interferometry(
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=3,
    )

    assert len(df) == 1
    assert df.loc[0, "rx_count"] == 2
    assert df.loc[0, "rx_a"] == 1
    assert df.loc[0, "rx_b"] == 2
    assert np.isfinite(df.loc[0, "projected_direction_cosine"])
    assert np.isfinite(df.loc[0, "projected_angle_deg"])


def test_cadi_file_order_columns_map_to_physical_rx_bitmask():
    array = CadiArray.from_receiver_count(
        n_receivers=3,
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=14,
    )
    row = pd.Series(
        {
            "rx1_i": 1.0,
            "rx1_q": 10.0,
            "rx2_i": 2.0,
            "rx2_q": 20.0,
            "rx3_i": 3.0,
            "rx3_q": 30.0,
        }
    )

    iq = map_file_iq_to_physical_rx(row, array)

    assert set(iq) == {2, 3, 4}
    assert iq[2] == complex(1.0, 10.0)
    assert iq[3] == complex(2.0, 20.0)
    assert iq[4] == complex(3.0, 30.0)


def test_cadi_array_standard_northern_hemisphere_layout():
    array = CadiArray.from_receiver_count(
        n_receivers=4,
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
    )

    assert array.rx_positions[1] == (0.0, 15.0)
    assert array.rx_positions[2] == (15.0, 0.0)
    assert array.rx_positions[3] == (0.0, -15.0)
    assert array.rx_positions[4] == (-15.0, 0.0)
    assert array.dipole_orientations == {
        1: 0.0,
        2: 90.0,
        3: 0.0,
        4: 90.0,
    }


def test_cadi_array_custom_receiver_layout():
    unit_vectors = {
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (-1.0, 0.0),
        4: (0.0, -1.0),
    }
    dipoles = {1: 0.0, 2: 90.0, 3: 0.0, 4: 90.0}

    array = CadiArray.from_receiver_count(
        n_receivers=3,
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=14,
        receiver_layout=CadiReceiverLayout.CUSTOM,
        unit_vectors=unit_vectors,
        dipole_orient_deg=dipoles,
    )

    assert array.active_rx == [2, 3, 4]
    assert array.rx_positions[2] == (0.0, 15.0)
    assert array.rx_positions[3] == (-15.0, 0.0)
    assert array.dipole_orientations[2] == 90.0


def test_cadi_array_custom_receiver_layout_requires_complete_maps():
    with pytest.raises(ValueError, match="requires unit_vectors"):
        CadiArray.from_receiver_count(
            n_receivers=3,
            lat=17.47,
            lon=78.57,
            diagonal_m=30.0,
            rx_bitmask=14,
            receiver_layout="custom",
        )


def test_cadi_array_describe_reports_receiver_geometry():
    array = CadiArray.from_receiver_count(
        n_receivers=3,
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=14,
    )

    desc = array.describe()

    assert desc["rx_bitmask"] == 14
    assert desc["active_rx"] == [2, 3, 4]
    assert len(desc["receivers"]) == 4
    assert len(desc["baselines"]) == 3
    assert desc["perpendicular_pairs"]


def test_debug_echo_geometry_reports_mapping_phases_and_aoa():
    array = CadiArray.from_receiver_count(
        n_receivers=3,
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=14,
    )
    row = pd.Series(
        {
            "frequency_hz": 3_000_000.0,
            "rx1_i": 10.0,
            "rx1_q": 0.0,
            "rx2_i": 0.0,
            "rx2_q": 10.0,
            "rx3_i": -10.0,
            "rx3_q": 0.0,
        }
    )

    debug = debug_echo_geometry(row, array)

    assert debug["receiver_mapping"][0]["physical_rx"] == 2
    assert debug["receiver_mapping"][1]["physical_rx"] == 3
    assert debug["receiver_mapping"][2]["physical_rx"] == 4
    assert len(debug["baseline_phases"]) == 3
    assert debug["mode"] in {"O", "X", "ambiguous", "unknown"}
    assert "method" in debug["aoa"]


def test_cadi_interferometry_uses_physical_rx_from_bitmask(monkeypatch):
    extractor = CadiExtractor("dummy.md4")
    extractor.dataset = SimpleNamespace(header=SimpleNamespace(noofreceivers=3))
    df_products = pd.DataFrame(
        {
            "frequency_hz": [3_000_000.0],
            "frequency_mhz": [3.0],
            "height_km": [120.0],
            "mean_power_db": [10.0],
            "doppler_bin": [4],
            "rx1_i": [10.0],
            "rx1_q": [0.0],
            "rx2_i": [0.0],
            "rx2_q": [10.0],
            "rx3_i": [-10.0],
            "rx3_q": [0.0],
            "coh_12": [1.0],
            "coh_13": [1.0],
            "coh_23": [1.0],
            "source_file": ["dummy.md4"],
            "record_datetime": [None],
        }
    )
    monkeypatch.setattr(extractor, "to_dataframe_products", lambda **kwargs: df_products)

    df = extractor.to_dataframe_interferometry(
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=14,
    )

    assert set(df["rx_a"]).union(set(df["rx_b"])) == {2, 3, 4}


def test_cadi_interferometry_rejects_one_receiver():
    extractor = CadiExtractor("dummy.md4")
    extractor.dataset = SimpleNamespace(header=SimpleNamespace(noofreceivers=1))

    with pytest.raises(ValueError, match="at least 2"):
        extractor.to_dataframe_interferometry(
            lat=17.47,
            lon=78.57,
            diagonal_m=30.0,
            rx_bitmask=1,
        )


def test_cadi_array_bitmask_count_mismatch_raises():
    array = CadiArray.from_receiver_count(
        n_receivers=3,
        lat=17.47,
        lon=78.57,
        diagonal_m=30.0,
        rx_bitmask=15,
    )
    with pytest.raises(ValueError, match="file header reports 3"):
        array.validate_rx_count(3)
