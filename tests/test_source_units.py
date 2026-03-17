"""Unit tests for pynasonde.vipir.ngi.source: Trace, Dataset, DataSource."""

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pynasonde.vipir.ngi.source import DataSource, Dataset, Trace


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

class TestTrace:
    def test_default_fields_are_none(self):
        t = Trace()
        assert t.traces is None
        assert t.trace_params is None

    def test_assign_dataframes(self):
        t = Trace()
        t.traces = pd.DataFrame({"x": [1, 2]})
        t.trace_params = pd.DataFrame({"a": [3, 4]})
        assert len(t.traces) == 2
        assert len(t.trace_params) == 2


# ---------------------------------------------------------------------------
# Dataset – direct instantiation and set_traces / get_n_traces
# ---------------------------------------------------------------------------

class TestDataset:
    def test_default_instantiation(self):
        ds = Dataset()
        assert ds.URSI == ""
        assert ds.year == 1970
        assert ds.latitude == pytest.approx(0.0)

    def test_set_traces(self):
        ds = Dataset()
        traces = {
            0: pd.DataFrame({"frequency": [5.0, 6.0], "height": [100.0, 200.0]}),
        }
        trace_params = {
            0: {"hs": 100.0, "fs": 6.0, "popt": [0.0, 0.0, 0.0]},
        }
        ds.set_traces(traces, trace_params)
        assert hasattr(ds, "trace")
        assert len(ds.trace.traces) == 2

    def test_get_n_traces_before_set(self):
        ds = Dataset()
        assert ds.get_n_traces() == 0

    def test_get_n_traces_after_set(self):
        ds = Dataset()
        traces = {0: pd.DataFrame({"frequency": [5.0], "height": [100.0]})}
        trace_params = {
            0: {"hs": 100.0, "fs": 5.0, "popt": [0.0, 0.0, 0.0]},
            1: {"hs": 200.0, "fs": 6.0, "popt": [0.1, 0.0, 0.0]},
        }
        ds.set_traces(traces, trace_params)
        assert ds.get_n_traces() == 2


# ---------------------------------------------------------------------------
# Dataset.__initialize__ – synthetic xarray Dataset
# ---------------------------------------------------------------------------

def _make_xr_dataset():
    """Build a minimal xarray Dataset that matches Dataset.__initialize__().

    Each array uses an explicit, unique dimension name to avoid xarray's
    'conflicting dimension sizes' error when multiple variables share dim_0.
    """
    n_freq = 4
    n_range = 6
    n_chars = 8

    # byte-string arrays for URSI / StationName
    # Use dtype=object (Python bytes elements) so __initialize__ goes through
    # the else branch and u.decode("latin-1") works per production expectation.
    ursi_bytes = np.array([b"K", b"R", b"8", b"3", b"5", b" ", b" ", b" "],
                          dtype=object)
    stn_bytes = np.array([b"T", b"e", b"s", b"t", b" ", b" ", b" ", b" "],
                         dtype=object)

    def scalar(v):
        return xr.DataArray(np.array(v))

    def vec_f(v):
        return xr.DataArray(np.asarray(v, dtype=float), dims=["freq"])

    def vec_r(v):
        return xr.DataArray(np.asarray(v, dtype=float), dims=["range"])

    def mat(v):
        return xr.DataArray(np.asarray(v, dtype=float), dims=["freq", "range"])

    ds = xr.Dataset(
        {
            "URSI": xr.DataArray(ursi_bytes, dims=["chars"]),
            "StationName": xr.DataArray(stn_bytes, dims=["chars"]),
            "year": scalar(2024),
            "daynumber": scalar(100),
            "month": scalar(4),
            "day": scalar(9),
            "hour": scalar(12),
            "minute": scalar(0),
            "second": scalar(0),
            "epoch": scalar(0),
            "latitude": scalar(37.88),
            "longitude": scalar(-75.44),
            "altitude": scalar(0.0),
            "MagLat": scalar(50.0),
            "MagLon": scalar(-70.0),
            "MagDip": scalar(65.0),
            "GyroFreq": scalar(1.3),
            "range_gate_offset": scalar(0.0),
            "gate_count": scalar(n_range),
            "gate_start": scalar(50.0),
            "gate_end": scalar(550.0),
            "gate_step": scalar(100.0),
            "Range0": scalar(50.0),
            "freq_start": scalar(1000.0),
            "freq_end": scalar(10000.0),
            "tune_type": scalar(2),
            "freq_count": scalar(n_freq),
            "linear_step": scalar(1000.0),
            "log_step": scalar(0.01),
            "Range": vec_r(np.linspace(50.0, 550.0, n_range)),
            "Frequency": vec_f(np.linspace(1000.0, 10000.0, n_freq)),
            "Time": vec_f(np.linspace(0, 100, n_freq)),
            "TxDrive": vec_f(np.ones(n_freq) * 10.0),
            "NumAve": vec_f(np.ones(n_freq)),
            "SCT_version": scalar(1.2),
            "SCT": scalar(0),
            "PREFACE": scalar(0),
            "Has_total_power": scalar(0),
            "total_power": mat(np.zeros((n_freq, n_range))),
            "total_noise": vec_f(np.zeros(n_freq)),
            "Has_O-mode_power": scalar(1),
            "O-mode_power": mat(np.random.rand(n_freq, n_range) * 30.0),
            "O-mode_noise": vec_f(np.ones(n_freq) * 3.0),
            "Has_X-mode_power": scalar(0),
            "X-mode_power": mat(np.zeros((n_freq, n_range))),
            "X-mode_noise": vec_f(np.zeros(n_freq)),
            "Has_Doppler": scalar(0),
            "Has_VLoS": scalar(0),
            "Has_SPGR": scalar(0),
            "Has_Zenith": scalar(0),
            "Has_Azimuth": scalar(0),
            "Has_Coherence": scalar(0),
        }
    )
    return ds


class TestDatasetInitialize:
    def test_initialize_populates_fields(self):
        xds = _make_xr_dataset()
        ds = Dataset().__initialize__(xds)
        assert ds.year == 2024
        assert ds.month == 4
        assert ds.day == 9

    def test_o_mode_power_array(self):
        xds = _make_xr_dataset()
        ds = Dataset().__initialize__(xds)
        assert ds.O_mode_power is not None
        assert ds.O_mode_power.shape == (4, 6)

    def test_time_constructed(self):
        xds = _make_xr_dataset()
        ds = Dataset().__initialize__(xds)
        assert isinstance(ds.time, dt.datetime)
        assert ds.time.year == 2024

    def test_sza_computed(self):
        xds = _make_xr_dataset()
        ds = Dataset().__initialize__(xds)
        # SZA should be a finite number between 0 and 180
        assert np.isfinite(ds.sza)

    def test_ursi_decoded(self):
        xds = _make_xr_dataset()
        ds = Dataset().__initialize__(xds)
        # URSI is joined from byte array; should be a plain string
        assert isinstance(ds.URSI, str)


# ---------------------------------------------------------------------------
# DataSource – __init__ with various combinations
# ---------------------------------------------------------------------------

class TestDataSource:
    def test_init_empty_folder(self, tmp_path):
        ds = DataSource(source_folder=str(tmp_path), file_ext="*.ngi.bz2")
        assert ds.file_paths == []
        assert ds.needs_decompression is True

    def test_init_no_bz2_extension(self, tmp_path):
        ds = DataSource(source_folder=str(tmp_path), file_ext="*.nc")
        assert ds.needs_decompression is False

    def test_init_explicit_file_names(self, tmp_path):
        f = tmp_path / "test.nc"
        f.write_bytes(b"")
        ds = DataSource(
            source_folder=str(tmp_path),
            file_ext="*.nc",
            file_names=["test.nc"],
        )
        assert len(ds.file_paths) == 1
        assert ds.file_paths[0] == os.path.join(str(tmp_path), "test.nc")

    def test_init_needs_decompression_flag(self, tmp_path):
        ds = DataSource(
            source_folder=str(tmp_path),
            file_ext="*.nc",
            needs_decompression=True,
        )
        assert ds.needs_decompression is True

    def test_init_glob_finds_files(self, tmp_path):
        # Create two dummy .nc files
        for name in ("a.nc", "b.nc"):
            (tmp_path / name).write_bytes(b"")
        ds = DataSource(source_folder=str(tmp_path), file_ext="*.nc")
        assert len(ds.file_paths) == 2
