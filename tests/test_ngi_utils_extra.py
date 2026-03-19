"""Additional tests to cover uncovered branches in pynasonde.vipir.ngi.utils."""

import numpy as np
import pandas as pd
import pytest

from pynasonde.vipir.ngi.utils import (
    TimeZoneConversion,
    get_color_by_index,
    get_gridded_parameters,
    load_toml,
    smooth,
    to_namespace,
)


class TestTimeZoneConversionEdge:
    def test_explicit_tz_name_skips_geocode(self):
        """When lat/lon are None the explicit tz name must be used."""
        # Provide lat=None so the geocoder branch is skipped
        conv = TimeZoneConversion(local_tz="UTC", lat=None, long=None)
        assert "UTC" in conv.local_tz


class TestLoadTomlExplicitPath:
    def test_explicit_fpath(self, tmp_path):
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text('[meta]\ntitle = "test"\n')
        cfg = load_toml(str(toml_file))
        assert cfg.meta.title == "test"


class TestSmoothErrors:
    def test_ndim_not_1_raises(self):
        arr = np.ones((3, 3))
        with pytest.raises(ValueError, match="1 dimension"):
            smooth(arr)

    def test_size_lt_window_raises(self):
        arr = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="bigger than window size"):
            smooth(arr, window_len=5)

    def test_window_len_lt_3_returns_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = smooth(arr, window_len=2)
        assert np.array_equal(result, arr)

    def test_invalid_window_name_raises(self):
        arr = np.linspace(0, 1, 20)
        with pytest.raises(ValueError, match="Window is on"):
            smooth(arr, window="notawindow")

    def test_hanning_window(self):
        arr = np.linspace(0, 1, 20)
        result = smooth(arr, window_len=5, window="hanning")
        assert result.shape == arr.shape


class TestGetColorByIndex:
    def test_returns_rgba_tuple(self):
        color = get_color_by_index(0, 10)
        assert len(color) == 4

    def test_different_indices_give_different_colors(self):
        c0 = get_color_by_index(0, 10)
        c9 = get_color_by_index(9, 10)
        assert c0 != c9

    def test_custom_cmap(self):
        color = get_color_by_index(2, 5, cmap_name="plasma")
        assert len(color) == 4


class TestGetGriddedParameters:
    def _make_df(self):
        freqs = np.repeat([5.0, 6.0, 7.0], 3)
        heights = np.tile([100.0, 200.0, 300.0], 3)
        power = np.random.rand(9)
        return pd.DataFrame({"freq": freqs, "height": heights, "power": power})

    def test_returns_three_arrays(self):
        df = self._make_df()
        X, Y, Z = get_gridded_parameters(df, "freq", "height", "power")
        assert X.shape == Y.shape == Z.shape

    def test_no_rounding(self):
        df = self._make_df()
        X, Y, Z = get_gridded_parameters(df, "freq", "height", "power", rounding=False)
        assert X is not None

    def test_xparam_time_skips_x_rounding(self):
        times = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])
        df = pd.DataFrame(
            {
                "time": times,
                "height": [100.0, 200.0, 100.0, 200.0],
                "power": [1.0, 2.0, 3.0, 4.0],
            }
        )
        X, Y, Z = get_gridded_parameters(df, "time", "height", "power")
        assert Z.shape[0] > 0
