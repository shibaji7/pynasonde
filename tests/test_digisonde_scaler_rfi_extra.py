"""More RFI-filter tests: column clipping, count diagnostics and the adaptive amplitude gate.

``test_digisonde_scaler_rfi_filter.py`` covers the basic behaviour of each filter. These
tests add the edge cases (empty frames, boundaries, IQR clipping), the diagnostic plots
and :class:`AdaptiveAmpGate`, whose threshold detectors are checked on synthetic
two-population amplitude data with a known gap between the populations.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="needs pynasonde[digisonde-scaler]")
pytest.importorskip("joblib", reason="needs pynasonde[digisonde-scaler]")

from pynasonde.digisonde.scaler import rfi_filter
from pynasonde.digisonde.scaler.rfi_filter import (
    AdaptiveAmpGate,
    ColumnDensityFilter,
    NeighborPredictedFilter,
    VerticalRfiFilter,
)

DATA = Path(__file__).resolve().parents[1] / "examples/data"
RSF_FILES = sorted(str(p) for p in DATA.glob("WS833_*.RSF"))


def echoes(freqs, heights, mode="O", amp=20.0):
    n = len(freqs)
    return pd.DataFrame(
        {
            "frequency_mhz": freqs,
            "height_km": heights,
            "amplitude_db": [amp] * n,
            "mode": [mode] * n,
        }
    )


EMPTY = pd.DataFrame(columns=["frequency_mhz", "height_km", "amplitude_db", "mode"])


# ── NeighborPredictedFilter ───────────────────────────────────────────────────


def test_neighbor_filter_empty_input_is_returned_as_a_copy():
    out = NeighborPredictedFilter().clean(EMPTY)
    assert out.empty and out is not EMPTY


def test_neighbor_filter_keeps_a_smooth_trace_untouched():
    freqs = np.arange(1.0, 9.0, 0.5)
    df = echoes(freqs, 200.0 + 5.0 * freqs)
    out = NeighborPredictedFilter(freq_bin=0.5).clean(df)
    assert len(out) == len(df)
    assert (out["npf_h_pred_km"] > 0).all()


def test_neighbor_filter_judges_a_lone_bin_against_its_own_lowest_echo():
    # with no neighbours the prediction is the bin's own minimum height, so the
    # lowest echo is kept and a much higher one at the same frequency is removed
    out = NeighborPredictedFilter(freq_bin=1.0, min_neighbors=2).clean(
        echoes([3.0, 3.0], [100.0, 900.0])
    )
    assert out["height_km"].tolist() == [100.0]
    assert out["npf_h_pred_km"].tolist() == [100.0]


def test_neighbor_filter_tolerance_controls_what_is_removed():
    freqs = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0]
    heights = [200.0, 200.0, 200.0, 200.0, 200.0, 260.0]  # stray echo 60 km up
    df = echoes(freqs, heights)
    tight = NeighborPredictedFilter(tolerance_km=40, n_neighbors=2, freq_bin=1.0)
    loose = NeighborPredictedFilter(tolerance_km=100, n_neighbors=2, freq_bin=1.0)
    assert 260.0 not in tight.clean(df)["height_km"].tolist()
    assert 260.0 in loose.clean(df)["height_km"].tolist()


# ── ColumnDensityFilter ───────────────────────────────────────────────────────


def test_column_filter_empty_input():
    assert ColumnDensityFilter().clean(EMPTY).empty


def test_column_filter_flags_nothing_on_narrow_columns():
    df = echoes([1.0, 1.0, 2.0, 2.0], [200.0, 220.0, 210.0, 230.0])
    out = ColumnDensityFilter(freq_bin=1.0).clean(df)
    assert len(out) == 4
    assert not out["cdf_rfi_col"].any()


def test_column_filter_iqr_clip_keeps_the_trace_echoes():
    heights = [298.0, 299.0, 300.0, 300.0, 301.0, 302.0, 303.0, 304.0, 90.0, 700.0]
    df = echoes([4.0] * len(heights), heights)
    out = ColumnDensityFilter(rfi_span_km=150, freq_bin=1.0).clean(df)
    assert sorted(out["height_km"]) == sorted(heights[:8])  # tails clipped
    assert out["cdf_rfi_col"].all()  # survivors came from a flagged column


def test_column_filter_hard_deletes_sparse_flagged_columns():
    # only 3 echoes in the flagged column: too few for an IQR, so all are dropped
    df = echoes([4.0, 4.0, 4.0, 5.0], [100.0, 300.0, 500.0, 250.0])
    out = ColumnDensityFilter(rfi_span_km=150, freq_bin=1.0).clean(df)
    assert out["frequency_mhz"].tolist() == [5.0]


def test_column_filter_hard_delete_option():
    heights = [298.0, 300.0, 302.0, 304.0, 90.0, 700.0]
    df = echoes([4.0] * 6 + [5.0], heights + [250.0])
    out = ColumnDensityFilter(rfi_span_km=150, freq_bin=1.0, hard_delete=True).clean(df)
    assert out["frequency_mhz"].tolist() == [5.0]
    assert not out["cdf_rfi_col"].any()


# ── VerticalRfiFilter ─────────────────────────────────────────────────────────


def striped_frame():
    # 60 O-mode echoes at 4.0 MHz (an RFI stripe); a thin real trace elsewhere
    stripe = echoes([4.0] * 60, np.linspace(100, 600, 60))
    xmode = echoes([4.0] * 3, [300.0, 310.0, 320.0], mode="X")
    trace = echoes([2.0, 3.0, 5.0, 6.0], [200.0, 210.0, 230.0, 250.0])
    return pd.concat([stripe, xmode, trace], ignore_index=True)


def test_vertical_filter_removes_all_polarizations_at_flagged_bins():
    f = VerticalRfiFilter(max_echoes_per_bin=40, freq_bin=0.05)
    out = f.clean(striped_frame())
    assert 4.0 not in out["frequency_mhz"].round(2).tolist()
    assert len(out) == 4  # only the real trace survives, X-mode at 4 MHz included
    assert f.rfi_freqs_mhz_ == [pytest.approx(4.0)]
    assert f.o_counts_.max() == 60


def test_vertical_filter_without_rfi_returns_everything():
    f = VerticalRfiFilter(max_echoes_per_bin=100)
    df = striped_frame()
    assert len(f.clean(df)) == len(df)
    assert f.rfi_freqs_mhz_ == []


def test_vertical_filter_empty_input_and_missing_mode_column():
    f = VerticalRfiFilter()
    assert f.clean(EMPTY).empty and f.rfi_freqs_mhz_ == [] and f.o_counts_.empty
    no_mode = striped_frame().drop(columns="mode")
    out = VerticalRfiFilter(max_echoes_per_bin=40).clean(no_mode)
    assert 4.0 not in out["frequency_mhz"].round(2).tolist()  # all rows are counted


def test_vertical_filter_diagnostic_plot(tmp_path):
    import matplotlib.pyplot as plt

    f = VerticalRfiFilter(max_echoes_per_bin=40)
    blank = f.plot_diagnostic()  # nothing cleaned yet: placeholder figure
    assert blank.axes[0].texts and "call clean" in blank.axes[0].texts[0].get_text()
    plt.close(blank)

    f.clean(striped_frame())
    out = tmp_path / "diag" / "rfi.png"
    fig = f.plot_diagnostic(out_path=out, time_label="2023-10-14 00:00")
    assert out.exists() and out.stat().st_size > 0
    ax = fig.axes[0]
    assert "2023-10-14 00:00" in ax.get_title(loc="left")
    colors = {tuple(np.round(p.get_facecolor(), 2)) for p in ax.patches}
    assert len(colors) == 2  # flagged bins and clean bins are coloured differently
    plt.close(fig)


# ── AdaptiveAmpGate: threshold detectors on synthetic data ────────────────────


@pytest.fixture(scope="module")
def two_populations():
    rng = np.random.default_rng(7)
    noise = rng.normal(8.0, 2.0, 4000)
    echo = rng.normal(40.0, 4.0, 4000)
    return np.clip(np.concatenate([noise, echo]), 0.5, None)


@pytest.mark.parametrize("method", ["otsu", "gmm", "kernel"])
def test_detectors_place_the_threshold_between_the_populations(method, two_populations):
    gate = AdaptiveAmpGate(method=method)
    threshold = gate._detect(two_populations)
    # Otsu may sit just above the noise population, so the lower bound is loose
    assert 12.0 < threshold < 35.0


def test_detect_falls_back_when_data_is_too_sparse_or_method_unknown():
    gate = AdaptiveAmpGate(fallback=27.0)
    assert gate._detect(np.arange(10.0)) == 27.0
    assert gate._detect(np.array([np.nan] * 100)) == 27.0
    gate.method = "bogus"
    with pytest.raises(ValueError, match="Unknown method"):
        gate._detect(np.linspace(1, 50, 200))


def test_otsu_on_an_empty_histogram_uses_the_fallback():
    gate = AdaptiveAmpGate(fallback=11.0)
    assert gate._otsu(np.array([])) == 11.0


def test_gmm_equal_widths_returns_the_midpoint(monkeypatch):
    class FakeGaussianMixture:
        means_ = np.array([[10.0], [30.0]])
        covariances_ = np.array([[[4.0]], [[4.0]]])

        def __init__(self, **_):
            pass

        def fit(self, _x):
            return self

    import sklearn.mixture

    monkeypatch.setattr(sklearn.mixture, "GaussianMixture", FakeGaussianMixture)
    assert AdaptiveAmpGate(method="gmm")._gmm(np.zeros(100)) == pytest.approx(20.0)


def test_gmm_falls_back_to_otsu_when_the_fit_fails(monkeypatch, two_populations):
    import sklearn.mixture

    class Failing:
        def __init__(self, **_):
            pass

        def fit(self, _x):
            raise RuntimeError("did not converge")

    monkeypatch.setattr(sklearn.mixture, "GaussianMixture", Failing)
    gate = AdaptiveAmpGate(method="gmm")
    assert gate._gmm(two_populations) == gate._otsu(two_populations)


def test_kernel_with_a_single_population_falls_back_to_otsu():
    rng = np.random.default_rng(3)
    unimodal = np.clip(rng.normal(30.0, 3.0, 2000), 1, None)
    gate = AdaptiveAmpGate(method="kernel")
    assert gate._kernel(unimodal) == gate._otsu(unimodal)


# ── AdaptiveAmpGate: windows, names and file loading ──────────────────────────


def test_valid_window_is_the_longest_stable_run():
    t = [dt.datetime(2023, 10, 14, h) for h in range(6)]
    window = AdaptiveAmpGate._valid_window
    # stable, drift, then a longer stable run
    assert window(t, np.array([20, 50, 20, 21, 20, 22.0]), 20.0) == (t[2], t[5])
    # the first of two equal runs wins
    assert window(t[:5], np.array([20, 20, 50, 20, 20.0]), 20.0) == (t[0], t[1])
    # a run that reaches the end is still found
    assert window(t[:4], np.array([50, 50, 20, 20.0]), 20.0) == (t[2], t[3])


def test_valid_window_edge_cases():
    t = [dt.datetime(2023, 10, 14, h) for h in range(3)]
    window = AdaptiveAmpGate._valid_window
    assert window(t, np.array([20.0, 21.0, 19.0]), 20.0) == (t[0], t[-1])
    # NaN thresholds (sparse files) do not shrink the window
    assert window(t, np.array([np.nan, np.nan, np.nan]), 20.0) == (t[0], t[-1])
    # nothing is stable: report the full range rather than nothing
    assert window(t, np.array([60.0, 70.0, 80.0]), 20.0) == (t[0], t[-1])


def test_time_from_name_parses_digisonde_timestamps():
    got = AdaptiveAmpGate._time_from_name("/x/WS833_2023287151418.RSF")
    assert got == dt.datetime(2023, 10, 14, 15, 14, 18)  # day 287 of 2023
    assert AdaptiveAmpGate._time_from_name("nonsense.RSF") == dt.datetime(1970, 1, 1)


def test_load_one_reads_a_real_rsf_file():
    rec = AdaptiveAmpGate._load_one(RSF_FILES[0], "amplitude_db")
    assert isinstance(rec["time"], dt.datetime)
    assert rec["amps"].dtype == float and len(rec["amps"]) > 0


def test_load_one_rejects_unsupported_missing_and_wrong_column(tmp_path):
    load = AdaptiveAmpGate._load_one
    assert load(str(tmp_path / "a.txt"), "amplitude_db") is None
    assert load(str(tmp_path / "missing.RSF"), "amplitude_db") is None
    assert load(RSF_FILES[0], "no_such_column") is None


@pytest.mark.parametrize(
    "ext, module, cls",
    [
        (".SBF", "pynasonde.digisonde.parsers.sbf", "SbfExtractor"),
        (".MMM", "pynasonde.digisonde.parsers.mmm", "ModMaxExtractor"),
    ],
)
def test_load_one_sbf_and_mmm_branches(monkeypatch, tmp_path, ext, module, cls):
    import importlib

    class FakeExtractor:
        date = None  # forces the filename fallback for the timestamp

        def __init__(self, path, extract_time_from_name=False):
            self.path = path

        def extract(self):
            pass

        def to_pandas(self):
            return pd.DataFrame({"amplitude_db": [5.0, 6.0, np.nan]})

    monkeypatch.setattr(importlib.import_module(module), cls, FakeExtractor)
    monkeypatch.setattr(rfi_filter, "normalize_echo_dataframe", lambda df, **_: df)
    rec = AdaptiveAmpGate._load_one(
        str(tmp_path / f"WS833_2023287010203{ext}"), "amplitude_db"
    )
    assert rec["time"] == dt.datetime(2023, 10, 14, 1, 2, 3)
    assert rec["amps"].tolist() == [5.0, 6.0]  # NaN dropped


def test_load_one_empty_frame_is_skipped(monkeypatch, tmp_path):
    import pynasonde.digisonde.parsers.sbf as sbf

    class Empty:
        date = None

        def __init__(self, *_a, **_k):
            pass

        def extract(self):
            pass

        def to_pandas(self):
            return pd.DataFrame()

    monkeypatch.setattr(sbf, "SbfExtractor", Empty)
    monkeypatch.setattr(rfi_filter, "normalize_echo_dataframe", lambda df, **_: df)
    assert (
        AdaptiveAmpGate._load_one(str(tmp_path / "a_2023287000000.SBF"), "amplitude_db")
        is None
    )


# ── AdaptiveAmpGate.fit ───────────────────────────────────────────────────────


def test_fit_with_no_files_or_no_loadable_files_uses_the_fallback(tmp_path):
    gate = AdaptiveAmpGate(n_jobs=1, fallback=22.0)
    assert gate.fit([]).threshold == 22.0
    assert gate.fit([str(tmp_path / "nothing.RSF")]).threshold == 22.0
    assert gate.time_window is None


def test_fit_on_the_bundled_rsf_files(tmp_path):
    gate = AdaptiveAmpGate(n_jobs=1).fit(RSF_FILES)
    assert gate.amp_lo <= gate.threshold <= gate.amp_hi
    assert len(gate.per_file_df) == len(RSF_FILES)
    assert list(gate.per_file_df.columns) == ["time", "n_echoes", "threshold"]
    assert gate.per_file_df["time"].is_monotonic_increasing
    start, end = gate.time_window
    assert start <= end
    assert (gate.pooled_amps_ > 0).all()

    out = tmp_path / "gate.png"
    gate.plot_diagnostic(out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_fit_out_of_bounds_detection_uses_the_fallback():
    gate = AdaptiveAmpGate(n_jobs=1, amp_lo=0.0, amp_hi=0.1, fallback=27.0)
    gate.fit(RSF_FILES)
    assert gate.threshold == 27.0


def test_plot_diagnostic_before_fit_does_nothing(tmp_path):
    out = tmp_path / "never.png"
    assert AdaptiveAmpGate().plot_diagnostic(out_path=out) is None
    assert not out.exists()
