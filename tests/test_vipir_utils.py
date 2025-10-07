"""Tests for VIPIR utility helpers."""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz

from pynasonde.vipir.ngi.utils import (
    TimeZoneConversion,
    load_toml,
    remove_outliers,
    running_median,
    setsize,
    smooth,
    to_local_time,
)


def test_load_toml_returns_namespace():
    cfg = load_toml()

    assert cfg.title == "Pynasonde"
    assert cfg.ngi.scaler.noise_constant == 2


def test_timezone_conversion_difference():
    converter = TimeZoneConversion(lat=35.0, long=-106.5)
    utc_time = dt.datetime(2024, 4, 8, 16, 0, 0)
    local_times = converter.utc_to_local_time([utc_time])

    assert len(local_times) == 1
    assert abs((utc_time - local_times[0]).total_seconds()) >= 4 * 3600


def test_remove_outliers_and_running_stats():
    df = pd.DataFrame({"value": [1, 2, 3, 100, 4, 5]})
    filtered = remove_outliers(df, "value", quantiles=[0.1, 0.9])
    assert filtered["value"].max() < 100

    med = running_median([1, 10, 1], window=2)
    assert len(med) == 3


def test_smooth_and_setsize(monkeypatch):
    arr = np.linspace(0, 1, 20)
    smoothed = smooth(arr, window_len=5, window="flat")
    assert smoothed.shape[0] == arr.shape[0]

    monkeypatch.setattr(plt.style, "use", lambda *_a, **_k: None)
    setsize(size=6)


def test_to_local_time_helper():
    tz1 = pytz.timezone("UTC")
    tz2 = pytz.timezone("US/Mountain")
    ts = [dt.datetime(2024, 4, 8, 12, 0, 0)]
    converted = to_local_time(ts, tz1, tz2)
    assert converted[0] != ts[0]
